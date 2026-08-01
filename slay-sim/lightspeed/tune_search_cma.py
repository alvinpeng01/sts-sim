"""CMA-ES tuning of expectimax_search's native heuristic/MCTS constants
(sts.get_search_params()/set_search_params(), see slaythespire.cpp's
TunableParams/g_params) -- the practical answer to "use RL to train
weights for expectimax," as opposed to distilling search into a neural
network (train_distillation_expectimax.py, a separate and complementary
effort). CMA-ES (evolution strategies) is the natural fit for this specific
problem: ~15 scalar constants, a non-differentiable objective (win rate /
HP over real episodes, not a loss with gradients), and an expensive-to-
evaluate fitness (each candidate needs real rollouts across several
encounters) -- exactly the regime ES was designed for, and literally
interchangeable with policy-gradient RL for small, expensive, black-box
optimization problems like this one (see OpenAI's 2017 ES paper).

Search space: each parameter is optimized as a MULTIPLICATIVE factor around
its current default (x_i = raw_i / default_i, CMA-ES searches x starting at
all-1.0s) rather than in raw units -- the raw constants span wildly
different scales (skill_danger_scale=30.0 vs attack_block_penalty_scale=
0.15 vs wc_chance=1.0), and a single isotropic CMA-ES step size doesn't
make sense applied directly to all of them. Working in relative-factor
space sidesteps that without needing per-parameter step-size tuning.

THREAD/PROCESS SAFETY: every candidate is evaluated in its own worker
PROCESS (multiprocessing, not threads) -- see set_search_params's own
docstring for why: g_params is unlocked global mutable state, so two
candidates with DIFFERENT parameter values must never be "in flight"
concurrently within the same process. Separate processes each get their
own independent copy of the C++ module's global state, sidestepping the
race entirely (the same reasoning ppo.py/distillation.py already use
separate processes for their own worker pools, just for a different
reason there -- CPU parallelism, not a correctness requirement).

Run:  PYTHONPATH=".;../sts_lightspeed/build" python -m lightspeed.tune_search_cma
"""

from __future__ import annotations
from .paths import native_build_path

import argparse
import json
import multiprocessing as mp
import time

import cma
import numpy as np

TIME_BUDGET_SECONDS = 8 * 60 * 60
N_EPISODES_PER_ENCOUNTER = 20  # doubled from the first 20-min run -- much bigger budget now, worth
                                # spending some of it on a cleaner (less noisy) fitness signal per
                                # candidate rather than only on more generations
SIMS = 150
# Broadened from the first two runs' 6 -- CENTURION_AND_HEALER/AUTOMATON/CHAMP add mechanic
# diversity (heal-support, block-heavy, buff-then-burst) not represented before, and COLLECTOR
# is a known-weak matchup (spawns adds, flagged earlier this session, see the eval log discussion)
# worth giving the tuner a direct shot at. AWAKENED_ONE/REPTOMANCER deliberately still excluded --
# kept as the held-out OOD generalization check (see validate_tuned_params.py), not folded into
# the training set now that it's grown.
# Composition deliberately mirrors a REAL run's encounter mix (~4-5 basic, 2 elite, 1 boss per
# act), not a boss-heavy sample. The previous set was 7/10 bosses, which is close to inverted:
# bosses are ~15% of a real run's fights, and they are the ONLY fights followed by the act-
# transition heal, so HP preserved there is largely refunded. Measured consequence at A20 on
# matched decks -- our paired HP deficit vs Silverbot is -7.1pp on NON-boss fights but only -3.4pp
# on bosses, i.e. the deficit is concentrated exactly in the fights the old set under-weighted.
# Non-boss fights are also where the deficit is cleanly attackable: all three non-boss encounters
# measured 30/30 for BOTH engines, so there is no win-rate/HP tradeoff to block progress there --
# GREMLIN_NOB alone gives up 13.3pp of HP with no compensating benefit whatsoever.
ENCOUNTERS = [
    # basic (8) -- the bulk of a real run
    "JAW_WORM", "TWO_LOUSE", "GREMLIN_GANG", "EXORDIUM_THUGS",
    "CHOSEN", "SHELLED_PARASITE_AND_FUNGI",
    "THREE_DARKLINGS", "ORB_WALKER",
    # elite (4) -- fought several times per run, no heal after
    "GREMLIN_NOB", "THREE_SENTRIES", "CENTURION_AND_HEALER", "SPHERIC_GUARDIAN",
    # boss (3) -- one per act, kept for signal on the hardest fights
    "THE_GUARDIAN", "AUTOMATON", "TIME_EATER",
]
# 2.0, raised from 1.0: with the rebalanced (mostly non-boss) ENCOUNTERS below, most fights are won
# by both engines essentially always, so the win term saturates and carries little gradient -- HP is
# the quantity actually in contention there, and it is the quantity that compounds across a real
# run's ~20 fights since only bosses are followed by a heal. A loss still scores 0 against a win's
# 1.0 + 2.0*hp_frac, so wins remain worth far more than any HP margin; this shifts emphasis toward
# HP without making the search willing to lose fights for it.
HP_FITNESS_WEIGHT = 2.0  # see _evaluate_candidate's own comment
# Ascension the tuner trains at. 20, not IroncladFightEnv's own default of 0: A20 is what's
# actually played, and it is a materially different optimization problem -- measured on identical
# decks/seeds, win rate falls 86.0% -> 73.0%, and at A0 five of the ten training encounters sit at
# 100% win rate, contributing no win-rate gradient at all. A20 both matches deployment and gives
# the tuner a more discriminating training signal.
TUNE_ASCENSION = 20
# Sequential-halving root allocation (slaythespire.cpp's nativeRunMctsSearchSeqHalving) instead of
# plain UCB1 over the whole budget. A/B'd as a drop-in on UCB1-fitted params at A20 on matched
# decks: 71.3% vs 70.3% win and +1.2pp paired HP -- both modest and near the noise floor, but
# positive on the harder side of the comparison (its competitor's tuned parameters). Tuning WITH it
# enabled is what tests whether the gain is real once params can adapt to it. Recorded in
# _fitness_config so scores measured under different selectors are never compared.
USE_SEQ_HALVING = True
SIGMA0 = 0.15  # tighter than the first run's 0.3 -- this run WARM-STARTS from that run's own best
               # point (see main()) rather than the original hand-tuned defaults, so it should
               # refine a neighborhood around an already-good solution, not re-explore broadly
N_WORKERS = 12
PREV_TUNED_PATH = "lightspeed/tuned_search_params.json"  # warm-start source, see main()

PARAM_NAMES = [
    "c_ucb", "c_ucb_chance", "wc_chance", "wa_chance",
    "loss_progress_credit_weight", "brewing_threat_estimate",
    "attack_base", "attack_finish_off_scale", "attack_block_penalty_scale",
    "aoe_bonus", "skill_base", "skill_danger_scale", "skill_haste_penalty",
    "power_score", "end_turn_time_warp_risk_score", "skill_haste_danger_threshold",
    "per_card_weight_scale",
    # A normalized version of Silver Automaton's hand-crafted combat card order,
    # added after passing two independent paired A20 gates at weight 1.0. Its
    # compiled default is the additive off-state, so CMA-ES tunes raw units.
    # PUCT-style prior bonus added to nativeSelectIdx on top of UCB1 (see slaythespire.cpp's
    # g_params.cPuct comment). c_puct's natural off-state is 0.0 (additive, like
    # per_card_weight_scale) -- puct_temperature only matters once c_puct is nonzero, so it's a
    # normal multiplicative param around its 10.0 default.
    "c_puct", "puct_temperature",
    # Five heuristic terms that exist in C++ but had never been in this search space, so they sat
    # at their 0.0 compiled defaults and the features were dead in every tuned config shipped so
    # far. All additive: their off-state IS 0.0, and the multiplicative scheme cannot turn a zero
    # on. Two of them guard tables that the card-data audit corrected, which is what surfaced
    # this -- fixing nativeImmediateBlockBase's 4 wrong values and 16 omissions changes nothing
    # while direct_block_score_weight is 0.0, and the same is true of the four Vulnerable
    # appliers (Beam Cell, Crush Joints, Indignation, Trip) added to isVulnerableApplier.
    "direct_block_score_weight", "vulnerable_apply_bonus",
    "weak_apply_bonus",
    "power_per_turn_value_weight", "power_immediate_value_weight",
    # policy_net_weight deliberately EXCLUDED from tuning (and load_policy_net dropped from
    # _worker_init below): a clean, isolated held-out ablation (net loaded vs not, everything
    # else -- including c_puct -- held fixed) found ~7x slower search for a ~1.7pp win-rate
    # bump that's within noise at n=300 and no HP-efficiency benefit. Not worth its cost,
    # especially working directly against wanting more simulations/decision headroom. Revisit
    # only if the net is retrained on more data or the cost is reduced further.
    # Extra weight on final HP within the search's own WIN-branch terminal reward (see
    # slaythespire.cpp's nativeExpectimaxTerminalReward comment) -- found via direct matched-
    # simulation-count comparison against Silver Automaton's own engine that our win score's HP
    # term was a much smaller PROPORTION of the total than theirs, and a hand-swept value of 4.0
    # roughly halved the measured HP-efficiency gap. Compiled default is a real, meaningful 1.0
    # (not an off-state), so this fits the normal unsigned_mult scheme directly -- x0=4.0 warm-
    # starts CMA-ES at the swept value rather than re-discovering it from scratch.
    "win_hp_weight",
    # "We have enough block" gate params (see HeuristicContext::blockSufficient's own comment) --
    # a direct A/B sweep at their compiled defaults (margin=4.0, penalty=8.0) found no clear
    # effect in isolation, but CMA-ES exploring them jointly with everything else (especially now
    # that win_hp_weight has changed the reward landscape they interact with) might still find a
    # real combination the isolated sweep couldn't see. Both have real, nonzero compiled defaults,
    # so both fit the normal unsigned_mult scheme with no override needed.
    "block_sufficiency_margin", "defensive_card_suppression_penalty",
    # Potions still held at fight end, ported from Silver Automaton's own evaluateEndState
    # (flat count * weight, not per-potion-type -- see nativeExpectimaxTerminalReward's own
    # comment). Natural off-state is 0.0, additive like per_card_weight_scale/c_puct.
    "potion_score_weight",
    # Small terminal penalty for energy left unspent. Values around 0.5 improved
    # shared-win ending HP in repeated A20 sweeps, but the win delta varied by seed
    # set, so this remains a tuner dimension rather than an active manual override.
    # The compiled default is the additive off-state.
    "energy_waste_weight",
    # Enemy block is an additive-off reward term.  It is most relevant to
    # Spheric Guardian and other block-heavy fights, so tune it jointly with
    # the terminal/heuristic weights rather than hard-coding a sweep winner.
    "enemy_block_weight",
    # How much credit the search gives a boss-fight win for the real STS act-transition heal
    # (0=ignore it entirely, matching our current single-fight eval methodology; 1=full credit,
    # matching Silver Automaton's own evaluateEndState and what's correct for real full-run play)
    # -- see slaythespire.cpp's nativeExpectimaxTerminalReward comment. A TUNABLE blend rather
    # than the hard on/off tried and reverted earlier (fully-on regressed measured HP-efficiency
    # -16 to -23pp on every boss encounter) -- letting CMA-ES weigh it against everything else,
    # including potion_score_weight, empirically instead of a hand-picked all-or-nothing choice.
    "boss_heal_credit_weight",
    # Multiplier on the search's win-branch per-turn penalty (see slaythespire.cpp's
    # g_params.winTurnPenaltyWeight comment). Previously a hardcoded constexpr with no way to
    # rebalance it -- win_hp_weight (~4) scaled the win score's HP term ~4x without touching the
    # turn term, so the finish-fast-vs-preserve-HP tradeoff needs re-tuning. Real nonzero compiled
    # default (1.0 = the original constant), so normal unsigned_mult.
    "win_turn_penalty_weight",
    # Per-still-standing-monster penalty on a LOSS, ported from Silver Automaton's own aliveScore
    # (see slaythespire.cpp's g_params.aliveMonsterPenaltyWeight comment). Natural off-state 0.0,
    # additive. Only bites in multi-monster encounters (DONU_AND_DECA/COLLECTOR/
    # CENTURION_AND_HEALER here); a constant offset in single-monster ones.
    "alive_monster_penalty_weight",
]

# Three parameterization kinds, since a single "x is a multiplicative factor
# around the default" scheme (x0=1.0 for everything) doesn't fit every
# parameter here:
#   - "unsigned_mult" (most params): raw = max(0.01, x) * default -- x
#     searched around 1.0, always positive.
#   - "signed_mult" (attack_finish_off_scale): raw = x * default -- x can go
#     NEGATIVE. This is what gives the tuner a genuine target-SELECTION-
#     preference knob for free, no new C++ parameter needed: the raw
#     formula is `score += (1 - target_hp_fraction) * attack_finish_off_scale`,
#     which with the default's positive sign always prefers LOW-hp targets
#     (finish them off). Every OTHER tunable param is a strictly-positive
#     factor on an already-positive default, so this term could previously
#     only be scaled toward zero, never REVERSED -- "prefer the tankiest/
#     highest-HP target instead" (sometimes correct in multi-monster
#     fights) was structurally unreachable. Letting x go negative removes
#     that blind spot.
#   - "additive" (per_card_weight_scale): raw = x DIRECTLY, not multiplied
#     by the default at all. Its natural default is 0.0 (off) -- multiplying
#     ANYTHING by zero is always zero, so the multiplicative scheme cannot
#     express turning this parameter on at all. Searched in raw units
#     instead, starting at 0.0 (matching the current off default), bounded
#     to ADDITIVE_BOUNDS below -- a range picked to let the per-card bonus
#     (pick rates run 0.05-0.89) meaningfully compete with the existing
#     ~8-14 type-level scores without swamping them.
PARAM_KIND = {
    "attack_finish_off_scale": "signed_mult",
    "per_card_weight_scale": "additive",
    "c_puct": "additive",
    "direct_block_score_weight": "additive",
    "vulnerable_apply_bonus": "additive",
    "weak_apply_bonus": "additive",
    "power_per_turn_value_weight": "additive",
    "power_immediate_value_weight": "additive",
    "policy_net_weight": "additive",
    "potion_score_weight": "additive",
    "energy_waste_weight": "additive",
    "enemy_block_weight": "additive",
    "boss_heal_credit_weight": "additive",
    "alive_monster_penalty_weight": "additive",
}
ADDITIVE_BOUNDS = {
    "per_card_weight_scale": (-5.0, 25.0),
    # Small values complement the reward-screen pick-rate prior; values >=2
    # regressed in the first A20 screen, so keep the joint search focused.
    # Non-negative only: a negative PUCT weight would penalize actions the heuristic prior favors,
    # inverting rather than sharpening it -- not a meaningful direction to explore. Upper bound
    # picked so the prior term can plausibly dominate the existing UCB1 term (which is O(c_ucb),
    # c_ucb's own tuned value has run ~1.5-7) at low visit counts without dwarfing it by orders of
    # magnitude.
    "c_puct": (0.0, 8.0),
    # Same additive-bonus philosophy as c_puct, scaled to nativeScoreAction's existing per-action
    # heuristic terms (attackBase/skillBase/powerScore/aoeBonus/etc. are all roughly 4-10 by
    # default) rather than to c_puct's own UCB1-relative scale -- allowed negative since the net
    # could legitimately learn to prefer actions the hand-tuned heuristic scores highly for the
    # wrong reason, same rationale as attack_finish_off_scale's sign flip.
    "policy_net_weight": (-10.0, 10.0),
    # Silver Automaton's own potionWeight is 11.0 against a total win score of ~183 (winBonus 53
    # + curHp ~130) -- roughly 6% of the win score at a typical single held potion. Our own total
    # win score is on a different scale now (win_hp_weight=4 makes it NATIVE_W_WIN(200) +
    # 4*curHp(~130) =~720), so their raw magnitude doesn't transfer -- bounded generously (0 to
    # 50) and let CMA-ES find the right scale rather than assuming proportional transfer holds
    # here the way it did for win_hp_weight (that one was validated by direct sweep, this isn't
    # yet). Non-negative only: a potion is never worth LESS than not having it.
    "potion_score_weight": (0.0, 50.0),
    # Stronger hand sweeps began trading wins for HP; leave room for joint tuning
    # without allowing energy waste to dominate the terminal score.
    "energy_waste_weight": (0.0, 5.0),
    # The focused calibrated screen peaks near 1.0; leave room on either
    # side without allowing temporary block to dominate real monster HP.
    "enemy_block_weight": (0.0, 3.0),
    # A blend fraction by construction (0=raw curHp, 1=full heal credit) -- bounds outside [0,1]
    # have no clean interpretation, unlike every other additive param here.
    "boss_heal_credit_weight": (0.0, 1.0),
    # Silver Automaton's own tuned aliveWeight is 3.4 against a ~183-scale win score and a loss
    # branch whose dominant term (monsterDamageWeight) is 37. Our loss branch is scaled very
    # differently (lossProgressCreditWeight's raw default is 150, currently tuned to ~3.4x that),
    # so their magnitude doesn't transfer directly -- bounded generously and left to CMA-ES, same
    # reasoning as potion_score_weight. Non-negative: a monster still standing is never GOOD.
    "alive_monster_penalty_weight": (0.0, 120.0),
    # Multiplies min(unblocked, actualBlock) -- a quantity that already runs 5-30 in a dangerous
    # turn, so unlike the flat bonuses below this wants a small factor, not a raw score. Upper
    # bound set so a full-value block card can reach roughly twice power_score (~14) at the top
    # of the range without swamping every other term.
    "direct_block_score_weight": (0.0, 2.0),
    # Flat per-application bonuses, so bounded on the same scale as the existing type-level
    # scores they compete with (attack_base ~3, skill_base ~2, aoe_bonus ~10, power_score ~14).
    # Non-negative both: applying a debuff a target lacks is never worse than not applying it.
    "vulnerable_apply_bonus": (0.0, 15.0),
    "weak_apply_bonus": (0.0, 15.0),
    # Multiply nativePowerPerTurnValue (1-7) x monsterHpRatio (0-1) and nativePowerImmediateValue
    # (0-3) respectively, so the products land in roughly 0-7 and 0-3 before weighting. Bounded to
    # let a Demon Form at full enemy HP plausibly outscore power_score's flat ~14 without letting
    # any Power dominate an urgently-needed block card.
    "power_per_turn_value_weight": (0.0, 5.0),
    "power_immediate_value_weight": (0.0, 5.0),
}


def _param_kind(name: str) -> str:
    return PARAM_KIND.get(name, "unsigned_mult")


def raw_value(name: str, x_i: float, default_i: float) -> float:
    kind = _param_kind(name)
    if kind == "additive":
        return x_i
    if kind == "signed_mult":
        return x_i * default_i
    return max(0.01, x_i) * default_i

OUT_PATH = "lightspeed/tuned_search_params.json"
LOG_PATH = "lightspeed/tune_search_cma_progress.log"


def _fitness_config() -> dict:
    """Identifies the objective a saved score was measured under -- see the warm-start guard in
    main() for why scores across differing configs must not be compared."""
    # Deck/HP/relic calibration is part of the objective just as much as the encounter list is --
    # shrinking Act 1's decks changes what a given score MEANS, so it must invalidate a stale score
    # bar the same way a changed encounter list does. Imported lazily because env.py pulls in the
    # native module, which main() only puts on sys.path at startup.
    from lightspeed.env import ACT_TIER_RESOURCES
    resources = {f"{act}/{tier}": list(vals) for (act, tier), vals in sorted(ACT_TIER_RESOURCES.items())}
    return {"hp_fitness_weight": HP_FITNESS_WEIGHT, "ascension": TUNE_ASCENSION,
            "sims": SIMS, "episodes_per_encounter": N_EPISODES_PER_ENCOUNTER,
            "seq_halving": USE_SEQ_HALVING, "encounters": ENCOUNTERS,
            "deck_resources": resources}


def _log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# --- worker-side: each candidate evaluated in its own process ---------------

_worker_envs = None

# Per-encounter (player_hp, extra_deck_cards, upgrade_chance, starter_removals,
# relic_count, n_boss_relics), keyed by real Act/tier via ACT_TIER_RESOURCES --
# NOT one flat (hp=130, extra_cards=30) for every encounter. That flat scheme
# (matching fair_deck.json's own generation params) was carried over from the
# single-fixed-deck era and quietly miscalibrated most of ENCOUNTERS: it's a
# reasonable approximation for TIME_EATER/DONU_AND_DECA (real Act3-boss tier
# is hp=130/extra_cards=40, close-ish) but badly wrong for e.g. GREMLIN_NOB/
# HEXAGHOST/THE_GUARDIAN (real Act1 tier is hp=70-85/extra_cards=20-25/
# upgrade_chance=0.3 -- these were being tested with an Act3-boss-sized,
# half-upgraded deck against an Act1 monster), which fully explains their
# suspiciously easy 100% win rates all session. build_full_encounter_resources
# is the SAME calibration the real production training scripts already use.
def _worker_init() -> None:
    global _worker_envs
    import slaythespire as sts
    from lightspeed.env import IroncladFightEnv, build_full_encounter_resources
    from lightspeed.cards import weighted_ironclad_deck

    # No load_policy_net call here (deliberately) -- see PARAM_NAMES's own comment on why
    # policy_net_weight was pulled from tuning: with no net loaded, nativePolicyNetScore always
    # returns 0.0, so leaving g_policyNet unloaded is what keeps policy_net_weight a guaranteed
    # no-op in g_params if it's ever left in a stale saved params file.
    sts.set_seq_halving(USE_SEQ_HALVING)

    encounter_resources = build_full_encounter_resources()
    _worker_envs = {}
    for enc_idx, enc_name in enumerate(ENCOUNTERS):
        enc = getattr(sts.MonsterEncounter, enc_name)
        _worker_envs[enc_name] = IroncladFightEnv(
            encounter=enc, encounter_resources=encounter_resources,
            deck_generator=weighted_ironclad_deck, ascension=TUNE_ASCENSION,
        )


def _evaluate_candidate(args) -> float:
    """Returns NEGATIVE mean per-episode score (CMA-ES minimizes). Score
    per episode: 1.0 for a win, plus a small HP-remaining bonus (0 to 0.3)
    on wins so the search doesn't stop caring about margin once a fight is
    already winnable most of the time -- matches the HP-efficiency gap
    this session's own Silverbot comparison surfaced (we were winning some
    fights at way lower HP than Silverbot even at 100% win rate)."""
    x, defaults, seed_base = args
    import slaythespire as sts

    params = {name: raw_value(name, x[i], defaults[name]) for i, name in enumerate(PARAM_NAMES)}
    sts.set_search_params(params)

    total_score = 0.0
    total_n = 0
    for enc_idx, enc_name in enumerate(ENCOUNTERS):
        env = _worker_envs[enc_name]
        for seed in range(N_EPISODES_PER_ENCOUNTER):
            obs = env.reset(seed=seed_base + seed)
            done = False
            steps = 0
            info = None
            while not done and steps < 150:
                # Pair the search's stochastic samples as well as the game
                # seed across every CMA candidate in this generation.
                search_seed = ((seed_base + seed) << 32) ^ (enc_idx << 16) ^ steps
                action, _ = sts.run_mcts_search(env.bc, SIMS, None, search_seed)
                obs, reward, done, info = env.step(action)
                steps += 1
            won = info["outcome"] == sts.BattleOutcome.PLAYER_VICTORY
            # Normalized against THIS episode's own max_hp (varies per encounter tier now,
            # not one flat 130 for everything -- see _worker_init's encounter_resources).
            #
            # HP_FITNESS_WEIGHT is 1.0, not the original 0.3: at 0.3 a full 0->100% HP swing was
            # worth 0.3 while a single extra win was worth 1.0, so the optimizer correctly (given
            # what it was told) traded HP away for marginal win rate. Measured consequence at A20
            # against Silver Automaton on matched decks/seeds: we won MORE fights (+3.0pp) while
            # ending them 8.7pp lower on HP, including 13.3pp lower on GREMLIN_NOB, a fight BOTH
            # engines win 30/30 -- pure waste with no win-rate compensation. HP is a resource that
            # carries across a real run's ~15-20 combats, so per-fight HP and per-fight win rate
            # are worth roughly comparable amounts; 1.0 encodes that.
            score = 1.0 + (HP_FITNESS_WEIGHT * info["player_hp"] / env.bc.player_max_hp if won else 0.0) if won else 0.0
            total_score += score
            total_n += 1
    return -(total_score / total_n)


def main():
    global TIME_BUDGET_SECONDS, N_WORKERS, OUT_PATH, LOG_PATH, PREV_TUNED_PATH
    parser = argparse.ArgumentParser(
        description="CMA-ES tune native MCTS weights with isolated artifacts.")
    parser.add_argument("--minutes", type=float,
                        default=TIME_BUDGET_SECONDS / 60.0,
                        help="wall-clock budget; default preserves the long-run setting")
    parser.add_argument("--workers", type=int, default=N_WORKERS,
                        help="separate candidate processes; do not exceed physical cores")
    parser.add_argument("--out", default=OUT_PATH,
                        help="candidate JSON artifact; never use the active config for a smoke run")
    parser.add_argument("--log", default=LOG_PATH)
    parser.add_argument("--warm-start", default=PREV_TUNED_PATH)
    args = parser.parse_args()
    if args.minutes <= 0 or args.workers < 1:
        raise ValueError("--minutes and --workers must be positive")
    TIME_BUDGET_SECONDS = args.minutes * 60.0
    N_WORKERS = args.workers
    OUT_PATH = args.out
    LOG_PATH = args.log
    PREV_TUNED_PATH = args.warm_start
    import sys
    sys.path.insert(0, native_build_path())
    import slaythespire as sts

    defaults = sts.get_search_params()  # the ORIGINAL hand-tuned C++ defaults -- x-space is always relative to THESE, never to the previous run's result, so a warm-started x0 stays on the same scale as PARAM_NAMES/bounds always assumed
    default_vec = np.array([defaults[name] for name in PARAM_NAMES])

    # Warm-start from the previous tuning run's own best point, if present,
    # instead of always restarting from x=[1.0]*n (the original hand-tuned
    # defaults) -- this run continues refining that result rather than
    # re-discovering the same improvement from scratch. Bounds widened vs
    # the first run's [0.1, 5.0]: that run's own c_ucb finding (~6.99,
    # x~4.66) was already close to the old upper bound, and score was still
    # climbing at generation 61 with no clear plateau -- give this longer
    # run room to explore past where the first one was artificially capped.
    def _x0_for(name, prev):
        kind = _param_kind(name)
        default_i = defaults[name]
        if name not in prev:
            # additive params' natural "off" state is x=0.0, not the
            # multiplicative schemes' x=1.0 -- there's nothing to warm-start
            # a brand-new param from either way, but starting an additive
            # param at 1.0 would mean "immediately on with weight 1.0",
            # not "off", which isn't the intended fresh-start behavior.
            return 0.0 if kind == "additive" else 1.0
        if kind == "additive":
            return prev[name]
        return max(0.02, min(9.9, prev[name] / default_i))

    prev_score = None
    prev_params = None
    try:
        with open(PREV_TUNED_PATH) as f:
            prev_data = json.load(f)
        prev = prev_data["params"]
        prev_params = prev
        prev_score = prev_data.get("score")
        # A saved score is only a meaningful bar for "did this run improve?" if it was produced by
        # the SAME objective. Changing HP_FITNESS_WEIGHT or TUNE_ASCENSION rescales the objective
        # (e.g. raising the HP term's weight lifts every score), so carrying an old score forward
        # as the save bar would either block all saves or let a worse point overwrite a better one.
        # On a config change, keep warm-starting the PARAMS (still a good starting point) but drop
        # the score bar so this run re-establishes its own baseline.
        saved_cfg = prev_data.get("fitness_config")
        if saved_cfg != _fitness_config():
            _log(f"fitness config changed ({saved_cfg} -> {_fitness_config()}); warm-starting "
                 f"params but DISCARDING the prior score {prev_score} as a save bar -- scores "
                 f"across different objectives are not comparable")
            prev_score = None
        x0 = [_x0_for(name, prev) for name in PARAM_NAMES]
        _log(f"warm-starting from {PREV_TUNED_PATH} (prior score {prev_data['score']:.3f}); "
             f"new params not in that file start fresh: "
             f"{[n for n in PARAM_NAMES if n not in prev]}")
    except FileNotFoundError:
        x0 = [_x0_for(name, {}) for name in PARAM_NAMES]
        _log(f"no {PREV_TUNED_PATH} found -- starting fresh from the hand-tuned defaults")

    _log("=== CMA-ES search-parameter tuning (continuation run) ===")
    _log(f"encounters={ENCOUNTERS}, episodes/encounter={N_EPISODES_PER_ENCOUNTER}, sims={SIMS}, "
         f"n_workers={N_WORKERS}, sigma0={SIGMA0}")
    _log(f"defaults: {defaults}")
    _log(f"x0: {dict(zip(PARAM_NAMES, x0))}")

    # Per-parameter bounds, by kind (see PARAM_KIND's own comment):
    #  - additive: raw units directly, from ADDITIVE_BOUNDS.
    #  - signed_mult: relative factor, allowed negative.
    #  - unsigned_mult: relative factor, strictly positive.
    def _bounds_for(name):
        kind = _param_kind(name)
        if kind == "additive":
            return ADDITIVE_BOUNDS[name]
        if kind == "signed_mult":
            return (-3.0, 10.0)
        return (0.02, 10.0)

    bounds_pairs = [_bounds_for(name) for name in PARAM_NAMES]
    lower_bounds = [lo for lo, _ in bounds_pairs]
    upper_bounds = [hi for _, hi in bounds_pairs]

    # Per-coordinate step-size scaling (CMA_stds): SIGMA0 alone is calibrated
    # for the relative-factor dimensions (multiplicative, ~1.0-centered, a
    # 0.15 step is a sensible ~15% perturbation there). per_card_weight_scale
    # is in RAW units spanning a ~30-wide range -- the same 0.15 absolute
    # step would be a tiny, near-useless perturbation on that scale, and
    # CMA-ES would need many generations just to move it away from its 0.0
    # start. CMA_stds rescales each coordinate's actual initial std to
    # sigma0 * CMA_stds[i], so this gives the additive dimension a real
    # (bounds-width/4) initial step while leaving every other dimension's
    # behavior exactly as before (CMA_stds=1.0 there).
    cma_stds = [
        ((hi - lo) / 4.0) / SIGMA0 if _param_kind(name) == "additive" else 1.0
        for name, (lo, hi) in zip(PARAM_NAMES, bounds_pairs)
    ]
    es = cma.CMAEvolutionStrategy(
        x0, SIGMA0,
        {"popsize": N_WORKERS, "bounds": [lower_bounds, upper_bounds], "CMA_stds": cma_stds, "verbose": -9},
    )

    # PAIRED incumbent comparison. Each generation is graded on a FRESH seed set (see seed_base
    # below), and seed-set difficulty dominates the signal: measured live, per-generation mean
    # scores swung 1.219-1.377 (sd ~0.06) while a real parameter improvement is worth perhaps
    # 0.02-0.05. Comparing a candidate against a score measured on DIFFERENT seeds therefore mostly
    # measures which seed set was easier -- with a stale bar, a run can either save a lucky fluke
    # or (observed) save nothing at all across many generations.
    #
    # So the incumbent is re-evaluated every generation on the SAME seeds as that generation's
    # candidates, and a candidate only replaces it by beating it on those shared seeds. This is the
    # same paired correction applied to the Silverbot HP comparison (paired_hp.py), where it moved
    # DONU_AND_DECA from -20.1pp to -10.9pp by removing exactly this kind of confound. Costs one
    # extra evaluation per generation.
    incumbent_x = np.array(x0, dtype=float)
    best_params = dict(prev_params) if prev_params is not None else dict(defaults)
    saved_any = False
    _log("save rule: a candidate must beat the incumbent ON THE SAME SEEDS (paired), re-measured "
         "every generation -- prior-run scores are not used as a bar (different seeds)")
    start = time.time()
    gen = 0
    seed_base = 0

    with mp.Pool(N_WORKERS, initializer=_worker_init) as pool:
        while time.time() - start < TIME_BUDGET_SECONDS and not es.stop():
            gen += 1
            candidates = es.ask()
            # Incumbent appended LAST so it shares this generation's seed_base exactly.
            args_list = [(np.array(x), defaults, seed_base) for x in candidates]
            args_list.append((incumbent_x, defaults, seed_base))
            seed_base += N_EPISODES_PER_ENCOUNTER  # fresh seeds each generation, avoids overfitting to one fixed seed set
            all_fitnesses = pool.map(_evaluate_candidate, args_list)
            fitnesses = all_fitnesses[:-1]
            incumbent_score = -all_fitnesses[-1]
            es.tell(candidates, fitnesses)

            gen_best_idx = int(np.argmin(fitnesses))
            gen_best_score = -fitnesses[gen_best_idx]
            proposed = gen_best_score > incumbent_score

            # TWO-STAGE ACCEPTANCE. Stage 1 (above) compares the BEST OF popsize candidates against
            # a SINGLE incumbent evaluation -- the max of N noisy draws beats one draw of equal true
            # quality most of the time, so stage 1 alone accepts almost always regardless of real
            # merit. Measured live: ~94% of 250+ generations "improved" while incumbent scores
            # showed no upward trend at all and end-to-end win rate moved 214->218/300 (noise).
            #
            # Stage 2 re-runs the challenger and the incumbent head-to-head on a FRESH seed set that
            # neither was selected on. The challenger was picked using stage-1 seeds, so its edge
            # there is partly luck that does not reproduce; an unbiased rematch is what separates a
            # real improvement from a lucky draw. Costs 2 extra evaluations, and only on generations
            # that pass stage 1.
            improved = False
            if proposed:
                confirm_args = [(np.array(candidates[gen_best_idx], dtype=float), defaults, seed_base),
                                (incumbent_x, defaults, seed_base)]
                seed_base += N_EPISODES_PER_ENCOUNTER
                cf = pool.map(_evaluate_candidate, confirm_args)
                chal_confirm, inc_confirm = -cf[0], -cf[1]
                improved = chal_confirm > inc_confirm
                if improved:
                    incumbent_x = np.array(candidates[gen_best_idx], dtype=float)
                    best_params = {name: raw_value(name, incumbent_x[i], defaults[name])
                                   for i, name in enumerate(PARAM_NAMES)}
                    with open(OUT_PATH, "w") as f:
                        # Records the CONFIRMATION-round score: measured on seeds the params were
                        # not selected on, so it is an honest estimate rather than a selection high.
                        json.dump({"score": chal_confirm, "fitness_config": _fitness_config(),
                                   "params": best_params}, f, indent=2)
                    saved_any = True

            elapsed = time.time() - start
            mean_score = -float(np.mean(fitnesses))
            tag = ""
            if proposed:
                tag = (f" | confirm chal={chal_confirm:.3f} inc={inc_confirm:.3f}"
                       f"{'  <-- ACCEPTED' if improved else '  (rejected)'}")
            _log(f"gen {gen:3d} (t={elapsed/60:4.1f}m): mean={mean_score:.3f} "
                 f"gen_best={gen_best_score:.3f} incumbent={incumbent_score:.3f} "
                 f"margin={gen_best_score - incumbent_score:+.3f}{tag}")

    _log(f"=== done after {gen} generations ({(time.time()-start)/60:.1f}m) ===")
    _log(f"best params: {best_params}")
    if saved_any:
        _log(f"saved to {OUT_PATH}")
    else:
        _log(f"NOT saved -- no candidate beat the incumbent on matched seeds; {OUT_PATH} untouched")


if __name__ == "__main__":
    main()
