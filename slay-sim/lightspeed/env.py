"""Gym-shaped environment wrapping sts_lightspeed's BattleContext for a
single isolated Ironclad fight.

Unlike sts/env.py's fixed 4-action vocabulary (a narrow deliberate choice for
that proof-of-concept), this env has no fixed action space at all: each
step() call is handed one of the *currently legal* Action objects from
get_legal_actions(), so there's nothing to mask -- the policy only ever
scores real options. This is what makes "learn all 75 cards" tractable: the
action-scoring policy (see policy.py) conditions on a card embedding + a few
context features per legal action, not a fixed one-hot output per card.

Scope for this first version, matching the phase plan: single fixed
encounter (default Jaw Worm) rather than randomized encounters, so training
initially isolates "learning cards" from "learning matchups". Deck is
randomized per episode (see lightspeed/cards.py) so every card gets
exposure across training.
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import numpy as np
import slaythespire as sts

from .cards import card_index, card_type, random_ironclad_deck, OTHER_CARD_INDEX
from .monsters import monster_index, OTHER_MONSTER_INDEX
from . import relics
from .relic_features import capture_active_relic_idxs, pad_relic_idxs
from . import potions
from .potion_features import capture_active_potion_idxs, pad_potion_idxs, encode_action_with_potion, POTION_ACTION_FEATURES

STATE_FEATURES = 48  # see _encode_state -- NOT counting relic identity,
                     # which enters the network downstream of this raw
                     # vector via its own embedding table (see policy.py's
                     # encode_state/relic_embedding), mirroring how card/
                     # monster identity already enter via embeddings rather
                     # than as raw STATE_FEATURES entries.
ACTION_FEATURES = 12 + POTION_ACTION_FEATURES  # 12 base (see _encode_action_and_card_idx)
                                                        # + 2 potion flags (is_potion, is_potion_discard)
                                                        # appended by encode_action_with_potion, same
                                                        # "identity via embedding, not raw features"
                                                        # split as relics: potion IDENTITY enters via
                                                        # potion_embedding (per-action lookup), not here

# Reward = W_SHAPE * (terminal_reward(...) + potential-based shaping).
#
# Reworks the previous Phi = player_hp - enemy_hp (implicit GAMMA=1) into a
# richer potential, plus a wider win/death terminal gap -- prompted by a
# real pattern in this project's own full-roster eval
# (lightspeed/overnight_v2_progress.log): multi-monster/add-spawning fights
# (Collector 41.7%, Reptomancer 25%) are exactly the ones where "chip the
# boss, ignore the monster about to hit you, die anyway" and a genuine win
# could accumulate similarly-sized total reward, because the old potential
# didn't care WHICH enemy HP dropped or whether anything was about to
# attack -- clearing a harmless monster and clearing one about to hit for
# half your HP scored identically. Two changes target that:
#   - BETA folds each living monster's telegraphed incoming damage into the
#     potential, on top of its raw HP -- the same signal
#     _encode_action_and_card_idx's target_threat_share already gives the
#     POLICY as an input feature, now also present in the REWARD.
#   - W_DEATH = 2x W_WIN widens the terminal gap so dying is unambiguously
#     worse than any near-win, regardless of how per-step shaping happened
#     to land on a given trajectory (previously WIN/LOSS were already
#     guaranteed ordered -- see the old comment below preserved in
#     terminal_reward()'s docstring -- but by a much narrower margin).
# The turn-efficiency nudges (TURN_PENALTY_PER_TURN_ON_WIN /
# TURN_SURVIVED_BONUS_PER_TURN_ON_LOSS, validated earlier this session --
# see checkpoint_turn_efficiency_validation.pt) are unchanged in spirit,
# just layered onto the new terminal magnitudes instead of the old ones.
#
# NOTE: changing the reward function invalidates the value head of every
# existing checkpoint (it was trained to predict returns under the OLD
# reward) -- treat any of the checkpoint_*.pt files as stale until
# retrained/fine-tuned against this new formula, same as this project's own
# established rule for any reward-affecting change.
W_HP = 1.5      # player HP weighted above raw 1:1 in the potential -> mild defensive lean
BETA = 3.0      # weight on an enemy's telegraphed incoming damage, added to its raw HP
GAMMA = 0.99    # must equal ppo.py's own discount for the shaping to actually telescope
                # to a policy-independent constant over an episode -- confirmed as
                # ppo.py's train()/collect_batch's own default gamma, not guessed
W_WIN = 200.0
W_DEATH = 400.0
W_SHAPE = 0.1   # same value/role as the old DENSE_STEP_SCALE: keeps per-step magnitudes
                # unchanged; W_WIN/W_DEATH are sized so that, after this same scaling,
                # the terminal reward lands back around the +/-20-40 ballpark the old
                # unscaled +hp_after / -max_hp terminal terms already used.
DENSE_STEP_SCALE = W_SHAPE  # kept: az_search.py imports this name directly.

TURN_PENALTY_PER_TURN_ON_WIN = 0.5
TURN_SURVIVED_BONUS_PER_TURN_ON_LOSS = 1.0


def potential(bc) -> float:
    """Phi(s) -- 0 at a terminal state by convention (see terminal_reward),
    otherwise weighted player HP minus each living monster's HP plus BETA
    times its telegraphed incoming damage this turn (0 for a Buff/non-
    attack intent, since get_monster_move_damage returns (0, 0) for those).
    A standalone function (not inlined in step()) because az_search.py's
    PUCT backup has to reproduce the exact same per-step reward the value
    head was trained against -- see its own module docstring."""
    if bc.outcome != sts.BattleOutcome.UNDECIDED:
        return 0.0
    phi = W_HP * bc.player_hp
    for i, m in enumerate(bc.monsters):
        if m.half_dead:
            # NOT actually gone -- Awakened One phase 1 dying (revives next
            # turn with a full 300/320 HP bar + Strength) or Darkling
            # Regrow. Its cur_hp is <= 0, so the plain "skip dead monsters"
            # path below would read it as gone and collapse phi to a
            # near-win right before the boss comes back swinging -- which is
            # exactly why inference-time search left Awakened One dead flat
            # (6.7% -> 6.7%) while lifting every other boss: the value it
            # backs up thinks the fight is basically won. Count the HP it's
            # about to revive with (max_hp, the closest proxy the binding
            # exposes) as still-standing threat instead. No incoming-damage
            # term -- a half-dead monster has no telegraphed move this turn.
            phi -= m.max_hp
            continue
        if m.cur_hp <= 0:
            continue
        dmg, hits = bc.get_monster_move_damage(i)
        phi -= m.cur_hp + BETA * dmg * hits
    return phi


def terminal_reward(bc, turn_at_terminal: int) -> float:
    """One-time reward added the step a fight ends. WIN is always
    W_WIN + residual HP; LOSS is a flat -W_DEATH. TURN_PENALTY/
    TURN_SURVIVED_BONUS apply on top, same as before -- previously a
    3-turn and a 15-turn win with equal final HP scored identically, and
    every loss scored the same flat -player_max_hp regardless of turn 2 vs
    turn 14; both terms preserved unchanged, just layered onto the new
    magnitudes. The min(-1.0, ...) floor that used to guarantee "loss always
    strictly below a win-at-0-HP's reward of 0" is now unreachable in
    practice (W_DEATH=400 already dwarfs any turn-count bonus at realistic
    turn counts) but left in as a documented invariant, not dead code."""
    if bc.outcome == sts.BattleOutcome.PLAYER_VICTORY:
        return W_WIN + bc.player_hp - TURN_PENALTY_PER_TURN_ON_WIN * turn_at_terminal
    return min(
        -1.0,
        -W_DEATH + TURN_SURVIVED_BONUS_PER_TURN_ON_LOSS * turn_at_terminal,
    )

# Act 1 "basic" tier -- roughly Jaw Worm's difficulty band, kept together so
# a training run varies matchup shape (single/multi-monster, different
# threat profiles) without also varying difficulty tier (elites/bosses need
# their own HP/deck calibration to be a fair fight, not just a bigger pool).
# Verified via random-policy smoke test to run cleanly through get_legal_actions
# with no crashes before being used for training; win rates at player_hp=50,
# extra_deck_cards=15 ranged from Gremlin Gang's 4/30 to Two Louse/Looter's
# 30/30 -- real difficulty variety, not trivial repeats of the same fight.
ACT1_BASIC_ENCOUNTERS = [
    sts.MonsterEncounter.JAW_WORM,
    sts.MonsterEncounter.CULTIST,
    sts.MonsterEncounter.TWO_LOUSE,
    sts.MonsterEncounter.SMALL_SLIMES,
    sts.MonsterEncounter.BLUE_SLAVER,
    sts.MonsterEncounter.RED_SLAVER,
    sts.MonsterEncounter.LOOTER,
    sts.MonsterEncounter.GREMLIN_GANG,
    sts.MonsterEncounter.EXORDIUM_WILDLIFE,
    sts.MonsterEncounter.EXORDIUM_THUGS,
]

# Elite/boss total HP confirmed by direct construction (sts.new_battle), not
# guessed: Gremlin Nob 82, Lagavulin 111, Three Sentries 117 (3 monsters) --
# roughly 2-3x basic tier's totals (Jaw Worm 42); Guardian 240, Hexaghost
# 250, Slime Boss 140 -- 3-6x. All 6 smoke-tested clean (no crashes) through
# get_legal_actions, including Lagavulin's sleep mechanic, Guardian's
# mode-shift, Hexaghost's ritual, and Slime Boss's split, none of which had
# been exercised through the bindings before this check.
ACT1_ELITE_ENCOUNTERS = [
    sts.MonsterEncounter.GREMLIN_NOB,
    sts.MonsterEncounter.LAGAVULIN,
    sts.MonsterEncounter.THREE_SENTRIES,
]
ACT1_BOSS_ENCOUNTERS = [
    sts.MonsterEncounter.THE_GUARDIAN,
    sts.MonsterEncounter.HEXAGHOST,
    sts.MonsterEncounter.SLIME_BOSS,
]

# Per-tier (player_hp, extra_deck_cards). Testing showed resource budget
# affects win rate independently of matchup novelty -- roughly doubled win
# rates across elites AND bosses alike when moving from 50/+15 to 70/+20 --
# so a mixed-tier pool needs per-tier resources, not one flat setting for
# everything (which would either trivialize basics or make bosses
# unwinnable). Boss numbers are a further step up from the elite-appropriate
# level already validated, not yet independently re-verified against boss
# win rates specifically -- a first calibration, not a tuned one.
TIER_RESOURCES = {
    "basic": (50, 15),
    "elite": (70, 20),
    "boss": (85, 25),
}


def build_encounter_resources(basic=(), elite=(), boss=()):
    """Build an encounter_resources dict for IroncladFightEnv from encounter
    lists grouped by tier."""
    resources = {}
    for enc in basic:
        resources[enc] = TIER_RESOURCES["basic"]
    for enc in elite:
        resources[enc] = TIER_RESOURCES["elite"]
    for enc in boss:
        resources[enc] = TIER_RESOURCES["boss"]
    return resources


# Act 2 and Act 3 pools. All 23 encounters below were smoke-tested via direct
# construction + random-policy playouts through get_legal_actions -- zero
# crashes across every one, including Automaton/Collector's mid-fight spawns,
# Gremlin Leader's summons, Reptomancer's dagger spawns, Awakened One's
# two-phase fight, and Time Eater's turn-count-based mechanic. Random-policy
# win rates at the resource levels below ranged from 0/20 (several elites and
# every boss -- expected, a random policy shouldn't beat these) to 19/20
# (Two Fungi Beasts), confirming these are real, variably-difficult fights
# rather than degenerate ones.
#
# Resources step up act-over-act, not just tier-over-tier: by Act 2/3 a real
# run has a bigger deck and higher max HP than it did in Act 1, so reusing
# Act 1's tier numbers here would understate what the fight is actually
# balanced around. These are first-pass calibrations from the smoke test
# (same status as Act 1's boss tier was before elite/boss cold-transfer
# testing) -- not yet independently re-tuned against a trained policy's win
# rate the way Act 1's were.
ACT2_BASIC_ENCOUNTERS = [
    sts.MonsterEncounter.THREE_BYRDS,
    sts.MonsterEncounter.CHOSEN,
    sts.MonsterEncounter.CHOSEN_AND_BYRDS,
    sts.MonsterEncounter.SHELL_PARASITE,
    sts.MonsterEncounter.SHELLED_PARASITE_AND_FUNGI,
    sts.MonsterEncounter.TWO_FUNGI_BEASTS,
]
ACT2_ELITE_ENCOUNTERS = [
    sts.MonsterEncounter.GREMLIN_LEADER,
    sts.MonsterEncounter.BOOK_OF_STABBING,
    sts.MonsterEncounter.CENTURION_AND_HEALER,
    sts.MonsterEncounter.SNAKE_PLANT,
    sts.MonsterEncounter.SNECKO,
]
ACT2_BOSS_ENCOUNTERS = [
    sts.MonsterEncounter.AUTOMATON,
    sts.MonsterEncounter.COLLECTOR,
    sts.MonsterEncounter.CHAMP,
]
ACT3_BASIC_ENCOUNTERS = [
    sts.MonsterEncounter.THREE_DARKLINGS,
    sts.MonsterEncounter.ORB_WALKER,
    sts.MonsterEncounter.WRITHING_MASS,
    sts.MonsterEncounter.THREE_SHAPES,
    sts.MonsterEncounter.SPHERE_AND_TWO_SHAPES,
    sts.MonsterEncounter.FOUR_SHAPES,
]
ACT3_ELITE_ENCOUNTERS = [
    sts.MonsterEncounter.NEMESIS,
    sts.MonsterEncounter.REPTOMANCER,
    sts.MonsterEncounter.SPHERIC_GUARDIAN,
]
ACT3_BOSS_ENCOUNTERS = [
    sts.MonsterEncounter.AWAKENED_ONE,
    sts.MonsterEncounter.TIME_EATER,
    sts.MonsterEncounter.DONU_AND_DECA,
]

# (player_hp, extra_deck_cards, upgrade_chance, starter_removals) per (act, tier).
# Act1 pool numbers (bare (hp, cards)) are held over unchanged from the
# earlier calibration; act2/3 gained two more dimensions here, not just
# bigger numbers on the same two: a real run's deck gets genuinely BETTER by
# Act 2/3, not just longer -- more rest-site upgrades (upgrade_chance climbs
# 0.3 -> 0.4 -> 0.5) and Smith/Peace Pipe-style starter-card removals (0 -> 1
# -> 2 Strike/Defend cards stripped from the starting 10). Before this,
# act2/3 decks were only bigger random pulls from the same pool at the same
# upgrade rate as Act 1 -- a longer deck, not a stronger one. First-pass
# calibration (same status as the rest of ACT_TIER_RESOURCES), not tuned
# against a trained policy yet.
# 5th element: relic_count, EXTRA relics beyond Burning Blood (granted
# separately/unconditionally, see reset()) -- scaled by tier the same
# rough-first-pass way as HP/deck size above (a real run accumulates more
# relics by Act 3 than Act 1), only used when relic_generator is actually
# set (see IroncladFightEnv.__init__), so this is inert/backward-compatible
# for every existing training script that never passes one.
# 6th element: n_boss_relics, how many of relic_count must be BOSS-rarity
# (see relics.py's weighted_ironclad_relics) -- a real run gets exactly one
# guaranteed boss-relic choice after each act boss kill, so anyone still in
# Act 1 has 0, Act 2 has 1 (from Act 1's boss), Act 3 has 2 (from Act 1 and
# Act 2's bosses). Uniform across all three tiers within an act (basic/
# elite/boss all fought AFTER the previous act's boss, or -- for an act's
# OWN boss tier -- before ITS boss relic is granted, same count as that
# act's basic/elite). Without this, general-pool frequency sampling would
# only include a boss relic by chance instead of guaranteeing the count a
# real run always has by that point.
ACT_TIER_RESOURCES = {
    # Act 1 extra_cards are 10 lower than the other acts' scaling would suggest: a real Act 1 adds
    # only ~5-8 cards to the 10-card starter deck, so the previous 15/20/25 produced 26/31/36-card
    # decks by Act 1 -- roughly an Act 2 deck fought with Act 1 HP and relics. The lean deck matters
    # for what is being optimized here, not just realism: fewer cards per shuffle means fewer
    # defensive options available on any given turn, which is exactly the axis the HP-preservation
    # tuning is working on.
    ("act1", "basic"): (50, 5, 0.3, 0, 2, 0),
    ("act1", "elite"): (70, 10, 0.3, 0, 3, 0),
    ("act1", "boss"): (85, 15, 0.3, 0, 4, 0),
    ("act2", "basic"): (90, 22, 0.4, 1, 6, 1),
    ("act2", "elite"): (110, 28, 0.4, 1, 7, 1),
    ("act2", "boss"): (120, 35, 0.4, 1, 8, 1),
    ("act3", "basic"): (100, 28, 0.5, 2, 9, 2),
    ("act3", "elite"): (115, 32, 0.5, 2, 10, 2),
    ("act3", "boss"): (130, 40, 0.5, 2, 11, 2),
}

ALL_ACT_TIER_GROUPS = [
    ("act1", "basic", ACT1_BASIC_ENCOUNTERS),
    ("act1", "elite", ACT1_ELITE_ENCOUNTERS),
    ("act1", "boss", ACT1_BOSS_ENCOUNTERS),
    ("act2", "basic", ACT2_BASIC_ENCOUNTERS),
    ("act2", "elite", ACT2_ELITE_ENCOUNTERS),
    ("act2", "boss", ACT2_BOSS_ENCOUNTERS),
    ("act3", "basic", ACT3_BASIC_ENCOUNTERS),
    ("act3", "elite", ACT3_ELITE_ENCOUNTERS),
    ("act3", "boss", ACT3_BOSS_ENCOUNTERS),
]

ALL_ENCOUNTERS = [enc for _act, _tier, encs in ALL_ACT_TIER_GROUPS for enc in encs]


def build_full_encounter_resources(groups=ALL_ACT_TIER_GROUPS):
    """Build an encounter_resources dict spanning any set of (act, tier,
    encounter_list) groups, using ACT_TIER_RESOURCES for each group's
    (player_hp, extra_deck_cards)."""
    resources = {}
    for act, tier, encs in groups:
        for enc in encs:
            resources[enc] = ACT_TIER_RESOURCES[(act, tier)]
    return resources


def saturate(x: float, c: float) -> float:
    """Saturating normalization x/(x+c) -> [0, 1) for a non-negative x, with
    the half-max point at x==c. Replaces the earlier bare linear x/CONST
    scaling on genuinely-unbounded quantities (block, incoming damage,
    strength, most power stacks): those CONSTs were picked as rough
    "typical single-turn" values back when the agent was a plain aggro
    deck, but Barricade/Metallicize/Demon Form (all now exposed to the
    policy) make block and damage accumulate unboundedly across turns, so a
    Barricade build routinely sat at 2x-4x the old divisor -- a linear ramp
    that spends most of its interesting range above 1.0, with the high-end
    distinctions the scaling bosses actually turn on compressed into a tail
    the small MLP under-resolves. x/(x+c) keeps good resolution at the low
    end (where most decisions live) AND never blows past 1.0 no matter how
    high the stack goes, with no arbitrary "typical max" to guess wrong.
    HP-type features stay plain fractions (already bounded [0,1] by their
    own max), not routed through this."""
    return x / (x + c) if x > 0 else 0.0


def _encode_state(bc, hand: list) -> Tuple[np.ndarray, int]:
    # Some encounters (Bronze Automaton, The Collector, Gremlin Leader,
    # Reptomancer -- all with mid-fight-spawn mechanics) reserve monster
    # slots that start as MonsterId::INVALID/0 HP until something spawns
    # into them; bc.monsters includes those placeholder entries. Targeting
    # already correctly excludes them (get_legal_actions checks
    # isTargetable() in the C++ binding), but naively counting len(monsters)
    # would overstate "how many enemies" for these encounters specifically --
    # filter to currently-alive monsters for that feature, found while
    # investigating Act 2/3 encounters that exercise these spawn slots for
    # the first time.
    all_monsters = bc.monsters  # one fetch: the binding builds a fresh list/objects each call
    alive = [(i, m) for i, m in enumerate(all_monsters) if m.cur_hp > 0]
    monsters = [m for _i, m in alive]
    total_enemy_hp = sum(m.cur_hp for m in monsters) or 1
    total_enemy_max_hp = sum(m.max_hp for m in monsters) or 1
    incoming = 0
    for i, _m in alive:
        dmg, hits = bc.get_monster_move_damage(i)
        incoming += dmg * hits
    # Enemy-side status: player_strength/dexterity above are the PLAYER's
    # own buffs, but nothing previously told the policy anything about the
    # enemies' -- e.g. Gremlin Nob's Enrage stacks Strength on itself, and
    # with no way to observe that, the policy had no way to learn "this
    # Skill will make the enemy hit harder later" no matter how much
    # training happened. max() across living monsters (not sum) so one
    # heavily-buffed enemy in a multi-monster fight reads as dangerous
    # regardless of how many other, unbuffed monsters are also alive --
    # same reasoning as target_threat_share using a per-target rather than
    # fight-wide aggregate.
    max_enemy_strength = max((m.strength for m in monsters), default=0)
    max_enemy_vulnerable = max((m.vulnerable for m in monsters), default=0)
    max_enemy_weak = max((m.weak for m in monsters), default=0)
    # max_enemy_block: Monster::block was already exposed in the binding
    # (used nowhere in this file before) -- Spheric Guardian opens with 40
    # block + Barricade-like behavior, and nothing previously told the
    # policy whether an attack would actually connect versus just chip an
    # enemy's block.
    max_enemy_block = max((m.block for m in monsters), default=0)
    # Enemy statuses via the generic get_monster_status_value() binding --
    # the policy previously saw only enemy strength/vulnerable/weak/block,
    # blind to: Poison (a whole win condition it couldn't track ticking
    # down the enemy), Plated Armor (Automaton/Shelled Parasite -- block
    # that regenerates every turn, so chipping is futile), Artifact
    # (Automaton/Spheric Guardian -- why the player's debuffs silently
    # whiff), and Mode Shift (The Guardian's damage-threshold phase flip).
    # max() across living monsters, same per-enemy-max reasoning as the
    # strength/block features above. Indices come from `alive` so a dead/
    # placeholder slot is never queried.
    max_enemy_poison = max((bc.get_monster_status_value(i, "POISON") for i, _m in alive), default=0)
    max_enemy_plated_armor = max((bc.get_monster_status_value(i, "PLATED_ARMOR") for i, _m in alive), default=0)
    max_enemy_artifact = max((bc.get_monster_status_value(i, "ARTIFACT") for i, _m in alive), default=0)
    max_enemy_metallicize = max((bc.get_monster_status_value(i, "METALLICIZE") for i, _m in alive), default=0)
    max_enemy_mode_shift = max((bc.get_monster_status_value(i, "MODE_SHIFT") for i, _m in alive), default=0)
    # Time Warp counter: THE signal for Time Eater, and previously WRONG.
    # An earlier version fed bc.player_cards_played_this_turn (the player's
    # per-turn counter, which resets to 0 every turn -- confirmed at
    # BattleContext.cpp:2837), labeled as "Time Warp progress". But Time
    # Warp is a MONSTER status (MS::TIME_WARP) that counts cards across the
    # WHOLE fight and only resets when it triggers at 12 -- it does NOT
    # reset at turn boundaries. The two coincide only during turn 1, then
    # diverge (verified empirically: per-turn reads 3,3,3,2 while the real
    # counter climbs 3,6,9,11). So for the exact boss the feature existed
    # to help, the policy was seeing a number that didn't track the
    # mechanic. This reads the actual monster counter. max() across living
    # monsters so it's naturally 0 in every non-Time-Eater fight (no monster
    # has the status) without special-casing the encounter.
    max_time_warp = max((bc.get_monster_status_value(i, "TIME_WARP") for i, _m in alive), default=0)
    # half_dead: true for exactly the turn a monster is in a death-but-
    # revives-next-turn state (Awakened One's stage 1->2 transition, and
    # Darkling's Regrow) -- see Monster::isHalfDead in sts_lightspeed. Such
    # a monster has cur_hp <= 0, so it's excluded from `monsters`/`alive`
    # above (not targetable, matching the real game -- it's not there to
    # hit), which means total_enemy_hp/len(monsters)/incoming all read
    # exactly like a real win that same turn. Without this flag the policy
    # can't distinguish "actually won" from "boss is about to come back
    # swinging" from the state alone -- checked against the full (unfiltered)
    # all_monsters list, since half_dead monsters are excluded from `alive`.
    any_half_dead = 1.0 if any(m.half_dead for m in all_monsters) else 0.0
    # Player powers beyond Strength/Dexterity: every one of these is a real,
    # reachable Power this engine implements for Ironclad (confirmed by
    # grepping BattleContext.cpp's buff<PS::...> call sites, not guessed),
    # and none of them were previously visible to the policy at all -- e.g.
    # it could play Metallicize/Barricade/Demon Form and never see the
    # stack it just built, so it had no way to factor its own scaling into
    # later decisions. Uses the generic get_player_status_value() binding
    # (added alongside this) rather than one dedicated binding per power.
    # BARRICADE/CORRUPTION are bool-only in the engine (see the
    # getStatusRuntime fix in Player.cpp) so they read 0/1 already. The rest
    # are intensity/counter stacks routed through saturate() -- Demon Form
    # (grows 2-3 Strength/turn, unbounded), Metallicize, Panache, etc. can
    # climb arbitrarily high over a long fight, exactly the linear-scaling
    # problem saturate() exists for; the per-power c is a rough half-max
    # (the stack value at which the feature reads 0.5), not a cap.
    g = bc.get_player_status_value
    player_powers = [
        saturate(g("ARTIFACT"), 2.0),
        g("BARRICADE"),
        saturate(g("METALLICIZE"), 6.0),
        saturate(g("RITUAL"), 3.0),
        saturate(g("RAGE"), 5.0),
        saturate(g("RUPTURE"), 3.0),
        saturate(g("COMBUST"), 8.0),
        saturate(g("DEMON_FORM"), 4.0),
        saturate(g("DARK_EMBRACE"), 2.0),
        saturate(g("EVOLVE"), 2.0),
        saturate(g("FEEL_NO_PAIN"), 4.0),
        saturate(g("FIRE_BREATHING"), 8.0),
        saturate(g("JUGGERNAUT"), 6.0),
        saturate(g("PANACHE"), 10.0),
        saturate(g("ENVENOM"), 2.0),
        saturate(g("FLAME_BARRIER"), 6.0),
        saturate(g("BRUTALITY"), 2.0),
        saturate(g("REGEN"), 6.0),
        g("CORRUPTION"),
    ]
    # Deck composition (draw_pile + discard_pile -- NOT hand, which is
    # already individually visible via each action's own card embedding;
    # this is specifically the "not currently visible" remainder). Added
    # because the policy previously had zero visibility into its own deck:
    # it could have a scaling payoff (Limit Break, a Strength engine) still
    # buried in the draw pile, or a deck getting clogged with shuffled-in
    # Wounds/Dazed/Void from a card or monster effect (Awakened One's
    # Sludge shuffles in Void), and had no way to plan around either.
    # get_card_type (not cards.card_type) is used because it resolves ANY
    # card id, including Status/Curse cards outside the 75-card Ironclad
    # pool that card_type() would silently return None for.
    remainder = list(bc.draw_pile) + list(bc.discard_pile)
    remainder_types = [sts.get_card_type(c.id) for c in remainder]
    remainder_total = max(1, len(remainder))
    frac_attack = remainder_types.count(sts.CardType.ATTACK) / remainder_total
    frac_skill = remainder_types.count(sts.CardType.SKILL) / remainder_total
    frac_power = remainder_types.count(sts.CardType.POWER) / remainder_total
    frac_status_or_curse = (
        remainder_types.count(sts.CardType.STATUS) + remainder_types.count(sts.CardType.CURSE)
    ) / remainder_total
    deck_remaining_size = saturate(len(remainder), 20.0)
    # Unblocked-incoming-as-fraction-of-HP: the policy CAN in principle
    # derive "am I about to take a big/lethal hit" from incoming/block/hp
    # separately, but a small MLP benefits from not having to rediscover a
    # max(0, incoming-block)/hp subtraction on every forward pass -- an
    # explicit, cheap defensive urgency signal.
    unblocked_incoming_frac = max(0, incoming - bc.player_block) / bc.player_hp if bc.player_hp > 0 else 1.0
    # any_enemy_charging: is any living monster's CURRENT queued move a
    # self/ally buff rather than an attack (Donu's Circle of Power/Deca's
    # Square of Protection alternate attack <-> charge-up, each charge turn
    # strengthening/blocking BOTH of them)? get_monster_move_damage reads 0
    # for a non-attack move, so potential()'s incoming-damage term already
    # treats a charging turn as equally safe as a genuinely harmless one --
    # this exposes the raw "someone is powering up right now" signal
    # directly as a state feature instead, so the network's own value
    # function can learn the right weight for it (an explicit reward-side
    # guess at "how much MORE dangerous" would need a next-turn damage
    # projection this project has no clean way to compute). Generalizes to
    # any other charge-then-attack boss, not just Donu & Deca.
    any_enemy_charging = 0.0
    for i, _m in alive:
        cat = bc.classify_monster_move(i)
        if cat.self_buffs or cat.buffs_ally:
            any_enemy_charging = 1.0
            break
    # any_enemy_used_haste: Time Eater's one-time <=50%-HP proc (heals to
    # 50%, WIPES every player debuff) is otherwise completely invisible to
    # the policy -- nothing previously indicated it exists or has already
    # fired, so a Vulnerable/Weak/Poison-stacking strategy has no signal
    # that its setup is about to be erased for free. miscInfo's meaning is
    # monster-type-specific (see get_monster_misc_info's own docstring), so
    # this is explicitly gated on monster identity rather than treated as a
    # universal status -- 0 in every fight that isn't Time Eater.
    any_enemy_used_haste = 0.0
    for i, m in alive:
        if m.name == "TIME_EATER" and bc.get_monster_misc_info(i) != 0:
            any_enemy_used_haste = 1.0
            break
    # Two normalization styles, deliberately: plain fractions for quantities
    # already bounded by their own denominator (HP fractions) or by the game
    # (energy, hand size, turn count, monster count, debuff durations, the
    # Time Warp counter which is bounded 0-12 by design); saturate() for
    # genuinely unbounded magnitudes (block, incoming damage, Strength,
    # enemy Poison/Plated Armor/Metallicize) -- see saturate()'s docstring.
    state = np.array([
        bc.player_hp / bc.player_max_hp,
        saturate(bc.player_block, 20.0),
        bc.player_energy / 3.0,
        saturate(bc.player_strength, 8.0),
        bc.player_dexterity / 10.0,
        total_enemy_hp / total_enemy_max_hp,
        len(monsters) / 3.0,
        saturate(incoming, 20.0),
        len(hand) / 10.0,
        bc.turn / 20.0,
        saturate(max_enemy_strength, 8.0),
        max_enemy_vulnerable / 5.0,
        max_enemy_weak / 5.0,
        max_time_warp / 12.0,
        any_half_dead,
        saturate(max_enemy_block, 20.0),
        saturate(max_enemy_poison, 15.0),
        saturate(max_enemy_plated_armor, 8.0),
        saturate(max_enemy_artifact, 2.0),
        saturate(max_enemy_metallicize, 6.0),
        saturate(max_enemy_mode_shift, 30.0),
        frac_attack,
        frac_skill,
        frac_power,
        frac_status_or_curse,
        deck_remaining_size,
        unblocked_incoming_frac,
        any_enemy_charging,
        any_enemy_used_haste,
    ] + player_powers, dtype=np.float32)
    # `hand` is passed in (not re-fetched here) for the same reason
    # `total_enemy_hp` is returned below: bc.hand crosses the same
    # C++-binding-builds-a-fresh-list boundary as bc.monsters (measured at
    # ~1.3 microseconds/access), and _observation() needs the SAME hand
    # list again for card lookups -- fetching it once at the top of
    # _observation() and threading it through beats each of this function
    # and _encode_action_and_card_idx fetching their own copy.
    #
    # total_enemy_hp is returned alongside the array -- _observation() needs
    # this exact same sum for _encode_action_and_card_idx's
    # target_threat_share feature, and computing it a second time there
    # would mean a second full bc.monsters fetch for a value already sitting
    # right here (this is the second time this exact redundancy has been
    # found in this function's neighborhood -- see _encode_action_and_card_idx's
    # docstring for the first one -- worth remembering when adding a new
    # per-action feature that touches bc.monsters again in the future).
    return state, total_enemy_hp


def _hand_card_for_action(hand: list, action):
    """CARD actions (from PLAYER_NORMAL) index into hand; CARD_SELECT
    actions (Exhume/Warcry/Armaments-style follow-up choices) may index into
    a different list entirely (exhaust pile, a generated choice set, ...) --
    not yet verified against source for every such card. Bounds-checked so
    an out-of-range index degrades to "unknown card" instead of crashing.
    Takes the already-fetched hand list (not bc) -- see _encode_state's
    docstring for why bc.hand shouldn't be re-fetched per call."""
    idx = action.source_idx
    if action.action_type == sts.ActionType.CARD and 0 <= idx < len(hand):
        return hand[idx]
    return None


def _encode_action_and_card_idx(bc, action, total_living_hp: int, hand: list) -> Tuple[np.ndarray, Optional[int], Optional[int]]:
    """Combined form of what used to be two separate functions
    (_encode_action + _card_idx_for_action), each independently calling
    _hand_card_for_action for the same action -- profiled at a real,
    measurable share of _observation()'s cost (11284 redundant calls across
    a 60-episode batch in one profile run) since _observation() built BOTH
    lists via separate comprehensions over the same `legal` actions.
    total_living_hp and hand are both passed in rather than recomputed/
    re-fetched here because they're the SAME for every action in a given
    _observation() call (both depend only on the current board, not which
    action is being encoded) -- computing/fetching them once per call
    instead of once per action removes a `bc.monsters`/`bc.hand` re-fetch
    per action (the binding builds a fresh object list on every access to
    either, see _encode_state's own note on this).

    Returns (action_features, card_idx, monster_idx) -- card_idx is None for
    END_TURN (no card to embed), OTHER_CARD_INDEX for a card we can't
    resolve. monster_idx is None for END_TURN or an untargeted card (no
    monster to embed), else the target monster's identity index (see
    monsters.py) -- added alongside target_is_attacking, same motivation:
    the network otherwise has no way to condition its strategy on WHICH
    monster it's targeting, only anonymous HP/strength/block numbers.

    target_threat_share (added after the minion-targeting failure diagnosed
    via the expectimax-vs-greedy comparison against Automaton/Collector/
    Gremlin Leader): target_hp_frac alone can't distinguish "the boss" from
    "a minion" -- a full-HP minion and a full-HP boss both read as 1.0. This
    is the target's CURRENT hp as a share of the encounter's TOTAL living-
    monster hp, so the boss (the bulk of the fight's total HP) reads as a
    high-threat-share target and a minion reads as low-share, even both at
    their own 100% HP -- the exact signal that made search correctly
    deprioritize minions where greedy/lowest-hp-first targeting couldn't.

    is_attack/is_skill/is_power (added alongside the enemy-status state
    features, same motivation): CardInstance (bc.hand's element type)
    doesn't expose a type, so nothing previously told the policy whether a
    card was an Attack, Skill, or Power -- meaning it had no way to learn
    "this Skill will feed Gremlin Nob's Enrage" even in principle, since
    "this is a Skill" wasn't observable at all. Looked up via
    cards.card_type(), a fixed property of the card id precomputed once at
    import time, not per-instance state.

    target_is_attacking (added alongside the Time Eater/Awakened One state
    features): target_threat_share ranks a target by its SHARE of total
    enemy HP, but says nothing about whether THIS SPECIFIC monster is the
    one about to hit you this turn -- in a Donu & Deca-style fight (one
    attacks while the other buffs, alternating) both monsters have equal
    HP share, so nothing previously distinguished "kill the one about to
    swing" from "kill the one that's buffing". 1.0 if get_monster_move_damage
    for this target is nonzero (a Buff/non-attack intent already returns
    (0, 0), same convention potential() relies on), else 0.0.

    target_self_buffs/target_buffs_ally/target_debuffs_player (added
    alongside classify_monster_move): target_is_attacking only answers
    "is this an attack" -- it doesn't distinguish Donu's Buff (buffs_ally)
    from a plain do-nothing move, or Time Eater's Ripple (debuffs_player,
    0 damage) from Haste (neither). classify_monster_move derives these by
    actually running the move on a throwaway BattleContext copy and
    observing what changed (see its own docstring for why this is exact
    rather than a guessed per-move-id table), not by hand-classifying 196
    move ids across ~47 monster types."""
    if action.action_type == sts.ActionType.END_TURN:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32), None, None

    card = _hand_card_for_action(hand, action)
    card_idx = OTHER_CARD_INDEX if card is None else card_index(card.id)
    ctype = card_type(card.id) if card is not None else None

    target_hp_frac = 0.0
    target_threat_share = 0.0
    target_is_attacking = 0.0
    target_self_buffs = 0.0
    target_buffs_ally = 0.0
    target_debuffs_player = 0.0
    monster_idx = None
    monsters = bc.monsters
    if 0 <= action.target_idx < len(monsters):
        m = monsters[action.target_idx]
        target_hp_frac = m.cur_hp / max(1, m.max_hp)
        target_threat_share = m.cur_hp / max(1, total_living_hp)
        dmg, hits = bc.get_monster_move_damage(action.target_idx)
        target_is_attacking = 1.0 if dmg * hits > 0 else 0.0
        cat = bc.classify_monster_move(action.target_idx)
        target_self_buffs = 1.0 if cat.self_buffs else 0.0
        target_buffs_ally = 1.0 if cat.buffs_ally else 0.0
        target_debuffs_player = 1.0 if cat.debuffs_player else 0.0
        monster_idx = monster_index(m.name)
    action_features = np.array([
        (card.cost_for_turn / 3.0) if card is not None else 0.0,
        1.0 if (card is not None and card.upgraded) else 0.0,
        target_hp_frac,
        target_threat_share,
        0.0,  # not END_TURN
        1.0 if ctype == sts.CardType.ATTACK else 0.0,
        1.0 if ctype == sts.CardType.SKILL else 0.0,
        1.0 if ctype == sts.CardType.POWER else 0.0,
        target_is_attacking,
        target_self_buffs,
        target_buffs_ally,
        target_debuffs_player,
    ], dtype=np.float32)
    return action_features, card_idx, monster_idx


class IroncladFightEnv:
    def __init__(self, encounter=sts.MonsterEncounter.JAW_WORM,
                 extra_deck_cards: int = 15, player_hp: Optional[int] = None,
                 deck_exclude=None, deck_force_include=None,
                 encounter_resources: Optional[dict] = None,
                 upgrade_chance: float = 0.3, starter_removals: int = 0,
                 ascension: int = 0, encounter_weights: Optional[List[float]] = None,
                 deck_generator=None, relic_generator=None, relic_count: int = 0,
                 potion_generator=None, potion_count: int = 0):
        """`encounter` is either a single MonsterEncounter (old behavior,
        every episode fights the same thing) or a list of them, in which
        case each reset() samples one uniformly at random -- "scale
        encounters": a policy trained against a fixed single fight can
        overfit to that specific matchup's patterns rather than learning
        cards/tactics that transfer, the same overfitting risk a fixed
        single deck had before deck randomization was added.

        `encounter_resources`: optional {MonsterEncounter: (player_hp,
        extra_deck_cards[, upgrade_chance, starter_removals])} overriding
        the flat args below for specific encounters. Needed once a pool
        mixes difficulty tiers (basic/elite/boss): testing showed resource
        budget affects win rate independently of matchup novelty -- a
        single flat HP/deck size across tiers either trivializes the easy
        fights or makes the hard ones unwinnable. The trailing two fields
        are optional per entry (falls back to the flat upgrade_chance/
        starter_removals args below when omitted) -- 2-tuple entries from
        before this was added still work unchanged. Encounters not in this
        dict fall back to the flat args entirely, so this is purely
        additive.

        `upgrade_chance`/`starter_removals`: flat defaults (used when an
        encounter isn't in encounter_resources, or when encounter_resources
        isn't used at all). `starter_removals` strips that many random
        Strike/Defend cards from the starting 10-card deck (mirroring
        Smith/Peace Pipe-style rest-site removals a real run accumulates by
        Act 2/3) -- 0 leaves the starter deck untouched."""
        self.encounter_pool = encounter if isinstance(encounter, (list, tuple)) else [encounter]
        self.extra_deck_cards = extra_deck_cards
        self.player_hp = player_hp  # None -> class default (80 for Ironclad)
        self.deck_exclude = deck_exclude  # held-out-card training: cards never sampled
        self.deck_force_include = deck_force_include  # held-out-card eval: cards guaranteed present
        self.encounter_resources = encounter_resources or {}
        self.upgrade_chance = upgrade_chance
        self.starter_removals = starter_removals
        # None -> random_ironclad_deck (uniform sampling, every prior run
        # this project trained), matching existing behavior exactly.
        # weighted_ironclad_deck (cards.py) is the real-data-informed
        # alternative -- same call signature, so either works here.
        self.deck_generator = deck_generator or random_ironclad_deck
        # None -> grant NO relics at all (not even Burning Blood), fully
        # unchanged from every prior run this project trained -- relics are
        # purely opt-in, same backward-compatibility stance as
        # deck_generator. relics.weighted_ironclad_relics is the real-data
        # alternative; relic_count is a flat default, overridable per
        # (act, tier) via a 5th entry in encounter_resources (see reset()).
        self.relic_generator = relic_generator
        self.relic_count = relic_count
        # None -> grant NO potions, same opt-in convention as relic_generator/
        # deck_generator. potions.uniform_ironclad_potions is the trainable
        # pool (see potions.py); potion_count is a flat default, no per-tier
        # override wired up yet (unlike relic_count's encounter_resources
        # 5th-entry mechanism) -- revisit if tier-scaled potion counts turn
        # out to matter, not assumed necessary yet.
        self.potion_generator = potion_generator
        self.potion_count = potion_count
        # Was hardcoded to 0 (no ascension arg at all) until this became
        # configurable -- discovered that meant every RL run this session
        # trained against A0 monsters despite sts/enemies.py's hand-derived
        # stats (used by the search+value-net mod) being calibrated to A20.
        # A20 was briefly the default; reverted back to A0 -- a quick A20
        # comparison run (checkpoint_5min_a20.pt) showed the difficulty
        # increase is real (Act2/3 bosses noticeably harder) but training
        # against A20 for real is a bigger commitment than a quick check,
        # deferred for now. Stays configurable either way.
        self.ascension = ascension
        self.bc = None
        self.last_encounter = None  # set by reset(); lets callers do per-encounter breakdowns
        self._active_relic_idxs: List[int] = []  # set by reset(); see capture_active_relic_idxs
        self.rng = random.Random()
        # None -> uniform (rng.choice), matching every previous training run.
        # A driver script can set this attribute directly (mutating the
        # live env between chunks) to len(encounter_pool) weights -- e.g.
        # biased toward encounters with a low recent win rate, see
        # train_curriculum.py. Also a real constructor kwarg (not just a
        # post-construction attribute) so _env_kwargs_from() can thread it
        # through to parallel workers, which get rebuilt from scratch each
        # chunk (see ppo.py's train_ppo/_worker_init) -- that's what makes
        # setting it on the driver's own env actually reach the workers.
        self.encounter_weights: Optional[List[float]] = encounter_weights

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng.seed(seed)
        if self.encounter_weights is None:
            self.last_encounter = self.rng.choice(self.encounter_pool)
        else:
            self.last_encounter = self.rng.choices(self.encounter_pool, weights=self.encounter_weights, k=1)[0]
        entry = self.encounter_resources.get(self.last_encounter)
        # potion_count has no per-tier override entry (unlike relic_count's
        # 5th/6th encounter_resources fields) -- always the flat default,
        # regardless of which branch below fires.
        potion_count = self.potion_count
        if entry is None:
            hp, extra_cards = self.player_hp, self.extra_deck_cards
            upgrade_chance, starter_removals = self.upgrade_chance, self.starter_removals
            relic_count = self.relic_count
            n_boss_relics = 0
        else:
            hp, extra_cards = entry[0], entry[1]
            upgrade_chance = entry[2] if len(entry) > 2 else self.upgrade_chance
            starter_removals = entry[3] if len(entry) > 3 else self.starter_removals
            relic_count = entry[4] if len(entry) > 4 else self.relic_count
            n_boss_relics = entry[5] if len(entry) > 5 else 0

        gc_seed = self.rng.randrange(1, 2**31)
        gc = sts.GameContext(sts.CharacterClass.IRONCLAD, gc_seed, self.ascension)

        # Relics granted BEFORE the hp override below, deliberately -- a
        # few relics in the pool (Mango/Strawberry/Pear/Lee's Waffle) raise
        # max_hp on pickup, and the explicit tier-calibrated hp assignment
        # must always win over that rather than stack with it, or every
        # tier's carefully-set ACT_TIER_RESOURCES HP value would silently
        # drift depending on which relics happened to be sampled.
        if self.relic_generator is not None:
            gc.obtain_relic(relics.BURNING_BLOOD)
            for r in self.relic_generator(self.rng, relic_count, n_boss_relics):
                gc.obtain_relic(r)

        # Snapshot which relics are active as embedding indices, from gc
        # (not bc.has_relic() during combat -- that binding only reflects
        # relics the engine's own logic re-checks repeatedly, like Ginger/
        # Turnip, not "does the player own this"; confirmed unreliable for
        # one-time battle-start-effect relics like Vajra). Captured once
        # here rather than queried live, since relics never change mid-
        # combat -- relic_generator=None (the default, matching every prior
        # run before relics existed) leaves this an empty list, same as
        # granting no relics at all.
        self._active_relic_idxs = capture_active_relic_idxs(gc) if self.relic_generator is not None else []

        # Potions granted here too (order vs. relics doesn't matter -- no
        # potion in the trainable pool affects max_hp, unlike Mango/
        # Strawberry/Pear/Lee's Waffle among relics, so there's no
        # override-ordering constraint to preserve the way relics have).
        # Unlike relic identity (captured once here, since relics never
        # change mid-combat), potion identity is captured fresh every
        # _observation() call instead -- potions get drunk/discarded
        # turn to turn, so a one-time snapshot here would go stale.
        if self.potion_generator is not None:
            for p in self.potion_generator(self.rng, potion_count):
                gc.obtain_potion(p)

        if hp is not None:
            gc.cur_hp = hp
            gc.max_hp = hp

        # Mirrors Smith/Peace Pipe-style rest-site removals a real run
        # accumulates by Act 2/3 -- re-fetches gc.deck each iteration since
        # remove_card(idx) shifts indices.
        for _ in range(starter_removals):
            removable = [i for i, c in enumerate(gc.deck) if c.is_starter_strike_or_defend]
            if not removable:
                break
            gc.remove_card(self.rng.choice(removable))

        deck = self.deck_generator(
            self.rng, extra_cards=extra_cards, upgrade_chance=upgrade_chance,
            exclude=self.deck_exclude, force_include=self.deck_force_include,
        )
        for card in deck:
            gc.obtain_card(card)

        self.bc = sts.new_battle(gc, self.last_encounter)
        if self.bc.outcome != sts.BattleOutcome.UNDECIDED:
            # Rare: a battle-start relic effect can already decide the fight
            # before the player ever acts (e.g. Neow's Lament sets every
            # monster to 1 HP, then Mercury Hourglass's battle-start 3 AoE
            # damage kills them all) -- get_legal_actions() then returns
            # empty even though nothing is wrong, since PLAYER_NORMAL is
            # never reached. Not a bug to raise on: just re-roll a fresh
            # episode from the same rng stream.
            return self.reset()
        return self._observation()

    def _observation(self):
        legal = self.bc.get_legal_actions()
        if not legal:
            # getLegalActions() only recognizes PLAYER_NORMAL and CARD_SELECT
            # (see the comment in sts_lightspeed's bindings-util.cpp). An
            # empty list while the fight is still undecided means we hit some
            # other input state (a potion/relic choice) that isn't wired up
            # yet -- fail loudly with the raw state ordinal rather than hang.
            raise RuntimeError(
                f"no legal actions but outcome is still undecided -- "
                f"unhandled InputState (raw={sts.get_input_state_raw(self.bc)})"
            )
        # hand fetched once here (not inside _encode_state or per-action)
        # and total_living_hp reused from _encode_state's own computation
        # (same sum, same bc.monsters fetch) rather than recomputed here --
        # see _encode_state's return-value comment and
        # _encode_action_and_card_idx's docstring for why these specific
        # redundancies are worth watching for.
        hand = self.bc.hand
        state, total_living_hp = _encode_state(self.bc, hand)
        # encode_action_with_potion wraps _encode_action_and_card_idx (adds
        # is_potion/is_potion_discard action features + a potion_idx) --
        # called unconditionally, not gated behind potion_generator being
        # set, since it's a strict superset of the old behavior (a
        # 0-potions-held episode just gets EMPTY_POTION_INDEX everywhere,
        # same as the relic side's "no relics -> empty active-idx list").
        encoded = [encode_action_with_potion(self.bc, a, total_living_hp, hand) for a in legal]
        action_feats = [e[0] for e in encoded]
        card_idxs = [e[1] for e in encoded]
        monster_idxs = [e[2] for e in encoded]
        action_potion_idxs = [e[3] for e in encoded]
        relic_idxs, relic_mask = pad_relic_idxs(self._active_relic_idxs)
        # Potion identity captured fresh here (not once at reset(), unlike
        # relics) -- see reset()'s own comment on why.
        potion_idxs, potion_mask = pad_potion_idxs(capture_active_potion_idxs(self.bc))
        return {
            "state": state,
            "actions": legal,
            "action_features": action_feats,
            "action_card_idx": card_idxs,
            "action_monster_idx": monster_idxs,
            "action_potion_idx": action_potion_idxs,
            "potion_idxs": potion_idxs,
            "potion_mask": potion_mask,
            "relic_idxs": relic_idxs,
            "relic_mask": relic_mask,
        }

    def step(self, action) -> Tuple[dict, float, bool, dict]:
        phi_before = potential(self.bc)

        action.execute(self.bc)

        done = self.bc.outcome != sts.BattleOutcome.UNDECIDED
        phi_after = potential(self.bc)  # already 0.0 if done, by potential()'s own convention
        reward = W_SHAPE * (GAMMA * phi_after - phi_before)
        if done:
            reward += W_SHAPE * terminal_reward(self.bc, self.bc.turn)

        obs = self._observation() if not done else None
        info = {"outcome": self.bc.outcome, "player_hp": self.bc.player_hp}
        return obs, reward, done, info
