//
// Created by keega on 9/16/2021.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/functional.h>

#include <sstream>
#include <algorithm>
#include <limits>
#include <vector>
#include <array>
#include <unordered_map>
#include <memory>
#include <new>
#include <type_traits>
#include <random>
#include <cmath>
#include <cstdint>
#include <utility>
#include <tuple>
#include <optional>

#include "sim/ConsoleSimulator.h"
#include "sim/search/ScumSearchAgent2.h"
#include "sim/SimHelpers.h"
#include "sim/PrintHelpers.h"
#include "game/Game.h"
#include "sim/search/GameAction.h"

#include "slaythespire.h"


using namespace sts;
using namespace pybind11::literals;

pybind11::dict sts::py::NNCardsRepresentation::as_dict() const {
    return pybind11::dict("cards"_a=cards, "upgrades"_a=upgrades);
}
pybind11::dict sts::py::NNRelicsRepresentation::as_dict() const {
    return pybind11::dict("relics"_a=relics, "relic_counters"_a=relicCounters);
}
pybind11::dict sts::py::NNMapRepresentation::as_dict() const {
    return pybind11::dict("xs"_a=xs, "ys"_a=ys, "roomTypes"_a=roomTypes,
                          "pathXs"_a=pathXs, "burningEliteX"_a=burningEliteX,
                          "burningEliteY"_a=burningEliteY);
}
pybind11::dict sts::py::NNRepresentation::as_dict() const {
    return pybind11::dict("fixed_observation"_a=fixedObservation, "deck"_a=deck.as_dict(),
                          "relics"_a=relics.as_dict(), "potions"_a=potions,
                          "map"_a=map.as_dict(), "mapX"_a=mapX, "mapY"_a=mapY);
}

namespace {
    // --- native (C++) port of az_search.py's _heuristic_pick/
    // _heuristic_playout and env.py's potential()/terminal_reward() ---
    //
    // Added on top of state_key_bundle (see BattleContext's own binding
    // below) because profiling still found the search's rollouts dominated
    // by Python-side cost even after that fix: ~26% of total search time
    // in env.py's potential() alone (its `for i, m in enumerate(bc.monsters)`
    // re-copies the ENTIRE monster vector across the language boundary on
    // EVERY call, and it's called on every _dense_reward step), plus a
    // separate Python<->C++ crossing for EVERY action of EVERY rollout
    // (get_legal_actions, action.execute, and several more per candidate
    // action inside _heuristic_pick's own scoring loop). Silver Automaton
    // affords ~10,000 iterations/decision partly because its entire search,
    // heuristic rollout policy, and reward accounting are native C++ with
    // no language-boundary crossings anywhere in that loop; this collapses
    // this project's own rollout to the same shape -- ONE call per rollout
    // (heuristic_playout) instead of one call per action-per-rollout.
    //
    // THIS FILE IS NOW THE MAINTAINED, AUTHORITATIVE IMPLEMENTATION. It
    // started as a duplicate of formulas/constants that also live in
    // Python (env.py's W_HP/BETA/W_WIN/W_DEATH/W_SHAPE/
    // TURN_PENALTY_PER_TURN_ON_WIN/TURN_SURVIVED_BONUS_PER_TURN_ON_LOSS,
    // az_search.py's _heuristic_pick/_predicted_incoming_damage), but the
    // Python side is no longer kept in lockstep as a hard rule -- it's a
    // debugging aid now (see expectimax_search.py's choose_action_python
    // docstring), not a sync target. Accuracy/quality fixes land HERE
    // going forward; the Python functions may drift and that's expected,
    // not a bug. Validated field-for-field
    // (exact chosen action at every step, not just the final reward)
    // against the pure-Python implementation across many paired rollouts
    // sharing identical starting state before this replaced it as the
    // default rollout path.
    constexpr int NATIVE_ROLLOUT_MAX_ACTIONS = 200;
    // Moved up from its original spot next to nativeLeafFeatures's own definition so
    // nativePolicyNetScore's forward declaration (which needs the array type) can use it --
    // nativeLeafFeatures itself is still defined in its original spot further down.
    // 10 originally, widened to 30 on 2026-08-01. The first TEN entries are
    // frozen in order and meaning: g_params.vf* apply to them positionally
    // (nativeLeafValueEstimate reads f[0]..f[9]), so the linear "value" leaf mode
    // is unchanged by the extension and only the learned estimators see the rest.
    //
    // Why widen. The linear leaf estimate measured -19.14 +/- 1.16 HP against a
    // rollout at matched simulations, and still -14.96 at 4x the simulations.
    // That was read as "static evaluation loses to rollouts", but the ten
    // features are DECK-BLIND -- no hand, no piles, no relics, and 3 of 19 player
    // statuses -- so they cannot tell a hand of Strikes from a hand holding
    // Corruption. The same vector also feeds nativePolicyNetScore, so one thin
    // representation was choking both learned components; the distillation
    // post-mortem in docs/03-combat-search.md reaches the same conclusion from
    // the action side.
    static constexpr int NATIVE_LEAF_FEATURE_DIM = 30;
    constexpr double NATIVE_W_HP = 1.5;
    constexpr double NATIVE_BETA = 3.0;
    constexpr double NATIVE_W_WIN = 200.0;
    constexpr double NATIVE_W_DEATH = 400.0;
    constexpr double NATIVE_W_SHAPE = 0.1;
    constexpr double NATIVE_TURN_PENALTY_PER_TURN_ON_WIN = 0.5;
    constexpr double NATIVE_TURN_SURVIVED_BONUS_PER_TURN_ON_LOSS = 1.0;

    // Runtime-tunable search parameters -- as opposed to the constexpr
    // block just above (NATIVE_W_HP/BETA/W_WIN/W_DEATH/W_SHAPE/
    // TURN_PENALTY_PER_TURN_ON_WIN/TURN_SURVIVED_BONUS_PER_TURN_ON_LOSS),
    // which mirror env.py's actual PPO training reward and must NOT be
    // tuned independently of it (that would silently diverge the search's
    // reward semantics from what the policy network is trained against).
    // These were all originally `constexpr` (compile-time), which made
    // even manual tuning painfully slow -- every candidate value needed a
    // full C++ rebuild (~10-15s) before it could be tested at all (see
    // this session's own C_UCB/loss-progress-credit-weight sweeps, each a
    // sed-edit-rebuild-test loop). Runtime-mutable now specifically so an
    // automated tuner (CMA-ES, see lightspeed/tune_search_cma.py) can
    // evaluate a new parameter vector via one fast Python call
    // (set_search_params) instead of a rebuild -- turns each fitness
    // evaluation from "rebuild + run episodes" into just "run episodes".
    // Defaults below are exactly this session's own hand-tuned/chosen
    // values, so leaving every field at its default reproduces today's
    // validated behavior exactly.
    struct TunableParams {
        double cUcb = 1.5;
        double cUcbChance = 1.5;
        double wcChance = 1.0;
        double waChance = 0.5;
        double lossProgressCreditWeight = 150.0;
        // See nativeExpectimaxTerminalReward's own comment. 1.0 = no-op (matches
        // nativeTerminalReward's fixed 1:1 curHp weight exactly).
        double winHpWeight = 1.0;
        // Extra final-HP value only in Act 1's opening easy hallway pool:
        // Cultist, Jaw Worm, Two Louse, and Small Slimes. GameContext draws
        // exactly its first three Act 1 normal fights from this pool, making
        // this a targeted full-run safety dial. 0.0 is a strict no-op.
        double earlyActEasyPoolHpSafetyWeight = 0.0;
        // See nativeExpectimaxTerminalReward's own comment. 0.0 = off (ported from Silver
        // Automaton's evalWeights.potionWeight=11.0, but not adopted as a default here --
        // their absolute scale isn't meaningful in our own differently-scaled reward, see
        // winHpWeight's own comment on why proportions, not raw magnitudes, transfer).
        double potionScoreWeight = 0.0;
        // RAVE bias b in the Gelly/Silver MC-RAVE schedule beta = amafN / (N + amafN + 4*N*amafN*b^2)
        // (see nativeSelectIdx). Controls how fast AMAF influence decays as an edge accumulates its
        // own real visits: smaller b trusts AMAF longer. Inert unless g_useRave is on.
        double raveBias = 0.05;
        // Softmax temperature for the ROLLOUT's action pick. 0.0 = strict argmax (the original
        // behaviour). Above 0, the rollout samples proportionally to exp(score/T) instead of always
        // taking the top-scoring action. Motivation: with argmax, every rollout launched from the
        // same node replays essentially the same line (only chance nodes differ), so repeated
        // simulations return highly correlated leaf values -- lots of compute, little new
        // information. Silver Automaton's own rollout agent takes a `randomize` flag for the same
        // reason. Larger T = more varied lines but weaker play inside each rollout.
        double rolloutTemperature = 0.0;
        // Penalty per point of energy left unspent when a turn ends (BattleContext::endTurn does
        // energyWasted += player.energy, so this is the cumulative fight total). Measured on
        // THE_GUARDIAN @A20: 23.7% of our turns ended with >=1 unspent energy, mean 0.389/turn --
        // roughly 4 energy, or 1-2 unplayed cards of block/damage, thrown away per fight. Applied
        // in the TERMINAL reward rather than the potential on purpose: potential-based shaping
        // telescopes to Phi(terminal) - Phi(start) over a full playout, so a potential term would
        // be policy-invariant and change no decision. Silver Automaton carries the same idea as
        // evalWeights.energyWasteWeight = 1.75 (loss branch only; applied to both branches here
        // since the HP deficit this targets is in fights we already WIN). 0.0 = off by default.
        double energyWasteWeight = 0.0;
        // Weight on the player's CURRENT BLOCK in nativeExpectimaxPotential. Was absent entirely,
        // which made a state holding 15 block score identically to the same state holding 0 --
        // so playing a Defend produced no measurable improvement while playing an attack visibly
        // lowered monster HP. Measured consequence on TWO_LOUSE @A20: on 21% of turns the search
        // ended the turn with a mean 1.71 energy unspent and playable cards in hand (29 Strikes,
        // 20 Defends declined), and that rate did NOT fall with 13x more simulations (21.2% at
        // 150 sims, 25.0% at 2000) -- i.e. the search was correctly optimizing an objective that
        // could not see defense, not failing to search deeply enough. Block is worth less than HP
        // point-for-point (it expires at end of turn and only pays off against damage that
        // actually arrives), so the tuned value is expected below NATIVE_W_HP.
        double blockWeight = 0.0;
        // Enemy block is temporary durability.  It is deliberately separate
        // from player block so tuning can decide how aggressively the search
        // should value breaking a guarder's armor at a rollout/truncation
        // leaf.  0.0 preserves existing reward semantics; a positive value
        // is the targeted Spheric Guardian experiment.
        double enemyBlockWeight = 0.0;
        // Bonus for playing a Vulnerable-applier while a living target still LACKS Vulnerable (see
        // isVulnerableApplier). Rewards applying the 1.5x damage multiplier before spending
        // attacks into it rather than after -- pure sequencing value that the type-level scores
        // (attackBase/skillBase/powerScore) structurally cannot express, since they treat every
        // ATTACK alike. Targets the measured symptom on small fights: on TWO_LOUSE @A20 Silverbot
        // closes in 2.20 turns against our 2.67 while banking equal block, and the extra half-turn
        // of enemy attacks is the HP we lose. 0.0 = off by default.
        double vulnerableApplyBonus = 0.0;
        // Weak counterpart of vulnerableApplyBonus (see nativeWeakApplyBonus). Kept separate
        // because Weak reduces incoming damage while Vulnerable amplifies outgoing, so the two
        // scale with different quantities and must be tunable independently. 0.0 = off.
        double weakApplyBonus = 0.0;
        // Scales nativePowerPerTurnValue, itself multiplied by the fraction of enemy HP still
        // standing (HeuristicContext::monsterHpRatio) as a remaining-fight proxy -- a Power played
        // at full enemy HP has many turns to pay off, the same Power played on a nearly-dead enemy
        // has none. Before this, every POWER scored a flat powerScore, so Demon Form and Evolve
        // were indistinguishable to the rollout policy except through the learned per-card prior.
        // 0.0 = off by default, so the build is behaviour-identical until tuned.
        double powerPerTurnValueWeight = 0.0;
        // Scales nativePowerImmediateValue. Deliberately NOT multiplied by the remaining-fight
        // proxy: Inflame's Strength lands the moment it resolves, so it keeps its value on the
        // last turn of a fight where a per-turn Power is worthless. 0.0 = off by default.
        double powerImmediateValueWeight = 0.0;
        // Win-branch HP credited as a FRACTION of max HP, scaled by 100 so its magnitude is
        // comparable to the absolute-HP terms. Fixes a real mismatch: the tuning fitness scores
        // player_hp/max_hp, but the terminal reward counts absolute curHp -- so with the encounter
        // set spanning 50..130 max HP, the search treats 10 HP as equally valuable in a 50-HP Jaw
        // Worm fight (20% of the pool) and a 130-HP Time Eater fight (8%), while the objective
        // grading it disagrees. Additive alongside the existing absolute term rather than replacing
        // it, so 0.0 reproduces current behaviour exactly and the two can be traded off by tuning.
        double winHpFractionWeight = 0.0;
        // Multiplier on NATIVE_W_WIN, the flat victory bonus. Sets how much a win is worth relative
        // to the HP preserved getting it -- the risk/greed dial. Left compiled-in and untunable
        // until now, which meant every HP-vs-win tradeoff explored today could only move one side
        // of that ratio.
        double winBonusWeight = 1.0;
        // See nativeExpectimaxTerminalReward's own comment. 0.0 = ignore the real STS boss-heal
        // entirely (matches this project's current single-fight eval methodology); 1.0 = full
        // credit (matches Silver Automaton's own evaluateEndState, correct for real full-run play).
        double bossHealCreditWeight = 0.0;
        // Blend between two proxies for "how much fight is left to collect a
        // per-turn Power's value in". At 0.0 this reproduces the original
        // behaviour exactly -- multiply by monsterHpRatio, the FRACTION of enemy
        // HP still standing. That fraction is scale-invariant and fight length is
        // not: a 300 HP boss at full health and a 45 HP Jaw Worm at full health
        // both give 1.0, while one fight runs twenty turns and the other three.
        //
        // Measured on the human benchmark (556 test fights, 2026-07-31): scaling
        // both Power weights by 2x improved boss fights by 1.06 HP each and cost
        // non-boss fights 0.54 each. That is the signature of one global scalar
        // serving two very different fight lengths, and it is why CMA-ES settled
        // where it did -- the average is dominated by the 456 short fights.
        // At 1.0 the multiplier is instead min(1, remainingEnemyHp/powerHorizonHp),
        // which is absolute rather than fractional and so separates them.
        double powerHorizonWeight = 0.0;
        // Multiplier on the whole POWER score, applied only on boss encounters.
        // 1.0 is a no-op. An alternative to powerHorizonWeight rather than a
        // complement: both exist to give Powers more value in long fights, but
        // this names the category directly where the horizon infers the property
        // from remaining enemy HP. Naming the category is exact for bosses and
        // blind to long ELITES (Reptomancer, Awakened One's second phase, Act 4);
        // the horizon catches those but can misjudge a high-HP short fight.
        // Measured against each other on the human benchmark -- see
        // docs/03-combat-search.md.
        double bossPowerMultiplier = 1.0;
        // Enemy HP at which a fight counts as "long enough for a Power to pay
        // off in full". Roughly an Act 1 boss; anything smaller scales down.
        double powerHorizonHp = 150.0;
        // Multiplier on NATIVE_TURN_PENALTY_PER_TURN_ON_WIN within the SEARCH's win-branch
        // terminal reward only (nativeTerminalReward itself is shared with PPO training, left
        // alone -- same pattern/reasoning as winHpWeight). 1.0 = no-op. Exists because that
        // constant is otherwise a hardcoded constexpr with no way to rebalance it: winHpWeight
        // (~4) scaled the win score's HP term up ~4x without touching the turn term, so the
        // finish-fast-vs-preserve-HP tradeoff it controls is now weighted very differently than
        // when 0.5 was chosen. Silver Automaton tunes their equivalent (victoryTurnPenalty) and
        // documents it as load-bearing in both directions: large enough to close out a winnable
        // fight rather than banking micro-HP by stalling, small enough not to trade real HP for
        // speed.
        double winTurnPenaltyWeight = 1.0;
        // Flat penalty per monster still standing at a LOSS, on top of the existing pooled-HP
        // lossProgressCreditWeight credit -- ports Silver Automaton's own aliveScore
        // (evaluateEndState: monstersAlive * -aliveWeight, their tuned value 3.4). Adds a
        // distinction the pooled HP ratio structurally cannot express: killing one of two
        // monsters outright and chipping both to half score the SAME under HP-ratio alone, even
        // though the former is strictly better (a dead monster deals no further damage). Only
        // meaningful in multi-monster encounters; single-monster fights always have exactly one
        // alive at a loss, making it a constant offset there. 0.0 = off until validated.
        double aliveMonsterPenaltyWeight = 0.0;
        double brewingThreatEstimate = 8.0;
        double attackBase = 10.0;
        double attackFinishOffScale = 5.0;
        double attackBlockPenaltyScale = 0.15;
        double aoeBonus = 6.0;
        double skillBase = 4.0;
        double skillDangerScale = 30.0;
        double skillHastePenalty = 5.0;
        // Exact one-hit damage supplied by the engine's own Strength/Weak/
        // Vulnerable-aware calculation. The type-level rollout score otherwise
        // knows an action is an Attack but not whether it hits for 6 or 30.
        // 0.0 preserves the existing policy until the term is validated.
        double attackDamageScoreWeight = 0.0;
        // Immediate block that prevents the currently predicted attack. This
        // is intentionally capped at unblocked incoming damage: block that
        // expires unused should not be valued, and generic Skill scoring has
        // no way to make that distinction. 0.0 is behavior-identical.
        double directBlockScoreWeight = 0.0;
        // Immediate self-damage from a card play. Without this, Offering and
        // Bloodletting are ranked only by their generic card priors despite
        // directly consuming the HP resource full-run planning must preserve.
        // 0.0 is behavior-identical until validation enables it.
        double selfDamageScorePenalty = 0.0;
        // Flat bonus for playing a poison-applying card (Silent only). Silent's
        // damage scales through cumulative poison rather than upfront hits, and
        // the generic ATTACK scoring has no way to value that delayed payoff.
        // 0.0 = off by default (behavior-identical).
        double silentPoisonApplyBonus = 0.0;
        // Potion scoring in the ROLLOUT. Until these existed, nativeScoreAction opened with a
        // flat `score = 5.0` for every non-CARD action, so the rollout could not tell drinking a
        // Fire Potion from ending the turn -- nor from THROWING THE POTION AWAY, since a discard
        // (targetIdx > 5, see Action::execute) scored the same 5.0. potionScoreWeight (tuned to
        // 29.7) prices potions in the TERMINAL evaluation only, which never reaches the rollout
        // that docs/03-combat-search.md measures as the ceiling on combat. Found by
        // lightspeed/_probe_card_identity.py, where collapsing every non-CARD action into one
        // `is_other` symbol cost more top-1 accuracy than card identity did.
        // rolloutPotionBase and rolloutNonCardBase both default to the old 5.0 and every other
        // weight to 0.0, so this branch reproduces the previous behaviour exactly until tuned --
        // same convention as perCardWeightScale and cPuct.
        double rolloutPotionBase = 5.0;
        // END_TURN (scored via nativeHeuristicVisitOrder) and card-select options, i.e. the other
        // half of what the old flat 5.0 covered. Separate from potions so raising one does not
        // silently raise the other.
        double rolloutNonCardBase = 5.0;
        // Mirrors skillDangerScale: a potion is worth most on the turn that would otherwise hurt.
        // The human benchmark says the same thing from outside the engine -- his damage is HIGHER
        // on fights he drinks in (22.3 vs 11.9), because he banks potions for the hard ones.
        double rolloutPotionDangerScale = 0.0;
        // Mirrors attackFinishOffScale, which the tuner moved 5 -> 20.4 and is the single largest
        // per-action term in the file. Only targeted potions (Fear/Fire/Poison/Weak, see
        // potionRequiresTarget) carry a real monster in targetIdx.
        double rolloutPotionFinishOffScale = 0.0;
        // Discarding a potion mid-combat throws the resource away for nothing -- there is no slot
        // pressure inside a fight. It scored identically to drinking one until this existed.
        double rolloutPotionDiscardPenalty = 0.0;
        // MAST -- see the block above nativeScoreAction. mastWeight is in heuristic-score
        // points per standard deviation of the search's own return distribution; 0.0 is
        // off and behaviour-identical. mastMinVisits is how many times a move must have
        // been played in THIS search before its average is trusted at all: at 1 the table
        // is pure noise early, and the rollout would chase whichever move happened to be
        // played in the first good simulation.
        double mastWeight = 0.0;
        double mastMinVisits = 3.0;
        // How much of a decision node's backed-up value comes from its BEST child rather
        // than from the sampled return along the path taken.
        //
        // 0.0 is the Monte-Carlo backup this search has always used: a node's value is the
        // running mean of every sampled return through it, so the estimate averages over
        // actions UCB1 deliberately explored to be bad. Keller & Helmert (ICAPS 2013) name
        // the pitfall -- one visit to a child far worse than its optimal sibling biases the
        // parent for many trials -- and 1.0 is their fix, Max-Monte-Carlo backup (MaxUCT):
        // V(decision) = max over actions of Q(action), with Q still the visit-weighted mean
        // over that action's sampled outcomes, which is exactly what W[idx]/N[idx] already
        // holds. Their UCT*/DP-UCT go further with solve labelling, but that needs the
        // declarative transition model P(s'|s,a); this engine has only a generative one, and
        // Max-Monte-Carlo is explicitly the variant that does not. UCB1's optimality proof
        // carries over unchanged, since it never stops exploring.
        //
        // The blend is a generalisation, not something the paper proposes. It exists because
        // a pure max is optimistic: with few samples per child it inherits the maximization
        // bias that motivates Double Q-learning, and at 100 simulations across ~10 actions
        // that is a real risk rather than a theoretical one. A weight between the endpoints
        // is the hedge, and CMA-ES can find it.
        //
        // Why this should matter here specifically: combat is a single-agent MDP with
        // scripted monsters, so V(s) = max_a Q(s,a) is the correct Bellman value and the
        // average is evaluating a policy nobody intends to play. The argument that rescues
        // the Monte-Carlo backup is asymptotic, and 100 simulations is not asymptotic.
        double backupMaxWeight = 0.0;
        // Draw-order honesty. 0.0 is the search this project has always run, which is
        // CLAIRVOYANT: run_mcts_search roots its tree in a full copy of the live
        // BattleContext including the ORDERED draw pile, so every simulation knows exactly
        // which cards are coming. Measured at -3.78 +/- 0.84 HP/fight (t = -4.50), and it is
        // why our combat appeared to beat Silverbot's -- blind we score -12.25 against their
        // honest -6.58. Non-zero permutes the pile's ORDER (never its contents) at the three
        // points the future actually enters the search: each DPW chance sample, each
        // rollout, and any in-tree action that draws.
        //
        // This is deliberately NOT one determinization per decision. Silverbot measured that
        // shape at 35.2% against 69.4% cheating, and in-tree belief averaging at 56.2% --
        // committing to a single sampled order is far worse than either cheating or
        // averaging, because the search then plans a line that only works under the order it
        // happened to draw (strategy fusion). Here every chance sample gets its own order and
        // DPW widening averages over them, which is the 56.2% shape rather than the 35.2%
        // one.
        //
        // EXPECT THIS TO COST HP. It removes information; that is the point. The number to
        // watch is not the delta at fixed simulations but whether the SIMULATION CURVE
        // un-flattens: combat is measured flat from 43 to 1500 sims, and perfect draw
        // knowledge collapsing chance-node variance is a leading suspect for why. Silverbot's
        // honest engine defaults to a far larger budget for exactly this reason.
        // 0 = off (clairvoyant). 1 = per-sample order: the pile is permuted once
        // per DPW chance sample and once per rollout, so the tree AVERAGES over
        // orders rather than committing to one per decision. That is already the
        // good half of Silverbot's +21pp result -- their losing variant sampled a
        // single order per decision and reused it for every simulation.
        //
        // 2 = LAZY, the canonical-CardPile semantics. The remaining leak at 1 is
        // that a sample's order is inherited by everything BELOW it, so a subtree
        // can plan around draws it should not know. At 2 the pile is re-permuted
        // after every action that drew, anywhere in the tree or the rollout, so
        // each draw is an independent uniform sample from the remaining multiset
        // -- which is what an unordered pile with lazy draws would give, without
        // restructuring CardManager (shared with the real-game path).
        double honestDrawOrder = 0.0;
        // Trial length: how many turns past the current one a simulation may reach
        // before the tree cuts off and the static leaf estimate is applied
        // (`node->bc.turn >= maxTurn` in nativeSimulate). 20 is the value this
        // search has always used, as the compile-time NATIVE_MAX_TURNS_PER_SEARCH.
        //
        // This is the fifth ingredient of the THTS framework (Keller & Helmert,
        // ICAPS 2013) -- heuristic function, backup function, action selection,
        // outcome selection, trial length -- and the only one never varied here.
        // Shortening it is what turns their DP-UCT into UCT*, on the reasoning
        // that a limited trial length "distributes resources better in the search
        // space" by investigating states nearer the root more thoroughly. Worth
        // testing here because extra simulations demonstrably fail to buy depth
        // under honest draws, and widening has been ruled out as the cause.
        double searchMaxTurns = 20.0;
        // How many root actions sequential halving is allowed to consider, chosen by
        // Gumbel-Top-k over the heuristic priors. 0 (or >= the legal-action count) keeps
        // every action, which is what this search did before the parameter existed and is
        // therefore the behaviour-identical default. See the candidate-set block in
        // nativeRunMctsSearchSeqHalving for why a smaller number is worth trying.
        double seqHalvingCandidates = 0.0;
        double powerScore = 6.0;
        // A normalized, per-card prior derived from Silver Automaton's own
        // cardPlayPriorities. 0.0 keeps the independently trained pick-rate
        // prior as the sole card-specific signal; nonzero values let tuning
        // Optional boss-specific override for the Silver card-order prior.
        // Hallways and elites benefit from a stronger copy of the compact
        // priority signal, while long boss mechanics (especially Time Eater)
        // need the conservative global setting.  A negative value means
        double endTurnTimeWarpRiskScore = 11.0;
        double skillHasteDangerThreshold = 0.1;  // was a hardcoded literal in nativeScoreAction's SKILL branch until made tunable
        double perCardWeightScale = 0.0;  // see cardPickRateWeight's own comment -- 0.0 (off) by default so this rebuild is behavior-identical until explicitly tuned
        // Weight on cardPlayRank, the play-priority table fitted from THIS
        // project's own search decisions (lightspeed/_fit_play_priority.py).
        // Fills the slot the borrowed Silver Automaton table vacated on
        // 2026-08-01 (removal measured -1.20 +/- 0.49 HP). 0.0 = off, so the
        // rebuild is behaviour-identical until the config enables it. The old
        // silver weight tuned to 4.3-5.0, which sets the sweep scale.
        double cardPlayPriorWeight = 0.0;
        // Negative means "inherit cardPlayPriorWeight". Bosses kept a separate
        // weight historically because the CMA run split them; preserved so a
        // future tune can split them again.
        double bossCardPlayPriorWeight = -1.0;
        // Paired determinizations across root candidates (honest regime only).
        // Sequential halving compares candidates by mean value, and every
        // difference of means pays the variance of BOTH arms unless their
        // samples are correlated. With honest draws the dominant sampling noise
        // is the draw-order permutation, and it is freshly drawn per chance
        // sample -- so sibling candidates are compared across DIFFERENT
        // futures. This keys the permutation by (search seed, per-candidate
        // visit index, turn) instead of the global stream: visit k of every
        // root candidate then sees the same draw stream wherever the pile
        // multiset matches, and the comparison differences out the shared
        // noise, the same trick _param_ab plays between configs. 0 = off.
        double pairedDeterminization = 0.0;
        // Merge root/tree candidates that are provably the same move. Two
        // copies of Strike in hand are two ACTIONS but one DECISION; sequential
        // halving splits its per-candidate budget between them, which is how a
        // 5-card starter hand wastes a third of its root budget and how the
        // savable-death hands (three Angers, two Strikes) starve the block
        // candidate that mattered. The merge key is every CardInstance field
        // except uniqueId, so cards with per-instance state (Ritual Dagger,
        // Rampage, Genetic Algorithm -- all in specialData) never merge unless
        // that state is identical too. 0 = off.
        double mergeDuplicateActions = 0.0;
        // Two-stage escalation: rerun the root search at escalationSims when
        // the decision is BOTH dangerous and contested. Uniform danger-gating
        // measured +2.78/+0.87 (train killers / full val) but escalated 31-48%
        // of decisions for 3-4x wall clock, because A20 danger is common; the
        // metareasoning literature and the duplicate-merge refutation both say
        // the budget belongs on contested decisions specifically. Danger:
        // unblocked >= escalationDangerFrac * hp, or the Heart is present
        // (Beat of Death is invisible to move telegraphs). Contested: top-2
        // survivor mean values within escalationQgap. 0 = off.
        double escalationSims = 0.0;
        double escalationQgap = 0.25;
        double escalationDangerFrac = 0.25;
        // Level-1 belief-MDP port (docs/13): merge DPW chance-node SIBLINGS
        // whose states are the same information set. Silverbot's search keys
        // nodes by information set -- pile as unordered multiset -- and >half
        // of their chance samples merge, which is where their honest budget
        // slope comes from (statistics pool on one posterior instead of
        // fragmenting across private determinized subtrees). Sibling scope
        // (same parent, same action) is safe by construction: visible zones
        // share a history, so equality differs only in hidden-order noise.
        //   0 = off; 1 = draw pile compared as multiset; 2 = +hand as
        //   multiset (drawn order is visible but gameplay-inert); 3 =
        //   +discard as multiset (order matters only through reshuffles,
        //   which honest mode re-randomizes anyway).
        double mergeChanceOutcomes = 0.0;
        // Honest-regime DPW widening. The shipped wc/wa pair (0.320/0.035)
        // caps every chance node at ONE sibling -- correct for clairvoyant
        // play at 100 sims (chance = monster rolls; the cap re-measured
        // -2.82 +/- 0.76 WORSE when lifted there), catastrophic under honest
        // draws (the tree commits to a single determinization per node; the
        // thrice-replicated flat honest budget curve was exactly this).
        // Honest play switches to this pair: -0.79 -> +9.11 slope on train
        // killers (t = 10.7), +4.08 +/- 0.49 (t = 8.41) on full val at 900.
        // Values are silverbot's own widening constants. -1 = inherit wc/wa.
        double honestWcChance = 4.6;
        double honestWaChance = 0.37;
        // Attacks into an INTANGIBLE monster deal 1 per hit; the rollout's
        // attack terms score them at full value, so every simulated future
        // happily burns damage into intangible turns and the tree cannot see
        // the waste. Measured before this term existed: 6.1 attacks landed
        // into intangible per Nemesis fight at 500 sims, and Nemesis carried
        // -11.4/fight of the matched-500 silverbot residual. Flat score
        // penalty on ATTACK cards whose chosen target is currently
        // intangible; block/setup/AoE-elsewhere lines win those turns
        // instead. 0 = off.
        double intangibleAttackPenalty = 0.0;
        // Companion to intangibleAttackPenalty: ARTIFACT charges eat the next
        // debuff, but the vulnerable/weak apply bonuses reward the play at
        // full value -- so rollouts happily Bash into a 3-charge Sentry.
        // 1 = the apply bonuses treat an artifact-holding target as
        // already-debuffed (bonus zero; the attack's damage terms still score,
        // only the phantom debuff credit disappears). 0 = off.
        double artifactAwareDebuffs = 0.0;
        // Battle-long tree reuse (silverbot's rerootAt, docs/13 piece 5). The
        // arena persists across a battle's decisions; the next search matches
        // the REAL post-action state against the previously chosen action's
        // stored children (info-set key under honest draws, exact key
        // otherwise) and re-roots on the matching subtree, keeping every
        // statistic its simulations earned. A miss -- including every battle
        // boundary -- clears the arena and starts cold, so correctness never
        // depends on the match firing. The old 1.34x reuse ceiling was
        // measured on cap-1 fragmented trees; widened honest trees are worth
        // keeping. 0 = off.
        double treeReuse = 0.0;
        // Survival mode for the rollout player. The Heart oracle grid
        // (2026-08-04) localized the last big deficit to the ROLLOUT POLICY:
        // clairvoyant at 3000 sims still dies 74% of Heart fights (human: 0%)
        // because the simulated player's attack-dominant scoring cannot
        // survive burst windows, flattening every leaf value the tree sees.
        // When unblocked telegraph >= survivalModeThreshold * curHp, ATTACK
        // scores are multiplied by survivalModeAttackScale -- a mode switch
        // the existing additive danger nudges could never amount to.
        // threshold <= 0 disables.
        double survivalModeThreshold = 0.0;
        double survivalModeAttackScale = 0.25;
        // End-state value family (2026-08-04): value that is not HP at battle
        // end, which an HP-only objective optimizes straight through. All
        // default-off; gates are TARGETED SLICES (Feed decks / thief fights /
        // Writhing Mass) measuring the behavior itself plus a paired HP
        // no-harm guard, since the benchmark objective cannot see these.
        // Bonus on a DRAW card when energy remains after paying for it, so the
        // cards it finds can still be played this turn. Zero when the draw
        // would be the turn's last action (nothing left to use it on).
        double drawFirstBonus = 0.0;
        // Scales a one-turn strength stripper by the fraction of current HP
        // the telegraph threatens: worthless on a safe turn, decisive on the
        // burst turn. Mirror of intangibleAttackPenalty.
        double burstDebuffTimingWeight = 0.0;
        double maxHpGainWeight = 0.0;    // per max-HP point vs the search root (Feed, Darkstone)
        double goldDeltaWeight = 0.0;    // per gold vs the root (thief escapes lose it; kills refund)
        double parasitePenaltyWeight = 0.0;  // flat, when Writhing Mass's implant fired (Omamori negates)

        // PUCT-style prior bonus in nativeSelectIdx, on top of (not replacing) the existing
        // UCB1 exploration term -- see nativeSelectIdx's own comment for the formula. cPuct=0.0
        // by default (off, byte-identical search to before this was added) since it's a pure
        // addition, same convention as perCardWeightScale above. puctTemperature has no effect
        // while cPuct is 0, but needs a sane nonzero default for when it's tuned on: nativeScoreAction's
        // raw scores range roughly 4-30 across action types, so 10.0 keeps the softmax over a
        // node's actions meaningfully peaked without collapsing to near-argmax.
        double cPuct = 0.0;
        double puctTemperature = 10.0;

        // Blend weight for the learned rollout-scoring net (nativePolicyNetScore) added into
        // nativeScoreAction -- see that function's own comment. 0.0 by default (off); has no
        // effect at all unless a net is also loaded via load_policy_net.
        double policyNetWeight = 0.0;

        // "We have enough block" gate -- see HeuristicContext::blockSufficient's own comment.
        // Ported from Silver Automaton's STS_BLOCK_OFFSET (default 4, no act-adjustment on our
        // side -- see that field's comment for why). Unlike most params above, this one is ON
        // by default (nonzero-effect margin, real suppression penalty) rather than defaulting to
        // a no-op -- CMA-ES's own x0/PARAM_NAMES handling already treats an unlisted param as
        // "not searched, stays at this literal default," so shipping a real starting guess here
        // (rather than 0.0) is what lets a direct A/B test measure its effect before any tuning
        // run touches it at all.
        double blockSufficiencyMargin = 4.0;
        // Applied (subtracted) to a defensive card's score (see isDefensiveCard) only
        // when blockSufficient is true -- large enough that a defensive card (skillBase=4 +
        // at-most-small dangerFraction term, since blockSufficient implies low danger) reliably
        // scores below any attack (attackBase=10+) or non-defensive skill, without being an
        // unconditional hard filter (a defensive card is still legal and still gets picked if
        // it's the only playable option, since scoring just picks the max of whatever's legal).
        double defensiveCardSuppressionPenalty = 8.0;

        // Leaf value-function weights -- used ONLY by nativeLeafValueEstimate
        // (leaf_eval_mode "value"/"truncated"), never touched in the default
        // "rollout" mode, so these have ZERO effect on production search until
        // that mode is selected. Defaults reproduce nativeExpectimaxPotential's
        // original 4-feature shape (vfHp=NATIVE_W_HP, vfMonsterHp=1,
        // vfIncoming=NATIVE_BETA, all enriching features 0) so a fresh "value"
        // run starts exactly at the un-enriched baseline and CMA-ES improves
        // from there (see tune_value_leaf.py).
        double vfHp = 1.5;          // == NATIVE_W_HP
        double vfMonsterHp = 1.0;
        double vfIncoming = 3.0;    // == NATIVE_BETA
        double vfBlock = 0.0;
        double vfEnergy = 0.0;
        double vfStrength = 0.0;
        double vfDexterity = 0.0;
        double vfAlive = 0.0;
        double vfTurn = 0.0;
        double vfMetallicize = 0.0;
    };
    TunableParams g_params;

    // Compact, explicitly early-Act card corrections layered on top of the
    // global pick-rate and Silver-order priors. The calibration resources use
    // max HP 50/70/85 for Act 1 basic/elite/boss respectively, while Act 2
    // begins at 90, so this threshold cleanly isolates the early-game policy
    // that carries the largest remaining HP-chip gap. Kept outside
    // TunableParams because it is a sparse per-card vector, not one scalar.
    constexpr int EARLY_ACT_BIAS_MAX_HP = 85;
    std::array<double, 372> g_earlyActCardBias {};
    bool g_hasEarlyActCardBias = false;

    // Root-allocation strategy: false = plain UCB1 over the whole budget (the long-standing
    // behaviour every tuned parameter set to date was measured under), true = sequential halving
    // (see nativeRunMctsSearchSeqHalving). Defaults OFF so this build is behaviour-identical until
    // explicitly switched on for an A/B -- turning it on changes what root visit counts mean, so
    // params tuned under one setting are not automatically valid under the other. Same
    // process-global mutability rule as g_params: set it before any concurrent search.
    bool g_useSeqHalving = false;

    // NativeStateKey is intentionally a compact heuristic key, not a full
    // BattleContext serialization. It omits gameplay-relevant state such as
    // card-instance fields, potion inventory, many powers, relic counters and
    // RNG state, so merging equal keys can merge different futures. Keep state
    // merging disabled until a complete canonical key is available and tested.
    bool g_useStateMerging = false;

    // RAVE / AMAF blending in nativeSelectIdx (see MctsNode::amafN's own comment). Defaults OFF so
    // the build stays behaviour-identical until switched on for an A/B; like g_useSeqHalving it
    // changes what the selection statistics mean, so parameter sets are not transferable across
    // the setting without re-measuring. Same process-global mutability rule as g_params.
    bool g_useRave = false;

    double nativePotential(const BattleContext &bc) {
        if (bc.outcome != Outcome::UNDECIDED) {
            return 0.0;
        }
        double phi = NATIVE_W_HP * bc.player.curHp;
        for (int i = 0; i < bc.monsters.monsterCount; ++i) {
            const Monster &m = bc.monsters.arr[i];
            if (m.halfDead) {
                phi -= m.maxHp;
                continue;
            }
            if (m.curHp <= 0) {
                continue;
            }
            const auto info = m.getMoveBaseDamage(bc);
            phi -= m.curHp + NATIVE_BETA * info.damage * info.attackCount;
        }
        return phi;
    }

    double nativeTerminalReward(const BattleContext &bc, int turnAtTerminal) {
        if (bc.outcome == Outcome::PLAYER_VICTORY) {
            return NATIVE_W_WIN + bc.player.curHp - NATIVE_TURN_PENALTY_PER_TURN_ON_WIN * turnAtTerminal;
        }
        return std::min(-1.0, -NATIVE_W_DEATH + NATIVE_TURN_SURVIVED_BONUS_PER_TURN_ON_LOSS * turnAtTerminal);
    }

    // Combined current/max HP ratio across all real (non-INVALID) monsters, win or lose --
    // mirrors Silver Automaton's own getNonMinionMonsterCurHpRatio (BattleSearcher.cpp), read
    // directly while investigating why more simulation budget wasn't moving Time Eater's win
    // rate at all (flat 0-1/15 from 200 to 1600 sims -- see conversation). The answer wasn't in
    // heuristic scoring: Silverbot's LOSS-branch evaluation gives partial credit for progress
    // made -- (1 - monsterHpRatio) * monsterDamageWeight -- while ours (nativeTerminalReward,
    // mirroring env.py's real terminal_reward -- shared with PPO training, out of scope to
    // change) scores EVERY loss as a near-flat catastrophic constant regardless of whether the
    // boss was chipped to 5% HP or untouched. In a genuinely hard, long fight where most/all
    // heuristic rollouts end in a loss, that flat penalty gives UCB1 almost no gradient to
    // climb -- every branch backs up a similarly bad value, so search can't tell "died turn 3
    // having done nothing" from "died turn 15 with the boss nearly dead", and more simulations
    // just samples the same undifferentiated noise more precisely instead of finding better
    // lines. This fixes that specifically for the search (see
    // nativeExpectimaxTerminalReward below), not env.py/nativeTerminalReward itself.
    double nativeMonsterHpRatio(const BattleContext &bc) {
        long long curTotal = 0;
        long long maxTotal = 0;
        for (int i = 0; i < bc.monsters.monsterCount; ++i) {
            const Monster &m = bc.monsters.arr[i];
            if (m.id == MonsterId::INVALID) {
                continue;
            }
            curTotal += std::max(0, static_cast<int>(m.curHp));
            maxTotal += m.maxHp;
        }
        if (maxTotal <= 0) {
            return 0.0;
        }
        return static_cast<double>(curTotal) / static_cast<double>(maxTotal);
    }

    // search-only terminal evaluation: identical to nativeTerminalReward on a WIN, but adds a
    // flat, bounded credit on a LOSS proportional to how much monster HP was depleted before
    // dying -- see nativeMonsterHpRatio's comment for why. g_params.lossProgressCreditWeight is
    // sized well below NATIVE_W_DEATH so a "fully depleted boss" loss (ratio -> 0, full credit)
    // still scores far worse than any win (worst case roughly -400+150+turn bonus vs a win's
    // +200ish) -- this differentiates losses from each other, it never makes losing look
    // better than winning.
    //
    // g_params.winHpWeight (default 1.0, a no-op matching nativeTerminalReward's own fixed 1.0-
    // per-HP): extra weight on final HP within the WIN branch ONLY, applied as an addition on top
    // of nativeTerminalReward's own curHp term rather than by touching NATIVE_W_WIN/curHp there
    // directly -- nativeTerminalReward is shared with env.py's real terminal_reward (PPO training
    // reward), out of scope to change (see nativeMonsterHpRatio's own comment for the same
    // reasoning re: the loss branch). Motivation: found while comparing matched-simulation-count
    // full fights against Silver Automaton's own engine -- their win-branch score is
    // winBonus(53) + curHp, ours is NATIVE_W_WIN(200) + curHp, so curHp is a much smaller
    // PROPORTION of our total win score (~39% at curHp~130) than theirs (~71%) despite an
    // identical *absolute* 1:1 weight -- and the HP-efficiency gap measured against their engine
    // did NOT shrink with 6.67x more search budget (150->1000 sims/decision), which rules out an
    // exploration/convergence explanation and points at exactly this kind of proportional-
    // dilution-in-backed-up-values issue instead. ~4.0 roughly matches their proportion at a
    // typical ~130 HP scale; see tune_search_cma.py/scratchpad sweep scripts for the actual
    // validated value.
    // Real STS act-transition heal after a BOSS ROOM victory (full heal below ascension 5, 75% of
    // missing HP at ascension 5+ -- ported from Silver Automaton's own BattleContext::
    // postBattleHealedHp). No gameContext/curRoom here to distinguish a boss fought via an event
    // (no heal) from one fought in its own room -- same fallback their own code uses when it
    // can't check the room either (a null gameContext, their standalone-battle-harness case).
    double nativePostBattleHealedHp(const BattleContext &bc) {
        if (!isBossEncounter(bc.encounter) || bc.player.hasRelic<RelicId::MARK_OF_THE_BLOOM>()) {
            return static_cast<double>(bc.player.curHp);
        }
        if (bc.ascension >= 5) {
            const double healAmount = std::round((bc.player.maxHp - bc.player.curHp) * 0.75);
            return std::min(static_cast<double>(bc.player.curHp) + healAmount, static_cast<double>(bc.player.maxHp));
        }
        return static_cast<double>(bc.player.maxHp);
    }

    // The exact Act 1 weak-enemy pool used by GameContext::generateWeakMonsters.
    // Encounter identity, rather than floor number, remains correct when shops
    // or events occur between the first three hallway fights.
    bool nativeIsAct1EasyPoolEncounter(MonsterEncounter encounter) {
        switch (encounter) {
            case MonsterEncounter::CULTIST:
            case MonsterEncounter::JAW_WORM:
            case MonsterEncounter::TWO_LOUSE:
            case MonsterEncounter::SMALL_SLIMES:
                return true;
            default:
                return false;
        }
    }

    // g_params.bossHealCreditWeight (0=ignore the heal entirely/pure raw-HP conservation, matching
    // this project's current single-isolated-fight evaluation methodology; 1=full credit for the
    // heal, matching Silver Automaton's own evaluateEndState and what's mathematically correct for
    // real full-run play) -- a TUNABLE blend rather than the hard on/off tried and reverted earlier
    // today (fully applying it regressed measured HP-efficiency -16 to -23pp on every boss
    // encounter, since it makes the search genuinely indifferent to boss-fight HP margins while our
    // eval still scores raw ending HP with no heal ever actually applied). Letting CMA-ES weigh this
    // against everything else (including potion_score_weight) empirically, the same way it found
    // win_hp_weight, rather than a hand-picked all-or-nothing choice -- defaults to 0.0 (matching
    // the reverted, currently-validated-good state) so this rebuild is behavior-identical until
    // explicitly tuned away from it. Only affects boss encounters (nativePostBattleHealedHp returns
    // curHp unchanged for anything else), so this is a true no-op for basic/elite fights regardless
    // of its value.
    // End-state family root references, captured at search entry
    // (nativeRunMctsSearchSeqHalving); consumed by the victory branch below.
    int g_rootMaxHp = 0;
    int g_rootGold = 0;

    double nativeExpectimaxTerminalReward(const BattleContext &bc, int turnAtTerminal) {
        const double base = nativeTerminalReward(bc, turnAtTerminal);
        // Potions still held at the end -- ported from Silver Automaton's own evaluateEndState
        // (potionScore = potionCount * potionWeight, a flat count-based term, not per-potion-type
        // -- their own tuned system doesn't distinguish potion types here either, so this isn't a
        // simplification relative to what's actually proven to work). Half credit on a loss,
        // matching their own potionScore/2 -- a potion not used before dying is worth less than
        // one banked after a win, but still isn't nothing (it reflects the player having options
        // they didn't burn). g_params.potionScoreWeight defaults to 0.0 (off) until validated.
        const double potionScore = bc.potionCount * g_params.potionScoreWeight;
        // Cumulative energy left unspent across the fight's turns -- see
        // g_params.energyWasteWeight's own comment. Charged on BOTH branches.
        const double energyPenalty = g_params.energyWasteWeight * bc.energyWasted;
        if (bc.outcome == Outcome::PLAYER_VICTORY) {
            // effectiveHp == curHp exactly when bossHealCreditWeight == 0.0 (the default) --
            // see nativeExpectimaxTerminalReward's own comment on why this stays a no-op until
            // explicitly tuned away from that.
            const double effectiveHp = bc.player.curHp
                + g_params.bossHealCreditWeight * (nativePostBattleHealedHp(bc) - bc.player.curHp);
            // base already carries nativeTerminalReward's own fixed
            // -NATIVE_TURN_PENALTY_PER_TURN_ON_WIN*turn, so this adds the DELTA needed to reach
            // an effective penalty of winTurnPenaltyWeight * that constant (a no-op at 1.0).
            const double turnPenaltyAdjust =
                (g_params.winTurnPenaltyWeight - 1.0) * NATIVE_TURN_PENALTY_PER_TURN_ON_WIN * turnAtTerminal;
            // base carries NATIVE_W_WIN at weight 1, so add only the delta to reach winBonusWeight.
            const double winBonusAdjust = (g_params.winBonusWeight - 1.0) * NATIVE_W_WIN;
            const double hpFraction = bc.player.maxHp > 0
                ? static_cast<double>(effectiveHp) / bc.player.maxHp : 0.0;
            const double hpSafetyWeight = nativeIsAct1EasyPoolEncounter(bc.encounter)
                ? g_params.earlyActEasyPoolHpSafetyWeight : 0.0;
            double endStateValue = 0.0;
            if (g_params.maxHpGainWeight != 0.0) {
                endStateValue += g_params.maxHpGainWeight
                    * (bc.player.maxHp - g_rootMaxHp);
            }
            if (g_params.goldDeltaWeight != 0.0) {
                endStateValue += g_params.goldDeltaWeight
                    * (bc.player.gold - g_rootGold);
            }
            if (g_params.parasitePenaltyWeight != 0.0
                && !bc.player.hasRelic<RelicId::OMAMORI>()) {
                for (int i = 0; i < bc.monsters.monsterCount; ++i) {
                    const Monster &m = bc.monsters.arr[i];
                    if (m.id == MonsterId::WRITHING_MASS && m.miscInfo) {
                        endStateValue -= g_params.parasitePenaltyWeight;
                        break;
                    }
                }
            }
            return base + (g_params.winHpWeight + hpSafetyWeight) * effectiveHp - bc.player.curHp + potionScore
                 + winBonusAdjust + g_params.winHpFractionWeight * 100.0 * hpFraction
                 - turnPenaltyAdjust - energyPenalty + endStateValue;
        }
        // Monsters still standing -- see g_params.aliveMonsterPenaltyWeight's own comment. Counts
        // halfDead (Awakened One phase 1, Darkling mid-revive) as ALIVE: curHp<=0 there means
        // "about to come back", not dead, matching how nativeMonsterHpRatio above already counts
        // a halfDead monster's FULL maxHp as outstanding threat.
        int aliveCount = 0;
        for (int i = 0; i < bc.monsters.monsterCount; ++i) {
            const Monster &m = bc.monsters.arr[i];
            if (m.id == MonsterId::INVALID) {
                continue;
            }
            if (m.curHp > 0 || m.halfDead) {
                ++aliveCount;
            }
        }
        return base + g_params.lossProgressCreditWeight * (1.0 - nativeMonsterHpRatio(bc)) + potionScore * 0.5
             - g_params.aliveMonsterPenaltyWeight * aliveCount - energyPenalty;
    }

    // Real predicted damage from monster `m`'s CURRENTLY QUEUED move -- NOT
    // Monster::getMoveBaseDamage's raw value, which is only a static
    // per-move-id base-damage TABLE LOOKUP (see MonsterMoveDamage.cpp) with
    // none of the adjustments the engine's ACTUAL damage resolution applies
    // (Monster::calculateDamageToPlayer, Monster.cpp): the monster's current
    // Strength is ADDED, the monster's own Weak status multiplies by 0.75,
    // and the player's Vulnerable status multiplies by 1.5, all before
    // flooring and clamping at 0. Every heuristic/reward signal that used
    // the raw table lookup as if it were the real predicted damage was
    // silently wrong by exactly this gap -- confirmed empirically on Time
    // Eater (stacks +2 Strength via repeated Time Warp procs, applies Weak
    // via Ripple): comparing the raw prediction against actual HP loss one
    // END_TURN later showed mismatches from -12 to +16. See
    // az_search.py's _predicted_incoming_damage (Python reference, kept
    // byte-for-byte equivalent to this) for the full story.
    int nativePredictedIncomingDamage(const BattleContext &sim, const Monster &m, double vulnMult) {
        const auto info = m.getMoveBaseDamage(sim);
        if (info.damage <= 0) {
            return 0;
        }
        const double weakMult = m.weak > 0 ? 0.75 : 1.0;
        const int perHit = std::max(0, static_cast<int>((info.damage + m.strength) * weakMult * vulnMult));
        return perHit * info.attackCount;
    }

    // Per-CARD (not per-type) generic quality prior, indexed by CardId
    // ordinal -- P(picked | offered) at a real reward screen across ~5500
    // decisions from 50 real streamer runs (58% win rate), plus a 0.05
    // smoothing floor, EXACTLY matching lightspeed/cards.py's own
    // PICK_RATE_WEIGHTS (see that file's own module comment for the full
    // data-source story: MaT1g3R/Slay-the-Spire-data, already used
    // in-project for weighted deck generation, not new data pulled in for
    // this). Non-Ironclad cards (never reachable in this project's
    // Ironclad-only search) default to the smoothing floor.
    //
    // WHY: added after this session's own Silverbot comparisons kept
    // finding the SAME pattern regardless of MCTS-parameter tuning -- on
    // every encounter where both bots win 100% of the time, Silverbot
    // still finishes with meaningfully more HP, on both the original fixed
    // deck AND a fresh set of 15 varied decks. The type-level scoring
    // above (attackBase/skillBase/powerScore, ~9 scalars total) cannot
    // express "Shrug It Off is generically stronger than a random Skill"
    // -- every card of a given type scores identically regardless of
    // which specific card it is. This is explicitly a MARGINAL signal
    // (real per-card preference in isolation, not synergy/combat-context-
    // aware -- see cards.py's own caveat), not Silver Automaton's full
    // hand-tuned-per-card cardPlayMap; a real but partial answer to the
    // same gap, not a full port of their approach.
    //
    // DRIFT WARNING: generated once via a one-off script reading
    // lightspeed.cards.PICK_RATE_WEIGHTS, pasted here as a literal --
    // NOT auto-synced. If ironclad_pick_rates.json is ever refreshed,
    // this array goes stale silently until regenerated by hand.
    constexpr double cardPickRateWeight[] = {0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.2328,0.05,0.05,0.25,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.4742,0.05,0.05,0.7091,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.4548,0.5095,0.17,0.05,0.148,0.05,0.05,0.05,0.05,0.2569,0.05,0.05,0.05,0.05,0.4974,0.05,0.05,0.05,0.05,0.2621,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.1136,0.05,0.1538,0.05,0.05,0.05,0.1782,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.8167,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.489,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.2643,0.05,0.05,0.05,0.05,0.7257,0.05,0.05,0.05,0.05,0.05,0.05,0.3625,0.05,0.05,0.1989,0.05,0.325,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.1676,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.4944,0.5778,0.05,0.05,0.05,0.05,0.05,0.8227,0.6214,0.5685,0.05,0.05,0.05,0.05,0.4667,0.05,0.05,0.0913,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.2265,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.1,0.1782,0.05,0.0892,0.05,0.05,0.2621,0.05,0.05,0.4574,0.05,0.85,0.05,0.1,0.05,0.3714,0.05,0.05,0.05,0.1041,0.1735,0.05,0.05,0.05,0.087,0.05,0.05,0.05,0.05,0.05,0.2,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.2231,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.8921,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.1409,0.05,0.05,0.05,0.05,0.4897,0.7079,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.3938,0.05,0.05,0.3389,0.05,0.05,0.1289,0.05,0.6214,0.05,0.05,0.0717,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.1311,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.65,0.05,0.05,0.3577,0.05,0.05,0.3,0.05,0.1963,0.05,0.05,0.9389,0.6125,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.3833,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.1342,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.0616,0.05,0.05,0.05,0.05,0.05,0.2358,0.05,0.1851,0.05,0.3427,0.05,0.05,0.05,0.05,0.05,0.1813,0.05,0.05,0.05,0.05,0.5365,0.05,0.06,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05};

    // Per-card play priority, rank 1 = played most readily. Fitted from this
    // project's own search by lightspeed/_fit_play_priority.py -- the rate at which
    // the search chooses to play a card when it is available -- with the silver
    // prior disabled during collection so the fit could not launder the ranking it
    // replaces back into itself. Rank 0 means no opinion and contributes nothing.
    // Consumed as (134 - rank) / 133, so the width is fixed at 133.
    constexpr std::array<unsigned char, 372> cardPlayRank = {
        0, 0, 0, 49, 0, 0, 0, 0, 0, 0, 0, 106, 21, 43, 72, 116,
        0, 0, 0, 0, 0, 74, 0, 0, 31, 91, 0, 29, 0, 0, 108, 0,
        0, 17, 34, 0, 56, 0, 117, 26, 15, 0, 36, 0, 0, 0, 0, 115,
        0, 0, 0, 25, 86, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0,
        58, 54, 0, 101, 44, 27, 0, 0, 0, 112, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 89, 0, 0, 0, 0, 75, 50, 0,
        53, 0, 0, 0, 47, 0, 0, 0, 113, 0, 0, 24, 0, 0, 0, 0,
        71, 78, 0, 0, 0, 0, 0, 64, 0, 0, 95, 0, 110, 0, 0, 0,
        0, 0, 59, 93, 87, 0, 0, 0, 0, 0, 0, 0, 114, 107, 0, 0,
        0, 0, 0, 28, 61, 2, 0, 0, 111, 0, 20, 63, 0, 69, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 68, 0, 0, 67, 0, 0,
        0, 37, 30, 46, 0, 18, 0, 0, 92, 84, 0, 3, 0, 12, 0, 40,
        0, 94, 0, 0, 0, 105, 45, 0, 42, 0, 79, 0, 0, 0, 0, 0,
        48, 0, 0, 0, 83, 0, 0, 0, 55, 0, 85, 0, 0, 0, 70, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 97, 0, 0, 0, 0, 76,
        98, 0, 9, 0, 14, 0, 0, 77, 0, 6, 11, 0, 0, 0, 0, 0,
        0, 0, 19, 109, 0, 32, 0, 0, 51, 0, 10, 0, 0, 82, 0, 0,
        0, 0, 0, 0, 0, 0, 90, 99, 0, 0, 0, 0, 0, 0, 0, 0,
        33, 41, 0, 81, 57, 0, 0, 100, 0, 38, 0, 5, 7, 13, 0, 0,
        0, 0, 0, 118, 0, 0, 0, 39, 0, 0, 0, 0, 52, 0, 0, 0,
        0, 65, 0, 0, 0, 0, 0, 88, 0, 16, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 102, 0, 0, 0, 0, 66, 104, 73, 22, 0, 4, 0, 0,
        62, 0, 0, 103, 0, 0, 0, 0, 60, 0, 35, 0, 0, 96, 23, 0,
        0, 80, 0, 0,
    };




    // Cards that hit ALL enemies -- character-agnostic (an AOE card is AOE
    // regardless of which character plays it via Prismatic Shard / foreign
    // influence / Discovery, etc.). Expanded from the original Ironclad-only
    // set to cover all four classes so the +aoeBonus term applies correctly
    // in multi-monster fights for any character.
    bool isAoeCard(CardId id) {
        return id == CardId::CLEAVE || id == CardId::IMMOLATE
            || id == CardId::THUNDERCLAP || id == CardId::WHIRLWIND  // Ironclad
            || id == CardId::DAGGER_SPRAY || id == CardId::ALL_OUT_ATTACK
            || id == CardId::DIE_DIE_DIE || id == CardId::CORPSE_EXPLOSION
            || id == CardId::CRIPPLING_CLOUD                         // Silent
            || id == CardId::DOOM_AND_GLOOM || id == CardId::HYPERBEAM
            || id == CardId::ELECTRODYNAMICS || id == CardId::THUNDER_STRIKE  // Defect
            || id == CardId::CONSECRATE || id == CardId::CONCLUDE;   // Watcher
    }

    // Cards that APPLY Vulnerable -- character-agnostic (the sequencing value
    // of applying a 1.5x damage multiplier before spending attacks is the same
    // regardless of who played it), covering all classes so the bonus works in
    // Prismatic-Shard/Discovery cross-class scenarios. Hand-authored for now;
    // the durable path is test-simulating each card once at startup.
    bool isVulnerableApplier(CardId id) {
        return id == CardId::BASH || id == CardId::THUNDERCLAP
            || id == CardId::UPPERCUT || id == CardId::SHOCKWAVE   // Ironclad
            || id == CardId::TERROR                                 // Silent
            || id == CardId::BEAM_CELL                              // Defect
            || id == CardId::CRUSH_JOINTS || id == CardId::INDIGNATION  // Watcher
            || id == CardId::TRIP;                                  // colorless
    }

    // Cards that APPLY Weak. Same sequencing argument as isVulnerableApplier:
    // Weak cuts incoming attack damage by 25%, so applying it BEFORE the enemy
    // swings is worth strictly more than after, and neither skillBase nor
    // attackBase can express that -- they treat every card of a type alike.
    // Derived from card text rather than hand-recalled, so it covers all four
    // classes plus colorless.
    bool isWeakApplier(CardId id) {
        return id == CardId::CLOTHESLINE || id == CardId::INTIMIDATE
            || id == CardId::UPPERCUT || id == CardId::SHOCKWAVE   // Ironclad
            || id == CardId::NEUTRALIZE || id == CardId::SUCKER_PUNCH
            || id == CardId::LEG_SWEEP || id == CardId::MALAISE
            || id == CardId::CRIPPLING_CLOUD                        // Silent
            || id == CardId::GO_FOR_THE_EYES                        // Defect
            || id == CardId::SASH_WHIP || id == CardId::WAVE_OF_THE_HAND  // Watcher
            || id == CardId::BLIND;                                 // colorless
    }

    // Silent cards that directly apply poison to a target. Catalyst, Noxious
    // Fumes, and Envenom are deliberate inclusions: they interact with poison
    // even though they don't apply it directly (Catalyst multiplies, Noxious
    // Fumes is AoE-per-turn, Envenom adds poison on attack damage).
    bool isSilentPoisonApplier(CardId id) {
        return id == CardId::DEADLY_POISON || id == CardId::POISONED_STAB
            || id == CardId::BOUNCING_FLASK || id == CardId::NOXIOUS_FUMES
            || id == CardId::CATALYST || id == CardId::ENVENOM
            || id == CardId::CRIPPLING_CLOUD || id == CardId::CORPSE_EXPLOSION;
    }

    // Cards that DRAW. Playing these before the rest of the turn strictly
    // dominates playing them after, whenever energy remains to use what they
    // find: the same cards get played either way, but drawing first widens
    // the choice set for every later play this turn. Standalone card value
    // cannot express an ordering preference between two cards both of which
    // will be played, which is exactly what the decision miner measured us
    // getting wrong (Pommel before Carnage, 60 states; Offering first, 62).
    bool isDrawCard(CardId id) {
        return id == CardId::POMMEL_STRIKE || id == CardId::OFFERING
            || id == CardId::BATTLE_TRANCE || id == CardId::SHRUG_IT_OFF
            || id == CardId::BURNING_PACT || id == CardId::TRUE_GRIT
            || id == CardId::WARCRY || id == CardId::DUAL_WIELD
            || id == CardId::ACROBATICS || id == CardId::BACKFLIP
            || id == CardId::PREPARED || id == CardId::PANIC_BUTTON   // Ironclad/Silent/colorless
            || id == CardId::ESCAPE_PLAN || id == CardId::CLOAK_AND_DAGGER
            || id == CardId::SKIM || id == CardId::COOLHEADED
            || id == CardId::COMPILE_DRIVER || id == CardId::COLD_SNAP   // Defect
            || id == CardId::COLLECT || id == CardId::FLASH_OF_STEEL;
    }

    // One-turn strength strippers. Their entire value is timing: played on the
    // turn a big multi-hit attack is telegraphed they erase most of it, played
    // any other turn they do nothing. Scored flat by skillBase today, so the
    // rollout spends them early and has nothing for the burst.
    bool isBurstDebuffCard(CardId id) {
        return id == CardId::DARK_SHACKLES || id == CardId::PIERCING_WAIL
            || id == CardId::DISARM;
    }

    // Cards whose primary purpose is generating immediate block -- suppressed
    // by defensiveCardSuppressionPenalty when block is already sufficient, so
    // the rollout prefers attacking instead of over-blocking. Character-agnostic:
    // covers all four classes plus colorless block sources (Dark Shackles,
    // Panic Button).
    bool isDefensiveCard(CardId id) {
        return id == CardId::POWER_THROUGH || id == CardId::TRUE_GRIT || id == CardId::IMPERVIOUS
            || id == CardId::SHRUG_IT_OFF || id == CardId::FLAME_BARRIER || id == CardId::ENTRENCH
            || id == CardId::DEFEND_RED || id == CardId::SENTINEL || id == CardId::SECOND_WIND
            || id == CardId::GHOSTLY_ARMOR || id == CardId::RAGE                         // Ironclad
            || id == CardId::DEFEND_GREEN || id == CardId::SURVIVOR || id == CardId::DEFLECT
            || id == CardId::DODGE_AND_ROLL || id == CardId::LEG_SWEEP || id == CardId::BLUR
            || id == CardId::BACKFLIP || id == CardId::PIERCING_WAIL                     // Silent
            || id == CardId::DEFEND_BLUE || id == CardId::LEAP || id == CardId::COOLHEADED
            || id == CardId::CHARGE_BATTERY || id == CardId::REINFORCED_BODY
            || id == CardId::STEAM_BARRIER || id == CardId::AUTO_SHIELDS
            || id == CardId::GLACIER || id == CardId::EQUILIBRIUM || id == CardId::FORCE_FIELD
            || id == CardId::GENETIC_ALGORITHM || id == CardId::BUFFER                    // Defect
            || id == CardId::DEFEND_PURPLE || id == CardId::PROTECT || id == CardId::HALT
            || id == CardId::EMPTY_BODY || id == CardId::DECEIVE_REALITY
            || id == CardId::PERSEVERANCE || id == CardId::SPIRIT_SHIELD                  // Watcher
            || id == CardId::DARK_SHACKLES || id == CardId::PANIC_BUTTON;                 // colorless
    }

    // Direct, one-shot base block available immediately from playing the card.
    // Delayed/conditional cards deliberately return zero: treating their
    // conditional payoff as this turn's certain block would be misleading.
    // Covers all four classes plus colorless sources.
    int nativeImmediateBlockBase(const CardInstance &card) {
        const bool up = card.upgraded;
        switch (card.id) {
            case CardId::DEFEND_RED: return up ? 8 : 5;
            case CardId::TRUE_GRIT: return up ? 9 : 7;
            case CardId::IMPERVIOUS: return up ? 40 : 30;
            case CardId::SHRUG_IT_OFF: return up ? 11 : 8;
            case CardId::FLAME_BARRIER: return up ? 16 : 12;
            case CardId::POWER_THROUGH: return up ? 20 : 15;
            case CardId::SENTINEL: return up ? 8 : 5;
            case CardId::GHOSTLY_ARMOR: return up ? 13 : 10;
            case CardId::IRON_WAVE: return up ? 7 : 5;
            case CardId::ARMAMENTS: return 5;
            case CardId::DEFEND_GREEN: return up ? 8 : 5;
            case CardId::SURVIVOR: return up ? 11 : 8;
            case CardId::DEFLECT: return up ? 7 : 4;
            case CardId::DODGE_AND_ROLL: return up ? 6 : 4;
            case CardId::LEG_SWEEP: return up ? 14 : 11;
            case CardId::BLUR: return up ? 8 : 5;
            case CardId::BACKFLIP: return up ? 8 : 5;
            case CardId::CLOAK_AND_DAGGER: return up ? 6 : 6;
            case CardId::ESCAPE_PLAN: return up ? 5 : 3;
            case CardId::FINESSE: return up ? 4 : 2;
            case CardId::DASH: return up ? 13 : 10;
            case CardId::DEFEND_BLUE: return up ? 8 : 5;
            case CardId::LEAP: return up ? 12 : 9;
            case CardId::CHARGE_BATTERY: return up ? 10 : 7;
            case CardId::STEAM_BARRIER: return up ? 8 : 6;
            case CardId::AUTO_SHIELDS: return up ? 15 : 11;
            case CardId::REINFORCED_BODY: return 0; // X cost
            case CardId::GLACIER: return up ? 10 : 7;
            case CardId::EQUILIBRIUM: return up ? 16 : 13;
            case CardId::BOOT_SEQUENCE: return up ? 13 : 10;
            case CardId::FORCE_FIELD: return up ? 16 : 12;
            case CardId::HOLOGRAM: return up ? 5 : 3;
            case CardId::DEFEND_PURPLE: return up ? 8 : 5;
            case CardId::PROTECT: return up ? 16 : 12;
            // BattleContext.cpp adds up ? 14 : 9 on top of this while in Wrath, but
            // stance is not visible from a CardInstance alone, so we return only the
            // unconditionally guaranteed block and deliberately under-count Halt
            // rather than assume the player is in Wrath.
            case CardId::HALT: return up ? 4 : 3;
            case CardId::EMPTY_BODY: return up ? 10 : 7;
            case CardId::DECEIVE_REALITY: return up ? 7 : 4;
            case CardId::VIGILANCE: return up ? 12 : 8;
            case CardId::EVALUATE: return up ? 10 : 6;
            case CardId::PROSTRATE: return 4;
            case CardId::THIRD_EYE: return up ? 9 : 7;
            case CardId::SWIVEL: return up ? 11 : 8;
            case CardId::SANCTITY: return up ? 9 : 6;
            case CardId::JUST_LUCKY: return up ? 3 : 2;
            // Base value only: BattleContext.cpp adds specialData, which grows every
            // time Perseverance is retained. That retain growth is not modelled here.
            case CardId::PERSEVERANCE: return up ? 7 : 5;
            case CardId::SPIRIT_SHIELD: return 0; // variable
            case CardId::GOOD_INSTINCTS: return up ? 9 : 6;
            case CardId::PANIC_BUTTON: return up ? 40 : 30;
            default: return 0;
        }
    }


    // One-shot, immediate value of a Power -- paid the moment it resolves and so
    // NOT scaled by remaining fight length, unlike nativePowerPerTurnValue.
    // Inflame is the whole reason this exists: its Strength arrives now, which
    // makes it worth playing late in a fight where Demon Form is not.
    int nativePowerImmediateValue(const CardInstance &card) {
        switch (card.id) {
            case CardId::INFLAME: return card.upgraded ? 3 : 2;  // Strength, immediately
            default:              return 0;
        }
    }

    int nativeImmediateSelfDamage(const CardInstance &card) {
        switch (card.id) {
            case CardId::OFFERING: return 6;
            case CardId::BLOODLETTING: return 3;
            case CardId::HEMOKINESIS: return 2;
            case CardId::JAX: return 3;
            default: return 0;
        }
    }

    // Shared per-node context both nativeHeuristicPick (rollout policy) and
    // nativeHeuristicVisitOrder (tree visit-priority, see its own comment)
    // need -- factored out so there's one place computing it, not two
    // copies that can drift.
    struct HeuristicContext {
        double unblocked;
        bool timeWarpRisk;
        bool hasteWastedDebuffs;
        int livingMonsters;
        // True when current block already covers predicted incoming damage (within
        // g_params.blockSufficiencyMargin) -- ports Silver Automaton's own "we have enough
        // block" gate (SimpleAgent.cpp: block > incoming - act - STS_BLOCK_OFFSET), found while
        // investigating why our rollouts finish fights with meaningfully less HP than theirs at
        // matched simulation counts (see silverbot-reference/comparison_tests's own numbers).
        // Their version subtracts the current act number as part of the margin; we use a flat
        // tunable margin instead since not every BattleContext this engine builds has a reliable
        // act available (e.g. nativeBuildBattleContext's live-bridge reconstructions).
        bool blockSufficient;
        // Fraction of total enemy max HP still standing, used by the POWER branch as a proxy for
        // how many turns a scaling Power has left to pay off. Computed here rather than per-action
        // for the same reason nativeVulnerableApplyBonus was hoisted: nativeScoreAction runs once
        // per legal action per rollout step, and recomputing a whole-monster-array sum in there
        // was measured as the dominant cost last time it happened.
        double monsterHpRatio;
        // See g_params.powerHorizonWeight. Precomputed here for the same reason
        // as monsterHpRatio -- nativeScoreAction runs once per legal action per
        // rollout step, so anything needing a monster-array sum belongs here.
        double powerHorizon;
        // Deck-composition counts over the whole combat deck (draw + hand + discard).
        // These exist so the conditional Powers -- Feel No Pain, Dark Embrace, Evolve,
        // Fire Breathing, Rupture, Corruption, Juggernaut, Barricade -- can be scored on
        // whether THIS deck actually enables them rather than on a blind constant. That
        // distinction is the whole reason nativePowerPerTurnValue used to return 0 for
        // all eight. Hoisted here for the same cost reason as monsterHpRatio.
        int exhaustCards;
        int statusCurseCards;
        int skillCards;
        int blockCards;
    };

    HeuristicContext nativeComputeHeuristicContext(const BattleContext &sim) {
        const double vulnMult = sim.player.hasStatus<PS::VULNERABLE>() ? 1.5 : 1.0;
        double incoming = 0.0;
        int livingMonsters = 0;
        for (int i = 0; i < sim.monsters.monsterCount; ++i) {
            if (sim.monsters.arr[i].curHp > 0) {
                incoming += nativePredictedIncomingDamage(sim, sim.monsters.arr[i], vulnMult);
                ++livingMonsters;
            }
        }
        const double unblocked = std::max(0.0, incoming - sim.player.block);

        bool timeWarpRisk = false;
        bool hasteWastedDebuffs = false;
        for (int i = 0; i < sim.monsters.monsterCount; ++i) {
            const Monster &m = sim.monsters.arr[i];
            if (m.curHp <= 0) {
                continue;
            }
            if (m.getStatus<MS::TIME_WARP>() >= 11) {
                timeWarpRisk = true;
            }
            if (m.id == MonsterId::TIME_EATER && m.miscInfo == 0 && m.curHp <= m.maxHp * 0.5) {
                hasteWastedDebuffs = true;
            }
        }
        const bool blockSufficient = static_cast<double>(sim.player.block) > incoming - g_params.blockSufficiencyMargin;

        // Absolute enemy HP still standing, saturating at powerHorizonHp -- the
        // long-fight proxy that monsterHpRatio cannot express. See
        // g_params.powerHorizonWeight.
        int monsterHpRemaining = 0;
        for (int i = 0; i < sim.monsters.monsterCount; ++i) {
            const Monster &m = sim.monsters.arr[i];
            if (m.id != MonsterId::INVALID && (m.curHp > 0 || m.halfDead)) {
                monsterHpRemaining += std::max(0, static_cast<int>(m.curHp));
            }
        }
        const double powerHorizon = std::min(
            1.0, static_cast<double>(monsterHpRemaining)
                 / std::max(1.0, g_params.powerHorizonHp));

        // One pass over the combat deck for the conditional-Power gates. Exhausted
        // cards are excluded on purpose: a Feel No Pain played after the deck's
        // exhaust fodder is already gone has nothing left to trigger on.
        int exhaustCards = 0, statusCurseCards = 0, skillCards = 0, blockCards = 0;
        const auto countPile = [&](const auto &pile, int size) {
            for (int i = 0; i < size; ++i) {
                const CardInstance &c = pile[i];
                const CardType t = cardTypes[static_cast<int>(c.id)];
                if (doesCardExhaust(c.id, c.upgraded)) {
                    ++exhaustCards;
                }
                if (t == CardType::STATUS || t == CardType::CURSE) {
                    ++statusCurseCards;
                }
                if (t == CardType::SKILL) {
                    ++skillCards;
                }
                if (nativeImmediateBlockBase(c) > 0) {
                    ++blockCards;
                }
            }
        };
        countPile(sim.cards.drawPile, sim.cards.drawPile.size());
        countPile(sim.cards.hand, sim.cards.cardsInHand);
        countPile(sim.cards.discardPile, sim.cards.discardPile.size());

        return {unblocked, timeWarpRisk, hasteWastedDebuffs, livingMonsters, blockSufficient,
                nativeMonsterHpRatio(sim), powerHorizon,
                exhaustCards, statusCurseCards, skillCards, blockCards};
    }

    // Recurring per-turn value of a Power, in rough damage-equivalent points.
    // The POWER branch of nativeScoreAction was a flat constant, so Demon Form,
    // Metallicize and Evolve all scored identically and only the learned
    // per-card prior could tell them apart. A Power's worth is (value per turn)
    // x (turns left), so this supplies the first factor and the caller scales
    // it by a remaining-fight proxy.
    //
    // Strength and Block are both counted at face value per turn: 2 Strength is
    // ~2 extra damage per attack played, 3 Block is ~3 damage mitigated. That
    // is an approximation -- Strength scales with attacks played per turn and
    // Metallicize's Block does not stack with Barricade absent -- but the
    // weight is tunable, so CMA-ES sets the exchange rate. What matters here is
    // the RELATIVE ordering, which face value gets right.
    //
    // The eight conditional Powers below used to return 0, on the reasoning that they
    // "pay off only in decks built for them, and guessing a per-turn number for them
    // would be worse than the status quo". That objection is about the CONDITION, not
    // the magnitude -- each card states its own payoff, and what was unknown was
    // whether this particular deck ever triggers it. So the condition is now measured
    // from the deck (HeuristicContext's counts) instead of guessed, and the magnitude
    // is the card's own text expressed per TURN rather than per trigger, which is the
    // unit the existing entries already use (Demon Form is 2 Strength/turn, not 2 per
    // proc). g_params.powerPerTurnValueWeight still scales the whole table, so CMA-ES
    // sets the exchange rate and only the relative ordering is asserted here.
    //
    // This matters because 8 of Ironclad's 14 Powers were in that zeroed set, including
    // Barricade and Corruption -- the two cards Ironclad decks are most often built
    // around -- leaving them priced identically to any other Power.
    int nativePowerPerTurnValue(const CardInstance &card, const HeuristicContext &ctx) {
        const bool up = card.upgraded;
        // Enough enablers that the Power fires roughly every turn rather than once.
        const bool exhaustDeck = ctx.exhaustCards >= 3;
        const bool statusDeck = ctx.statusCurseCards >= 3;
        const bool skillDeck = ctx.skillCards >= 6;
        const bool blockDeck = ctx.blockCards >= 4;
        switch (card.id) {
            case CardId::DEMON_FORM:   return up ? 3 : 2;   // Strength/turn
            case CardId::METALLICIZE:  return up ? 4 : 3;   // Block/turn
            case CardId::COMBUST:      return up ? 7 : 5;   // AoE damage/turn, costs 1 HP
            case CardId::BERSERK:      return 3;            // 1 energy/turn, at the cost of Vulnerable
            case CardId::BRUTALITY:    return 1;            // 1 card/turn, costs 1 HP
            case CardId::MAGNETISM:    return 1;            // 1 colorless card/turn

            // Block retention. Worth roughly a second helping of whatever block the
            // deck already lays down each turn, so it needs a deck that lays some.
            case CardId::BARRICADE:    return blockDeck ? 4 : 0;
            // Skills cost 0 and exhaust: about an energy a turn once skills are dense.
            case CardId::CORRUPTION:   return skillDeck ? 3 : 0;
            // 3/4 Block per exhaust, ~1 exhaust a turn in a deck built for it.
            case CardId::FEEL_NO_PAIN: return exhaustDeck ? (up ? 4 : 3) : 0;
            // 1 card per exhaust.
            case CardId::DARK_EMBRACE: return exhaustDeck ? 1 : 0;
            // 5/7 damage per Block gain, and blockDeck is what makes that recur.
            case CardId::JUGGERNAUT:   return blockDeck ? (up ? 7 : 5) : 0;
            // 1/2 cards per Status drawn; Statuses arrive well under once a turn, so
            // the per-trigger number is discounted rather than taken at face value.
            case CardId::EVOLVE:       return statusDeck ? (up ? 2 : 1) : 0;
            // 6/10 AoE per Status/Curse drawn, discounted the same way as Evolve.
            case CardId::FIRE_BREATHING: return statusDeck ? (up ? 3 : 2) : 0;
            // 1/2 Strength per HP-loss card. Ungated: the cards that trigger it are
            // Powers and attacks this pass does not separately count, and 1 is small
            // enough that being wrong costs little.
            case CardId::RUPTURE:      return up ? 2 : 1;
            default:                   return 0;
        }
    }

    // Per-CARD-action score (ATTACK/SKILL/POWER/other), shared by both
    // nativeHeuristicPick (which skips END_TURN, handling it via the
    // separate timeWarpRisk shortcut below) and nativeHeuristicVisitOrder
    // (which scores END_TURN too, since the tree needs a real ranking for
    // it, not just a rollout's single pick).
    // Fixed-width per-action feature vector for the learned rollout-scoring net (see
    // nativePolicyNetScore's own comment, defined further down near ValueNet -- forward-declared
    // here since it needs nativeLeafFeatures/ValueNetLayer, both defined later in this file).
    // Order: [is_attack, is_skill, is_power, is_other, target_hp_missing_fraction,
    // target_block_fraction_capped, is_aoe_into_multiple_monsters, card_pick_rate_weight].
    // Deliberately NOT duplicating state-level context (danger fraction, current HP, etc.) that
    // nativeLeafFeatures already provides -- the net sees both vectors concatenated, so it learns
    // the state/action interaction itself instead of having it hand-engineered twice.
    static constexpr int NATIVE_ACTION_FEATURE_DIM = 8;
    std::array<double, NATIVE_ACTION_FEATURE_DIM> nativeActionFeatures(
            const BattleContext &sim, const search::Action &a, const HeuristicContext &ctx) {
        if (a.getActionType() != search::ActionType::CARD) {
            return {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0};
        }
        const CardInstance &card = sim.cards.hand[a.getSourceIdx()];
        const CardType ctype = cardTypes[static_cast<int>(card.id)];
        double targetHpMissingFraction = 0.0, targetBlockFraction = 0.0, isAoeMulti = 0.0;
        if (ctype == CardType::ATTACK) {
            const int targetIdx = a.getTargetIdx();
            if (targetIdx >= 0 && targetIdx < sim.monsters.monsterCount
                && sim.monsters.arr[targetIdx].curHp > 0) {
                const Monster &target = sim.monsters.arr[targetIdx];
                targetHpMissingFraction = 1.0 - static_cast<double>(target.curHp) / std::max(1, static_cast<int>(target.maxHp));
                targetBlockFraction = std::min(static_cast<int>(target.block), 20) / 20.0;
            }
            if (ctx.livingMonsters >= 2 && isAoeCard(card.id)) {
                isAoeMulti = 1.0;
            }
        }
        return {
            ctype == CardType::ATTACK ? 1.0 : 0.0,
            ctype == CardType::SKILL ? 1.0 : 0.0,
            ctype == CardType::POWER ? 1.0 : 0.0,
            0.0,
            targetHpMissingFraction, targetBlockFraction, isAoeMulti,
            cardPickRateWeight[static_cast<int>(card.id)],
        };
    }

    // Forward declaration -- defined near ValueNet/nativeValueNetEstimate below, since it shares
    // nativeLeafFeatures and the ValueNetLayer forward-pass shape with that code. Internally
    // returns 0.0 whenever no policy net has been loaded (see load_policy_net), so callers never
    // need to guard on g_policyNet.loaded themselves -- g_params.policyNetWeight times a
    // guaranteed 0.0 is always a true no-op regardless of that weight's value. Takes the state's
    // nativeLeafFeatures pre-computed (stateFeatures) rather than computing them itself -- see
    // nativeScoreAction's own comment for why (it's a per-STATE quantity, identical across every
    // action scored within the same decision, so computing it once per decision instead of once
    // per action-scored is a real, measured speedup with rollout mode's heaviest hot-path caller).
    double nativePolicyNetScore(const BattleContext &sim, const search::Action &a, const HeuristicContext &ctx,
                                 const std::array<double, NATIVE_LEAF_FEATURE_DIM> &stateFeatures);
    // Forward declaration -- see this function's own (later) definition for the full comment.
    // Needed here since nativeHeuristicPick/nativeHeuristicScores below now compute this ONCE
    // per decision and thread it through, rather than each nativeScoreAction call computing its
    // own copy.
    std::array<double, NATIVE_LEAF_FEATURE_DIM> nativeLeafFeatures(const BattleContext &bc);

    // stateFeatures: nativeLeafFeatures(sim), precomputed ONCE by the caller and threaded through
    // to nativePolicyNetScore -- a per-STATE quantity that used to get recomputed from scratch on
    // every single call to this function (i.e. once per LEGAL ACTION scored in a decision, not
    // once per decision), which mattered a lot with a loaded policy net: nativeHeuristicPick is
    // the rollout's per-step action-pick, called up to NATIVE_ROLLOUT_MAX_ACTIONS times per
    // rollout, each call scoring every legal action -- measured ~7x slower per search with a net
    // loaded before this fix, most of which was this exact redundant recomputation.
    // Non-zero only when this card applies Vulnerable AND some living target does not already have
    // it -- re-applying to an already-Vulnerable target buys nothing, so it must not score a bonus.
    // Targeted cards check their own target; untargeted ones (Thunderclap/Shockwave hit everything)
    // check whether ANY living monster still lacks it.
    double nativeVulnerableApplyBonus(const BattleContext &sim, const search::Action &a,
                                       const CardInstance &card) {
        if (g_params.vulnerableApplyBonus == 0.0 || !isVulnerableApplier(card.id)) {
            return 0.0;
        }
        if (card.requiresTarget()) {
            const int t = a.getTargetIdx();
            if (t < 0 || t >= sim.monsters.monsterCount) {
                return 0.0;
            }
            const Monster &m = sim.monsters.arr[t];
            if (g_params.artifactAwareDebuffs != 0.0
                && m.getStatus<MS::ARTIFACT>() > 0) {
                return 0.0;
            }
            return (m.curHp > 0 && m.getStatus<MS::VULNERABLE>() <= 0) ? g_params.vulnerableApplyBonus : 0.0;
        }
        for (int i = 0; i < sim.monsters.monsterCount; ++i) {
            const Monster &m = sim.monsters.arr[i];
            if (g_params.artifactAwareDebuffs != 0.0
                && m.getStatus<MS::ARTIFACT>() > 0) {
                continue;
            }
            if (m.curHp > 0 && m.getStatus<MS::VULNERABLE>() <= 0) {
                return g_params.vulnerableApplyBonus;
            }
        }
        return 0.0;
    }

    // Weak counterpart of nativeVulnerableApplyBonus -- same shape, same
    // already-has-it gate (re-applying to an already-Weak target buys nothing),
    // same targeted/untargeted split via requiresTarget(). Separate parameter
    // rather than sharing vulnerableApplyBonus because the two are not worth
    // the same: Vulnerable multiplies OUR damage output, Weak reduces THEIR
    // damage, so their value scales with different things and CMA-ES needs to
    // move them independently.
    double nativeWeakApplyBonus(const BattleContext &sim, const search::Action &a,
                                 const CardInstance &card) {
        if (g_params.weakApplyBonus == 0.0 || !isWeakApplier(card.id)) {
            return 0.0;
        }
        if (card.requiresTarget()) {
            const int t = a.getTargetIdx();
            if (t < 0 || t >= sim.monsters.monsterCount) {
                return 0.0;
            }
            const Monster &m = sim.monsters.arr[t];
            if (g_params.artifactAwareDebuffs != 0.0
                && m.getStatus<MS::ARTIFACT>() > 0) {
                return 0.0;
            }
            return (m.curHp > 0 && m.getStatus<MS::WEAK>() <= 0) ? g_params.weakApplyBonus : 0.0;
        }
        for (int i = 0; i < sim.monsters.monsterCount; ++i) {
            const Monster &m = sim.monsters.arr[i];
            if (g_params.artifactAwareDebuffs != 0.0
                && m.getStatus<MS::ARTIFACT>() > 0) {
                continue;
            }
            if (m.curHp > 0 && m.getStatus<MS::WEAK>() <= 0) {
                return g_params.weakApplyBonus;
            }
        }
        return 0.0;
    }

    // --- MAST: Move-Average Sampling Technique --------------------------------
    // An online average-return table keyed on the MOVE -- here (CardId, upgraded), plus
    // one slot each for END_TURN, potions, and everything else -- accumulated during a
    // single search and used to bias the rollout's action choice.
    //
    // This exists because the offline route was measured and rejected. Distilling the
    // search into a rollout net lost -2.12 HP at matched wall clock, and
    // lightspeed/_probe_card_identity.py traced that to nativeActionFeatures never
    // identifying WHICH card is being played: widening it with a learned embedding is
    // worth +2.3pp top-1 against a net costing 4.97x search speed, which cannot pay. The
    // same probe found a per-card SCALAR captured most of that signal at zero inference
    // cost. MAST is that scalar, learned online against the state distribution of the
    // fight actually being searched, for the price of a table lookup.
    //
    // Scale: rollout returns are NATIVE_W_SHAPE * expectimax rewards (hundreds) while
    // nativeScoreAction's own scores span roughly 4-30, so the table is consumed as a
    // z-score against the search's own return distribution. mastWeight is therefore in
    // units of heuristic-score points per standard deviation, and does not need
    // re-tuning when the reward weights move.
    constexpr int MAST_END_TURN = 0;
    constexpr int MAST_OTHER = 1;
    constexpr int MAST_CARD_BASE = 2;
    constexpr int MAST_POTION_BASE = MAST_CARD_BASE + 372 * 2;
    constexpr int MAST_TABLE_SIZE = MAST_POTION_BASE + 64;
    std::array<double, MAST_TABLE_SIZE> g_mastW {};
    std::array<std::int32_t, MAST_TABLE_SIZE> g_mastN {};
    double g_mastTotalW = 0.0;
    double g_mastTotalW2 = 0.0;
    std::int64_t g_mastTotalN = 0;
    // Every move played in the CURRENT simulation, in-tree and rollout alike. Global and
    // thread_local rather than threaded through nativeSimulate's signature, matching
    // g_gumbelRng's existing treatment: a search is single-threaded per process, and
    // g_params is already unlocked process-global state under the same rule.
    thread_local std::vector<int> g_mastTrace;

    bool nativeMastActive() {
        return g_params.mastWeight != 0.0;
    }

    void nativeMastReset() {
        g_mastW.fill(0.0);
        g_mastN.fill(0);
        g_mastTotalW = 0.0;
        g_mastTotalW2 = 0.0;
        g_mastTotalN = 0;
        g_mastTrace.clear();
    }

    int nativeMastKey(const BattleContext &bc, const search::Action &a) {
        switch (a.getActionType()) {
            case search::ActionType::CARD: {
                const CardInstance &card = bc.cards.hand[a.getSourceIdx()];
                return MAST_CARD_BASE + static_cast<int>(card.id) * 2 + (card.upgraded ? 1 : 0);
            }
            case search::ActionType::POTION:
                return MAST_POTION_BASE + static_cast<int>(bc.potions[a.getSourceIdx()]);
            case search::ActionType::END_TURN:
                return MAST_END_TURN;
            default:
                // Card-select actions index a task-dependent pile, not the hand, so there
                // is no move identity to key on without resolving the task first.
                return MAST_OTHER;
        }
    }

    void nativeMastRecord(const BattleContext &bc, const search::Action &a) {
        g_mastTrace.push_back(nativeMastKey(bc, a));
    }

    void nativeMastUpdate(double value) {
        for (const int key : g_mastTrace) {
            g_mastW[key] += value;
            g_mastN[key] += 1;
        }
        // The baseline is the same population the per-move averages are drawn from, so a
        // move with no evidence scores exactly 0 and cannot outrank a measured one by
        // accident.
        const auto n = static_cast<double>(g_mastTrace.size());
        g_mastTotalW += value * n;
        g_mastTotalW2 += value * value * n;
        g_mastTotalN += static_cast<std::int64_t>(g_mastTrace.size());
    }

    double nativeMastScore(int key) {
        const std::int32_t n = g_mastN[key];
        if (n < g_params.mastMinVisits || g_mastTotalN < 2) {
            return 0.0;
        }
        const auto total = static_cast<double>(g_mastTotalN);
        const double mean = g_mastTotalW / total;
        const double variance = g_mastTotalW2 / total - mean * mean;
        if (!(variance > 1e-12)) {
            return 0.0;
        }
        return ((g_mastW[key] / n) - mean) / std::sqrt(variance);
    }

    double nativeScoreAction(const BattleContext &sim, const search::Action &a, const HeuristicContext &ctx,
                              const std::array<double, NATIVE_LEAF_FEATURE_DIM> &stateFeatures) {
        double score;
        if (a.getActionType() == search::ActionType::POTION) {
            // See TunableParams' potion block for why this branch exists and why it is a no-op at
            // the defaults. A potion action is either a drink or a discard; Action::execute
            // routes on targetIdx > 5, so that is what discriminates them here too.
            if (a.getTargetIdx() > 5) {
                score = g_params.rolloutPotionBase - g_params.rolloutPotionDiscardPenalty;
            } else {
                score = g_params.rolloutPotionBase
                    + (ctx.unblocked / std::max(1, sim.player.curHp)) * g_params.rolloutPotionDangerScale;
                const Potion p = sim.potions[a.getSourceIdx()];
                // Gated on potionRequiresTarget: an untargeted potion's targetIdx is a
                // placeholder 0, so reading monsters.arr[0] for it would score the potion
                // against a monster it has nothing to do with.
                if (g_params.rolloutPotionFinishOffScale != 0.0 && potionRequiresTarget(p)) {
                    const int targetIdx = a.getTargetIdx();
                    if (targetIdx >= 0 && targetIdx < sim.monsters.monsterCount
                        && sim.monsters.arr[targetIdx].curHp > 0) {
                        const Monster &target = sim.monsters.arr[targetIdx];
                        score += (1.0 - static_cast<double>(target.curHp)
                                    / std::max(1, static_cast<int>(target.maxHp)))
                            * g_params.rolloutPotionFinishOffScale;
                    }
                }
            }
        } else if (a.getActionType() != search::ActionType::CARD) {
            score = g_params.rolloutNonCardBase;  // END_TURN via nativeHeuristicVisitOrder, and card-select options
        } else {
            const CardInstance &card = sim.cards.hand[a.getSourceIdx()];
            const CardType ctype = cardTypes[static_cast<int>(card.id)];
            // Added uniformly to every CARD branch below (including the
            // STATUS/CURSE fallback) -- see cardPickRateWeight's own comment.
            // 0.0 at the default perCardWeightScale, so this is a no-op until
            // explicitly tuned.
            // A per-card play-priority prior used to sit here, sourced from Silver
            // Automaton's hand-curated 133-card ordering. Removed 2026-08-01: it was
            // their data rather than ours, and carrying it obliged us to carry their
            // copyright notice. Measured cost of removal: -1.20 +/- 0.49 HP (t = -2.45)
            // on 500 paired train fights. The slot is worth refilling from our own
            // search -- see lightspeed/_fit_play_priority.py, which fits a conditional
            // logit over the cards available at each decision.
            const int playRank = cardPlayRank[static_cast<int>(card.id)];
            const double playPrior = playRank == 0 ? 0.0
                : static_cast<double>(134 - playRank) / 133.0;
            const double playPriorWeight = isBossEncounter(sim.encounter)
                && g_params.bossCardPlayPriorWeight >= 0.0
                ? g_params.bossCardPlayPriorWeight
                : g_params.cardPlayPriorWeight;
            double perCardBonus = g_params.perCardWeightScale * cardPickRateWeight[static_cast<int>(card.id)]
                + playPriorWeight * playPrior;
            if (g_hasEarlyActCardBias && sim.player.maxHp <= EARLY_ACT_BIAS_MAX_HP) {
                perCardBonus += g_earlyActCardBias[static_cast<int>(card.id)];
            }
            if (g_params.selfDamageScorePenalty != 0.0) {
                perCardBonus -= g_params.selfDamageScorePenalty * nativeImmediateSelfDamage(card);
            }
            if (g_params.drawFirstBonus != 0.0 && isDrawCard(card.id)
                && sim.player.energy > std::max(0, static_cast<int>(card.costForTurn))) {
                perCardBonus += g_params.drawFirstBonus;
            }
            const bool survivalMode = g_params.survivalModeThreshold > 0.0
                && ctx.unblocked >= g_params.survivalModeThreshold
                    * std::max(1, static_cast<int>(sim.player.curHp));
            if (ctype == CardType::ATTACK) {
                double s = g_params.attackBase;
                const int targetIdx = a.getTargetIdx();
                if (targetIdx >= 0 && targetIdx < sim.monsters.monsterCount
                    && sim.monsters.arr[targetIdx].curHp > 0) {
                    const Monster &target = sim.monsters.arr[targetIdx];
                    s += (1.0 - static_cast<double>(target.curHp) / std::max(1, static_cast<int>(target.maxHp))) * g_params.attackFinishOffScale;
                    s -= std::min(static_cast<int>(target.block), 20) * g_params.attackBlockPenaltyScale;
                    const int baseDamage = getBaseDamage(card.id, card.upgraded);
                    if (g_params.attackDamageScoreWeight != 0.0 && baseDamage > 0) {
                        s += g_params.attackDamageScoreWeight
                            * sim.calculateCardDamage(card, targetIdx, baseDamage);
                    }
                    if (g_params.intangibleAttackPenalty != 0.0
                        && target.getStatus<MS::INTANGIBLE>() > 0) {
                        s -= g_params.intangibleAttackPenalty;
                    }
                }
                if (ctx.livingMonsters >= 2 && isAoeCard(card.id)) {
                    s += g_params.aoeBonus;
                }
                s += nativeVulnerableApplyBonus(sim, a, card);
                s += nativeWeakApplyBonus(sim, a, card);
                if (survivalMode) {
                    s *= g_params.survivalModeAttackScale;
                }
                score = s + perCardBonus;
            } else if (ctype == CardType::SKILL) {
                // SKILL branch gets the same term: Shockwave applies Vulnerable and is a Skill.
                // Continuous danger_fraction, not a binary in-danger gate --
                // the old flat 9.0-when-in-danger score was STILL below
                // ATTACK's base 10.0, so attacking structurally always won
                // regardless of actual danger. Confirmed via direct trace:
                // Time Eater (0/15 win rate) had the player at ZERO block at
                // the start of EVERY turn across every seed traced. Scaling
                // continuously with how much of current HP is unblocked-and-
                // incoming lets a genuinely dangerous turn score decisively
                // above any attack.
                const double dangerFraction = ctx.unblocked / std::max(1, sim.player.curHp);
                double s = g_params.skillBase + dangerFraction * g_params.skillDangerScale;
                if (g_params.burstDebuffTimingWeight != 0.0
                    && isBurstDebuffCard(card.id)) {
                    s += g_params.burstDebuffTimingWeight * dangerFraction;
                }
                // Only deprioritize skills when there's no real danger this
                // turn -- hasteWastedDebuffs was meant to catch DEBUFF-
                // applying skills about to be wiped for free (Haste wipes
                // player-applied debuffs ON THE MONSTER, not block), but
                // applied unconditionally to EVERY skill including plain
                // Defend. That's exactly backwards: the Haste threshold
                // (<=50% HP) is typically deep into the fight with several
                // Time Warp Strength stacks already applied, i.e. exactly
                // when defense matters most. Gating on "no real danger"
                // preserves the original intent (skip a wasted debuff when
                // it's safe to instead attack) without ever undermining
                // actual survival.
                if (ctx.hasteWastedDebuffs && dangerFraction < g_params.skillHasteDangerThreshold) {
                    s -= g_params.skillHastePenalty;
                }
                // "We have enough block" gate -- see HeuristicContext::blockSufficient's own
                // comment. Only suppresses the specific defensive-card roster, not skills in
                // general, so a genuinely useful non-defensive skill (draw, utility) isn't
                // penalized just because the turn happens to be safe.
                if (ctx.blockSufficient && isDefensiveCard(card.id)) {
                    s -= g_params.defensiveCardSuppressionPenalty;
                }
                if (g_params.directBlockScoreWeight != 0.0) {
                    const int baseBlock = nativeImmediateBlockBase(card);
                    if (baseBlock > 0) {
                        const int actualBlock = sim.calculateCardBlock(baseBlock);
                        s += g_params.directBlockScoreWeight * std::min(ctx.unblocked,
                            static_cast<double>(actualBlock));
                    }
                }
                s += nativeVulnerableApplyBonus(sim, a, card);
                s += nativeWeakApplyBonus(sim, a, card);
                if (survivalMode) {
                    s *= g_params.survivalModeAttackScale;
                }
                score = s + perCardBonus;
            } else if (ctype == CardType::POWER) {
                // Was a flat powerScore for every Power. Two additive terms now separate them:
                // recurring value discounted by how much fight is left to collect it in, and
                // one-shot value that does not decay. Both weights default to 0.0.
                double s = g_params.powerScore;
                if (g_params.powerPerTurnValueWeight != 0.0) {
                    // Blend the fractional and absolute remaining-fight proxies;
                    // powerHorizonWeight = 0 reproduces the original exactly.
                    const double horizon =
                        (1.0 - g_params.powerHorizonWeight) * ctx.monsterHpRatio
                        + g_params.powerHorizonWeight * ctx.powerHorizon;
                    s += g_params.powerPerTurnValueWeight
                        * nativePowerPerTurnValue(card, ctx) * horizon;
                }
                if (g_params.powerImmediateValueWeight != 0.0) {
                    s += g_params.powerImmediateValueWeight * nativePowerImmediateValue(card);
                }
                if (g_params.bossPowerMultiplier != 1.0 && isBossEncounter(sim.encounter)) {
                    s *= g_params.bossPowerMultiplier;
                }
                score = s + perCardBonus;
            } else {
                score = 1.0 + perCardBonus;  // STATUS/CURSE cards -- not in the tunable set, this branch is rarely reached and never the right choice regardless of exact value
            }
        }
        // Learned rollout-scoring blend, additive on top of the hand-tuned heuristic above --
        // see nativePolicyNetScore's own comment. g_params.policyNetWeight defaults to 0.0 and
        // nativePolicyNetScore itself returns 0.0 with no net loaded, so this is a genuine no-op
        // until both a net is loaded (load_policy_net) AND the weight is explicitly tuned on.
        score += g_params.policyNetWeight * nativePolicyNetScore(sim, a, ctx, stateFeatures);
        // Online per-move average return -- see the MAST block above. Zero-weight by
        // default, and zero for any move the current search has not yet sampled
        // mastMinVisits times, so this is a genuine no-op until tuned on.
        if (nativeMastActive()) {
            score += g_params.mastWeight * nativeMastScore(nativeMastKey(sim, a));
        }
        if (a.getActionType() == search::ActionType::CARD && g_params.silentPoisonApplyBonus != 0.0
            && sim.player.cc == CharacterClass::SILENT) {
            const CardInstance &card = sim.cards.hand[a.getSourceIdx()];
            if (isSilentPoisonApplier(card.id)) {
                score += g_params.silentPoisonApplyBonus;
            }
        }
        return score;
    }

    // Forward declaration -- nativeHeuristicPickFast falls back to this for non-PLAYER_NORMAL
    // states; defined immediately below it.
    search::Action nativeHeuristicPick(const BattleContext &sim, const std::vector<search::Action> &legal);
    // Forward declaration -- defined next to g_policyNet; see its own comment for why every
    // per-action-scoring caller gates nativeLeafFeatures on it.
    bool nativePolicyNetActive();

    // Allocation-free rollout action pick: scores candidate actions as it enumerates them,
    // instead of materializing the full legal-action list first. sts::py::getLegalActions returns
    // a freshly heap-allocated std::vector by value, and the rollout calls it once PER STEP (up to
    // NATIVE_ROLLOUT_MAX_ACTIONS per rollout, one rollout per simulation) -- measured at 71.7% of
    // total simulation cost, making it the single hottest path in the search. Silver Automaton's
    // own rollout (SimpleAgent::chooseBattleCardPlay) avoids the same cost by scanning the hand
    // into stack-allocated fixed_lists and never building an action vector; this is the same idea.
    //
    // Behaviourally identical to nativeHeuristicPick(sim, getLegalActions(sim)): it walks
    // candidates in getLegalActions' exact emission order (cards by hand index then target index,
    // then potions, END_TURN last) and keeps the first strict maximum, so ties resolve the same
    // way. Only PLAYER_NORMAL is handled directly -- CARD_SELECT (and anything else) falls back to
    // the original list-building path, since those enumerate through a different code path
    // (Action::enumerateCardSelectActions) not worth duplicating for a state the rollout rarely
    // reaches.
    // Standard Gumbel-max sampling: argmax_i(score_i/T + G_i) with G_i ~ Gumbel(0,1) draws exactly
    // from softmax(score/T). Lets the streaming picker below sample without materializing or
    // normalizing the candidate set -- it stays a running maximum, just over a perturbed key.
    // Seeded from the caller's search seed, NOT from random_device. It used to be
    // `static thread_local std::mt19937_64 gumbelRng(std::random_device{}())`, which
    // put the rollout's sampling outside every reproducibility guarantee this file
    // otherwise makes. That was dormant while rolloutTemperature was 0.0 (argmax never
    // calls this), so it only surfaced when tuning first moved that parameter off zero:
    // identical bc + identical search_seed then returned different actions, and paired
    // comparisons silently lost their common random numbers.
    thread_local std::mt19937_64 g_gumbelRng(0x9E3779B97F4A7C15ULL);
    // Draw-order resampling in the rollout (see g_params.honestDrawOrder). Its own stream
    // rather than a shared one: whether honest draw order is on must not shift the Gumbel
    // sequence the rollout policy samples from, or the two parameters stop being separable
    // and every paired comparison between them silently loses its common random numbers.
    thread_local std::mt19937_64 g_drawShuffleRng(0xD1B54A32D192ED03ULL);

    void nativeSeedGumbel(std::uint64_t seed) {
        // SplitMix64 finalizer: run_mcts_search's seeds are often small and highly
        // correlated across calls (see how `play` derives them), and mt19937_64 seeded
        // with near-identical small values produces visibly correlated early output.
        std::uint64_t z = seed + 0x9E3779B97F4A7C15ULL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        g_gumbelRng.seed(z ^ (z >> 31));
        // Same search seed, independent stream -- see g_drawShuffleRng. A second SplitMix64
        // pass with a different additive constant, so the two streams are decorrelated
        // rather than offset copies of each other.
        std::uint64_t d = seed + 0xD1B54A32D192ED03ULL;
        d = (d ^ (d >> 30)) * 0xBF58476D1CE4E5B9ULL;
        d = (d ^ (d >> 27)) * 0x94D049BB133111EBULL;
        g_drawShuffleRng.seed(d ^ (d >> 31));
    }

    // Permute the draw pile without touching its contents. This is the whole mechanism
    // behind g_params.honestDrawOrder -- see TunableParams for why it exists and what it
    // costs. Fisher-Yates over CardManager's fixed_list, which supports indexed assignment
    // (set_draw_pile_order's binding already relies on that).
    // An action whose index is bound to the CURRENT draw-pile order. Card-select
    // actions are enumerated as positions into `drawPile` and validated by what
    // sits at that position -- SECRET_WEAPON, for instance, is only legal if
    // `drawPile[idx].getType() == ATTACK` (Action.cpp) -- and SCRY indexes the
    // top N. Permuting the pile between enumeration (against the PARENT state)
    // and execution therefore re-points the index at a different card, and the
    // engine dumps the whole BattleContext to stderr when the action it is
    // handed is no longer valid. Shuffling for these is also pointless: a
    // card-select screen shows the player the pile, so the choice is over card
    // identities and the order carries no hidden information to protect.
    bool nativeActionBindsPileOrder(const search::Action &action) {
        switch (action.getActionType()) {
            case search::ActionType::SINGLE_CARD_SELECT:
            case search::ActionType::MULTI_CARD_SELECT:
            case search::ActionType::SCRY:
                return true;
            default:
                return false;
        }
    }

    void nativeShuffleDrawPile(BattleContext &bc, std::mt19937_64 &gen) {
        auto &pile = bc.cards.drawPile;
        for (int i = static_cast<int>(pile.size()) - 1; i > 0; --i) {
            std::uniform_int_distribution<int> dist(0, i);
            const int j = dist(gen);
            const CardInstance tmp = pile[i];
            pile[i] = pile[j];
            pile[j] = tmp;
        }
    }

    bool nativeHonestDrawOrder() {
        return g_params.honestDrawOrder != 0.0;
    }


    double nativeGumbelNoise() {
        std::uniform_real_distribution<double> u(1e-12, 1.0);
        return -std::log(-std::log(u(g_gumbelRng)));
    }

    search::Action nativeHeuristicPickFast(const BattleContext &sim) {
        if (sim.inputState != InputState::PLAYER_NORMAL) {
            return nativeHeuristicPick(sim, sts::py::getLegalActions(sim));
        }

        const HeuristicContext ctx = nativeComputeHeuristicContext(sim);
        // Gated: these features feed nativePolicyNetScore alone, and cost a full monster loop
        // with per-monster damage prediction. See nativePolicyNetActive's own comment.
        const auto stateFeatures = nativePolicyNetActive()
                ? nativeLeafFeatures(sim)
                : std::array<double, NATIVE_LEAF_FEATURE_DIM>{};

        search::Action best{search::ActionType::END_TURN};
        bool haveBest = false;
        double bestScore = -std::numeric_limits<double>::infinity();

        // END_TURN is unconditionally emitted in PLAYER_NORMAL, so under timeWarpRisk it is always
        // available as the seeded best -- matching nativeHeuristicPick's own END_TURN scan.
        // Comparison key: the raw heuristic score at T=0, or a Gumbel-perturbed one above it (see
        // g_params.rolloutTemperature). Applied to EVERY candidate including the timeWarpRisk
        // END_TURN seed, so the sampling distribution stays consistent across all of them.
        const double temp = g_params.rolloutTemperature;
        const auto key = [&](double score) {
            return temp > 0.0 ? score / temp + nativeGumbelNoise() : score;
        };

        if (ctx.timeWarpRisk) {
            bestScore = key(g_params.endTurnTimeWarpRiskScore);
            haveBest = true;
        }

        const auto consider = [&](const search::Action &a) {
            const double k = key(nativeScoreAction(sim, a, ctx, stateFeatures));
            if (k > bestScore) {
                bestScore = k;
                best = a;
                haveBest = true;
            }
        };

        if (sim.isCardPlayAllowed()) {
            for (int handIdx = 0; handIdx < sim.cards.cardsInHand; ++handIdx) {
                const auto &c = sim.cards.hand[handIdx];
                if (!c.canUseOnAnyTarget(sim)) {
                    continue;
                }
                if (c.requiresTarget()) {
                    for (int tIdx = 0; tIdx < sim.monsters.monsterCount; ++tIdx) {
                        if (!sim.monsters.arr[tIdx].isTargetable()) {
                            continue;
                        }
                        consider(search::Action(search::ActionType::CARD, handIdx, tIdx));
                    }
                } else {
                    consider(search::Action(search::ActionType::CARD, handIdx));
                }
            }
        }

        for (int potionIdx = 0; potionIdx < sim.potionCapacity; ++potionIdx) {
            const auto p = sim.potions[potionIdx];
            if (p == Potion::INVALID || p == Potion::EMPTY_POTION_SLOT) {
                continue;
            }
            consider(search::Action(search::ActionType::POTION, potionIdx, 6));  // discard
            if (p == Potion::FAIRY_POTION) {
                continue;  // not manually drinkable -- see isValidPotionAction
            }
            if (potionRequiresTarget(p)) {
                for (int tIdx = 0; tIdx < sim.monsters.monsterCount; ++tIdx) {
                    if (!sim.monsters.arr[tIdx].isTargetable()) {
                        continue;
                    }
                    consider(search::Action(search::ActionType::POTION, potionIdx, tIdx));
                }
            } else {
                consider(search::Action(search::ActionType::POTION, potionIdx, 0));
            }
        }

        // No scoreable action means the legal list held END_TURN alone, so the original's
        // `return legal[0]` fallback is END_TURN -- which `best` already holds.
        (void)haveBest;
        return best;
    }

    search::Action nativeHeuristicPick(const BattleContext &sim, const std::vector<search::Action> &legal) {
        const HeuristicContext ctx = nativeComputeHeuristicContext(sim);
        // Gated: these features feed nativePolicyNetScore alone, and cost a full monster loop
        // with per-monster damage prediction. See nativePolicyNetActive's own comment.
        const auto stateFeatures = nativePolicyNetActive()
                ? nativeLeafFeatures(sim)
                : std::array<double, NATIVE_LEAF_FEATURE_DIM>{};

        const search::Action *best = nullptr;
        double bestScore = -std::numeric_limits<double>::infinity();

        if (ctx.timeWarpRisk) {
            for (const auto &a : legal) {
                if (a.getActionType() == search::ActionType::END_TURN) {
                    best = &a;
                    bestScore = g_params.endTurnTimeWarpRiskScore;
                    break;
                }
            }
        }

        for (const auto &a : legal) {
            if (a.getActionType() == search::ActionType::END_TURN) {
                continue;
            }
            const double score = nativeScoreAction(sim, a, ctx, stateFeatures);
            if (score > bestScore) {
                bestScore = score;
                best = &a;
            }
        }
        if (best == nullptr) {
#ifdef sts_asserts
            // An empty legal list means getLegalActions has no case for
            // sim.inputState, not that the position is genuinely stuck -- that
            // is a missing enumeration upstream, and indexing legal[0] here
            // turns it into a bare segfault a long way from the cause. Cost an
            // afternoon once when InputState::SCRY was unhandled (Melange on an
            // Ironclad, via lightspeed/relics.py's pool); assert loudly instead.
            assert(!legal.empty() && "getLegalActions returned no actions for this inputState");
#endif
            return legal[0];
        }
        return *best;
    }

    // Heuristic-informed visit ORDER for a freshly expanded tree node's
    // unavoidable first visit to every legal action -- NOT the rollout
    // pick (nativeHeuristicPick picks ONE action to actually play; this
    // instead ranks ALL legal actions so nativeSelectIdx tries the
    // heuristically-best ones FIRST during the "every edge gets visited
    // once" cold-start phase, rather than whatever order getLegalActions
    // happened to enumerate them in). Motivation: a decision node with
    // 10+ legal actions currently burns 10+ simulations trying every one
    // once -- including obviously-bad options -- before UCB1
    // differentiation has any effect at all; for a limited sim budget
    // (100s here, not Silverbot's 1000s+), that's a real cost. This does
    // NOT change the exploitation formula at all (Q + C*sqrt(...) once
    // N[i] > 0 is untouched) -- only which action gets its guaranteed
    // first try first. Silver Automaton has no equivalent of this (its
    // rollout agent and its tree search are independent, unlike here
    // where the same heuristic now informs both); this is closer to a
    // cheap, rule-based stand-in for PUCT's prior term than a port of
    // anything Silverbot does.
    // Single source of truth for the per-action heuristic scores that both the cold-start visit
    // order and the PUCT prior (nativeHeuristicPriors below) are derived from -- factored out of
    // nativeHeuristicVisitOrder so the two consumers can never compute different scores for the
    // same node (they used to be two separate loops before the PUCT prior was added).
    std::vector<double> nativeHeuristicScores(
            const BattleContext &sim,
            const std::vector<search::Action> &legal) {
        const HeuristicContext ctx = nativeComputeHeuristicContext(sim);
        // Gated: these features feed nativePolicyNetScore alone, and cost a full monster loop
        // with per-monster damage prediction. See nativePolicyNetActive's own comment.
        const auto stateFeatures = nativePolicyNetActive()
                ? nativeLeafFeatures(sim)
                : std::array<double, NATIVE_LEAF_FEATURE_DIM>{};
        const int n = static_cast<int>(legal.size());
        std::vector<double> scores(n);
        for (int i = 0; i < n; ++i) {
            if (legal[i].getActionType() == search::ActionType::END_TURN) {
                scores[i] = ctx.timeWarpRisk ? g_params.endTurnTimeWarpRiskScore : 5.0;
            } else {
                scores[i] = nativeScoreAction(sim, legal[i], ctx, stateFeatures);
            }
        }
        return scores;
    }

    std::vector<int> nativeHeuristicVisitOrder(const std::vector<double> &scores) {
        const int n = static_cast<int>(scores.size());
        std::vector<int> order(n);
        for (int i = 0; i < n; ++i) {
            order[i] = i;
        }
        std::sort(order.begin(), order.end(), [&](int a, int b) { return scores[a] > scores[b]; });
        return order;
    }

    // PUCT-style prior: softmax of the same heuristic scores used for the cold-start visit
    // order, over the node's legal actions. Not a learned policy -- a normalized version of the
    // existing hand-tuned heuristic, in the same spirit as AlphaMapleSAT's deductive (non-neural)
    // PUCT prior. Max-subtracted before exp() for numerical stability (scores can be tens in
    // magnitude, e.g. skillDangerScale's contribution). Callers must keep puctTemperature nonzero
    // (default 10.0, same never-change-mid-flight trust as every other g_params field -- see
    // set_search_params's docstring).
    std::vector<double> nativeHeuristicPriors(const std::vector<double> &scores) {
        const int n = static_cast<int>(scores.size());
        std::vector<double> priors(n);
        double maxScore = -std::numeric_limits<double>::infinity();
        for (double s : scores) {
            maxScore = std::max(maxScore, s);
        }
        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
            priors[i] = std::exp((scores[i] - maxScore) / g_params.puctTemperature);
            sum += priors[i];
        }
        for (int i = 0; i < n; ++i) {
            priors[i] /= sum;
        }
        return priors;
    }

    // maxTurn: absolute turn number (see NATIVE_MAX_TURNS_PER_SEARCH's own
    // comment) at which the rollout stops early and falls back to the static
    // potential, same as hitting the action cap -- defaults to "no cap" so
    // standalone/direct callers (Python's bc.heuristic_playout(), scripts
    // calling it outside a real search) keep their original unbounded-by-
    // turn behavior; only nativeSimulate's internal calls pass a real cap.
    // raveTrace (optional): when non-null, every action this rollout plays is appended as raw
    // Action bits. RAVE credits actions played anywhere below a node, and the rollout is where the
    // overwhelming majority of a simulation's actions occur, so without this the AMAF statistics
    // would see only the handful of in-tree actions and lose nearly all of their sample advantage.
    bool nativePairingActive();
    std::uint64_t nativePairSeed(int turn, int localSampleIndex);

    double nativeHeuristicPlayout(const BattleContext &bc, int maxTurn = std::numeric_limits<int>::max(),
                                   std::vector<std::uint32_t> *raveTrace = nullptr) {
        BattleContext sim(bc);
        // One independent draw order per rollout. This is where most of the clairvoyance
        // was actually being spent: the rollout is ~90% of search time and plays many turns
        // ahead, so a fixed order let every playout read the entire future of the fight.
        // Shuffling once here makes each rollout a single honest sample from the belief --
        // draws inside it still come off the sampled order, which is what a Monte-Carlo
        // sample should be, and a mid-rollout reshuffle uses the engine's own seeded
        // shuffleRng as before.
        if (nativeHonestDrawOrder()) {
            if (nativePairingActive()) {
                std::mt19937_64 pairedRng(nativePairSeed(sim.turn, -1));
                nativeShuffleDrawPile(sim, pairedRng);
            } else {
                nativeShuffleDrawPile(sim, g_drawShuffleRng);
            }
        }
        for (int step = 0; step < NATIVE_ROLLOUT_MAX_ACTIONS; ++step) {
            if (sim.outcome != Outcome::UNDECIDED) {
                // nativeExpectimaxTerminalReward, not nativeTerminalReward directly -- this is
                // the dominant source of leaf values feeding the whole tree (one rollout per
                // newly-expanded node), so it's exactly where the loss-progress-credit gap
                // mattered most. See that function's own comment.
                return NATIVE_W_SHAPE * nativeExpectimaxTerminalReward(sim, sim.turn);
            }
            if (sim.turn >= maxTurn) {
                return NATIVE_W_SHAPE * nativePotential(sim);
            }
            const search::Action action = nativeHeuristicPickFast(sim);
            if (raveTrace != nullptr) {
                raveTrace->push_back(action.bits);
            }
            // Recorded BEFORE execute, while `sim` still holds the hand the action indexes
            // -- a MAST key is the card's identity, which the source index alone cannot give.
            if (nativeMastActive()) {
                nativeMastRecord(sim, action);
            }
            const std::size_t drawBeforeStep = sim.cards.drawPile.size();
            action.execute(sim);
            if (g_params.honestDrawOrder >= 2.0
                && sim.cards.drawPile.size() < drawBeforeStep) {
                nativeShuffleDrawPile(sim, g_drawShuffleRng);
            }
        }
        return NATIVE_W_SHAPE * nativePotential(sim);
    }

    // --- native (C++) port of expectimax_search.py's full MCTS loop ---
    //
    // Builds on nativeHeuristicPlayout/nativePotential/nativeTerminalReward
    // above (see their own comment block for the general "why native"
    // rationale and DRIFT WARNING pattern -- the same warning applies here:
    // C_UCB/WC_CHANCE/WA_CHANCE/GAMMA/MAX_CALL_DEPTH/BREWING_THREAT_ESTIMATE
    // below duplicate expectimax_search.py's own module-level constants).
    //
    // ONE STRUCTURAL DIFFERENCE from the Python original, by necessity: DPW's
    // revisit-an-existing-outcome step samples with probability proportional
    // to each sibling's visit count, and the Python version draws that
    // sample via np.random.choice -- numpy's own global RNG state, NOT bc's
    // gameplay RNG streams. A native port can't reproduce numpy's specific
    // PRNG algorithm/consumption sequence without also porting numpy itself,
    // so this uses its own std::mt19937_64 for that one weighted-choice step
    // instead. This is a difference in EXPLORATION-ORDER randomness only,
    // not game-state randomness (that still flows through bc's own RNG
    // streams via the identical seed_rng/decorrelate_rng formulas, just
    // inlined here rather than called through the Python-facing bindings).
    // The same applies to CRN seeding (_crn_seed): Python's version hashes
    // (crn_base, state_key, local_sample_index) via CPython's tuple hash,
    // which isn't a stable algorithm worth reproducing bit-for-bit in C++;
    // this uses its own deterministic combiner over the same three inputs,
    // which satisfies CRN's actual requirement (repeatable given the same
    // inputs) without needing to match Python's specific hash values.
    //
    // Net effect: the native and Python searches do NOT produce bit-
    // identical trees even given equal starting state and crn_base, unlike
    // state_key_bundle/heuristic_playout above, which are exact. Correctness
    // was instead validated via (1) exact unit-level match of every
    // deterministic sub-piece (UCB1 formula, DPW widening-cap formula, RNG-
    // consumption classification, the state key itself) against the Python
    // versions on synthetic/paired inputs, and (2) statistical win-rate/HP
    // parity over many full episodes across several encounters, including
    // Awakened One (the encounter that crashed twice earlier this session
    // from exactly this class of bug) -- see scratchpad/
    // validate_mcts_native.py.
    // g_params.cUcb: UCB1 exploration constant for deterministic-destined
    // edges (see TunableParams' own comment for why this and the other
    // fields below are runtime-mutable rather than constexpr).
    //
    // g_params.cUcbChance: separate exploration constant for edges whose
    // destination is a chance node (currently only END_TURN, the sole
    // stochastic action-type here -- some CARD plays turn out to be
    // stochastic too via the RNG-probe classification, but that's
    // discovered only after first execution, so it can't bias the
    // SELECTION decision itself; only END_TURN is known in advance).
    // Silver Automaton's own evaluateEdge does exactly this
    // (BattleSearcher.cpp: `exploration = edge.node->isRandomNode ?
    // explorationParameterChance : explorationParameter`) -- read directly
    // while looking for real algorithmic (not just heuristic-tuning)
    // differences. The rationale: exploring INTO inherent randomness has a
    // different exploration/exploitation shape than exploring a
    // deterministic choice -- a single sample of a chance node's outcome
    // tells you less about its true expectation than a single sample of a
    // deterministic child tells you about ITS value, so it can be
    // worthwhile to weight that uncertainty differently.
    //
    // g_params.wcChance/waChance: DPW widening-cap formula parameters.
    // g_params.brewingThreatEstimate: must conceptually match
    // expectimax_search.py's own BREWING_THREAT_ESTIMATE, though that
    // Python constant is no longer kept in lockstep as a hard rule (see
    // choose_action_python's docstring) -- tuning this one independently
    // is fine now.
    constexpr double NATIVE_GAMMA = 0.99;
    constexpr int NATIVE_MAX_CALL_DEPTH = 150;
    // Absolute cap on simulated turns past the search's own root, checked
    // independently of NATIVE_MAX_CALL_DEPTH/NATIVE_ROLLOUT_MAX_ACTIONS (both
    // of which cap ACTIONS, not turns, and neither prevents compounding
    // ACROSS a deep tree branch chained into its own leaf rollout). Found
    // necessary while investigating a genuine unbounded-card-generation path:
    // Dual Wield (a real Ironclad card that duplicates an Attack/Power card)
    // played repeatedly across a sufficiently long simulated line can grow a
    // card pile past any fixed capacity tried (64/256/512/4000 all failed) --
    // not because any single mechanic is unbounded, but because nothing
    // capped how many TOTAL simulated turns one MCTS decision could explore
    // (150 tree-deep + 200 more per leaf rollout, chainable across many of a
    // decision's simulations). 20 turns past the root is generous for any
    // real fight's remaining length within a single decision's search.
    constexpr int NATIVE_MAX_TURNS_PER_SEARCH = 20;

    // expectimax_search.py's OWN _potential -- a STANDALONE reimplementation
    // of env.py's potential() shape (W_HP * player_hp, minus each living
    // monster's HP and BETA-weighted incoming damage), NOT a call into
    // nativePotential/env.py's actual potential(), for two stacked reasons:
    // (1) the brewing-threat flat penalty (see module comment on
    // g_params.brewingThreatEstimate) was always kept separate from the
    // training-reward-shared function; (2) the incoming-damage term here
    // uses nativePredictedIncomingDamage (the corrected Strength/Weak/
    // Vulnerable-aware version above), NOT getMoveBaseDamage's raw table
    // lookup the way nativePotential (and env.py's potential()) still do --
    // fixing that gap in nativePotential/env.py would be a training-reward
    // change (checkpoint-invalidating, out of scope here); fixing it in
    // THIS function costs nothing since nothing depends on its exact shape.
    // nativePotential itself is UNCHANGED and still used, deliberately, by
    // nativeHeuristicPlayout's rollout leaf fallback, matching
    // az_search.py's _heuristic_playout (imported unchanged by
    // expectimax_search.py) calling env.py's real potential() directly.
    double nativeExpectimaxPotential(const BattleContext &bc) {
        if (bc.outcome != Outcome::UNDECIDED) {
            return 0.0;
        }
        // Block counts toward potential (see g_params.blockWeight). Capped at the damage actually
        // incoming: block beyond what any monster will throw this turn expires unused, so paying
        // for it is not an improvement and an uncapped term would reward over-blocking.
        double phi = NATIVE_W_HP * bc.player.curHp;
        const double vulnMult = bc.player.hasStatus<PS::VULNERABLE>() ? 1.5 : 1.0;
        double totalIncoming = 0.0;
        for (int i = 0; i < bc.monsters.monsterCount; ++i) {
            const Monster &m = bc.monsters.arr[i];
            if (m.halfDead) {
                phi -= m.maxHp;
                continue;
            }
            if (m.curHp <= 0) {
                continue;
            }
            const int dmg = nativePredictedIncomingDamage(bc, m, vulnMult);
            totalIncoming += dmg;
            phi -= m.curHp + g_params.enemyBlockWeight * std::max(0, m.block)
                    + NATIVE_BETA * dmg;
            if (dmg == 0) {
                phi -= NATIVE_BETA * g_params.brewingThreatEstimate;
            }
        }
        phi += g_params.blockWeight * std::min(static_cast<double>(bc.player.block), totalIncoming);
        return phi;
    }

    double nativeExpectimaxDenseReward(const BattleContext &bcBefore, const BattleContext &bcAfter) {
        return NATIVE_W_SHAPE * (NATIVE_GAMMA * nativeExpectimaxPotential(bcAfter) - nativeExpectimaxPotential(bcBefore));
    }

    // Single source of truth for the leaf value-function feature vector -- the
    // SAME 10 raw features feed both the linear estimate (nativeLeafValueEstimate)
    // and, once trained, the native value-net (nativeValueNetEstimate) AND the
    // Python-side training-data collection (leaf_features() binding). Keeping
    // one extractor guarantees train-time and inference-time features can never
    // silently diverge. Raw magnitudes (no signs) -- the linear weights / net
    // carry the sign. Order is load-bearing: it must match VALUE_NET_FEATURES
    // in the training script and the g_params.vf* application below. (NATIVE_LEAF_FEATURE_DIM
    // itself now lives near the other NATIVE_* constants at the top of the file -- see there.)
    std::array<double, NATIVE_LEAF_FEATURE_DIM> nativeLeafFeatures(const BattleContext &bc) {
        const double vulnMult = bc.player.hasStatus<PS::VULNERABLE>() ? 1.5 : 1.0;
        double monsterHp = 0.0, incoming = 0.0;
        int alive = 0;
        for (int i = 0; i < bc.monsters.monsterCount; ++i) {
            const Monster &m = bc.monsters.arr[i];
            if (m.halfDead) {
                monsterHp += m.maxHp;  // about to revive -- count as full threat
                continue;
            }
            if (m.curHp <= 0) {
                continue;
            }
            ++alive;
            monsterHp += m.curHp;
            incoming += nativePredictedIncomingDamage(bc, m, vulnMult);
        }
        // --- composition of every pile, which the original ten could not see ---
        int handAttack = 0, handSkill = 0, handPower = 0, handDead = 0;
        double handCost = 0.0;
        for (int i = 0; i < bc.cards.cardsInHand; ++i) {
            const CardInstance &c = bc.cards.hand[i];
            switch (cardTypes[static_cast<int>(c.id)]) {
                case CardType::ATTACK: ++handAttack; break;
                case CardType::SKILL:  ++handSkill;  break;
                case CardType::POWER:  ++handPower;  break;
                default:               ++handDead;   break;  // status / curse
            }
            if (c.costForTurn >= 0) {
                handCost += c.costForTurn;
            }
        }
        int deckAttack = 0, deckSkill = 0, deckPower = 0, deckDead = 0;
        const auto countRest = [&](const auto &pile, std::size_t n) {
            for (std::size_t i = 0; i < n; ++i) {
                switch (cardTypes[static_cast<int>(pile[i].id)]) {
                    case CardType::ATTACK: ++deckAttack; break;
                    case CardType::SKILL:  ++deckSkill;  break;
                    case CardType::POWER:  ++deckPower;  break;
                    default:               ++deckDead;   break;
                }
            }
        };
        countRest(bc.cards.drawPile, bc.cards.drawPile.size());
        countRest(bc.cards.discardPile, bc.cards.discardPile.size());

        // Worst single hit matters separately from the total: 30 incoming from one
        // monster is a different problem from 10 each from three.
        double worstHit = 0.0;
        int monstersVulnerable = 0, monstersWeak = 0;
        for (int i = 0; i < bc.monsters.monsterCount; ++i) {
            const Monster &m = bc.monsters.arr[i];
            if (m.curHp <= 0 && !m.halfDead) {
                continue;
            }
            worstHit = std::max(worstHit, static_cast<double>(
                nativePredictedIncomingDamage(bc, m, vulnMult)));
            if (m.vulnerable > 0) ++monstersVulnerable;
            if (m.weak > 0) ++monstersWeak;
        }
        int potions = 0;
        for (int i = 0; i < bc.potionCapacity; ++i) {
            const auto p = bc.potions[i];
            if (p != Potion::EMPTY_POTION_SLOT && p != Potion::INVALID) {
                ++potions;
            }
        }
        const auto status = [&](PlayerStatus s) {
            return static_cast<double>(bc.player.getStatusRuntime(s));
        };
        return {
            // [0..9] FROZEN -- g_params.vf* index these positionally.
            static_cast<double>(bc.player.curHp),
            static_cast<double>(bc.player.block),
            static_cast<double>(bc.player.energy),
            static_cast<double>(bc.player.strength),
            static_cast<double>(bc.player.dexterity),
            status(PlayerStatus::METALLICIZE),
            monsterHp,
            incoming,
            static_cast<double>(alive),
            static_cast<double>(bc.turn),
            // [10..13] resources the ten omitted entirely
            static_cast<double>(bc.player.maxHp),
            static_cast<double>(potions),
            worstHit,
            monsterHp > 0.0 ? incoming / std::max(1.0, static_cast<double>(bc.player.curHp)
                                                       + bc.player.block) : 0.0,
            // [14..19] hand composition -- what can actually be played right now
            static_cast<double>(bc.cards.cardsInHand),
            static_cast<double>(handAttack),
            static_cast<double>(handSkill),
            static_cast<double>(handPower),
            static_cast<double>(handDead),
            handCost,
            // [20..24] what is left in the deck, and where
            static_cast<double>(bc.cards.drawPile.size()),
            static_cast<double>(bc.cards.discardPile.size()),
            static_cast<double>(deckAttack),
            static_cast<double>(deckSkill),
            static_cast<double>(deckPower + deckDead),
            // [25..27] player debuffs that change every damage number
            status(PlayerStatus::VULNERABLE),
            status(PlayerStatus::WEAK),
            status(PlayerStatus::FRAIL),
            // [28..29] monster debuffs the search spends terms trying to apply
            static_cast<double>(monstersVulnerable),
            static_cast<double>(monstersWeak),
        };
    }

    // Enriched static value estimate for a NON-terminal state, used as the
    // leaf value in "value"/"truncated" leaf-eval modes (terminal states are
    // handled by nativeLeafValue's own outcome check, never reach here). This
    // is the tunable analog of Silverbot's evaluateEndState: a weighted sum of
    // the nativeLeafFeatures above, whose weights (g_params.vf*) are CMA-ES-
    // tuned so this estimate can STAND IN for a full rollout-to-terminal (see
    // tune_value_leaf.py). Held-out this reaches ~53% win at 3.4x rollout speed
    // -- a real recovery from the un-tuned 29%, but a linear ceiling below
    // rollout's 83%, which is why the value-NET (nonlinear) is the next step.
    double nativeLeafValueEstimate(const BattleContext &bc) {
        const auto f = nativeLeafFeatures(bc);
        return g_params.vfHp * f[0] + g_params.vfBlock * f[1] + g_params.vfEnergy * f[2]
             + g_params.vfStrength * f[3] + g_params.vfDexterity * f[4] + g_params.vfMetallicize * f[5]
             - g_params.vfMonsterHp * f[6] - g_params.vfIncoming * f[7]
             - g_params.vfAlive * f[8] - g_params.vfTurn * f[9];
    }

    // Native forward pass of the trained leaf value-net (train_value_net.py,
    // loaded via load_value_net). A tiny MLP over the same nativeLeafFeatures,
    // used in the "valuenet" leaf-eval mode -- the nonlinear successor to the
    // linear nativeLeafValueEstimate (which held-out at ~53% win; the net beats
    // the linear model at predicting rollout value, R^2 0.36 vs 0.21). Weights
    // are plain nested vectors (small net -- ~10->32->32->1 -- so the per-leaf
    // cost stays far below a rollout, the whole point). The net is trained on
    // W_SHAPE-scaled rollout values with target de-normalization folded into its
    // output layer, so its output is directly comparable to what the ROLLOUT
    // leaf returns -- NO extra W_SHAPE multiply here (unlike the linear estimate,
    // whose output is in raw potential units and IS ×W_SHAPE'd by the caller).
    struct ValueNetLayer {
        std::vector<std::vector<double>> W;  // [out][in]
        std::vector<double> b;               // [out]
        bool applyTanh;
    };
    struct ValueNet {
        bool loaded = false;
        std::array<double, NATIVE_LEAF_FEATURE_DIM> mu{}, sd{};
        std::vector<ValueNetLayer> layers;
    };
    ValueNet g_valueNet;

    double nativeValueNetEstimate(const BattleContext &bc) {
        const auto f = nativeLeafFeatures(bc);
        std::vector<double> x(NATIVE_LEAF_FEATURE_DIM);
        for (int i = 0; i < NATIVE_LEAF_FEATURE_DIM; ++i) {
            x[i] = (f[i] - g_valueNet.mu[i]) / g_valueNet.sd[i];
        }
        for (const auto &layer : g_valueNet.layers) {
            const int outDim = static_cast<int>(layer.b.size());
            std::vector<double> y(outDim);
            for (int o = 0; o < outDim; ++o) {
                double acc = layer.b[o];
                const auto &row = layer.W[o];
                for (int i = 0; i < static_cast<int>(x.size()); ++i) {
                    acc += row[i] * x[i];
                }
                y[o] = layer.applyTanh ? std::tanh(acc) : acc;
            }
            x = std::move(y);
        }
        return x[0];
    }

    // Learned rollout-scoring net: a small MLP over [nativeLeafFeatures (state), nativeActionFeatures
    // (the one action being scored)] concatenated, trained (see train_policy_net.py) to imitate
    // expectimax search's OWN visit-count preference among a decision's legal actions -- a DIFFERENT
    // use of learning than g_valueNet above (which replaces a full rollout with a static state
    // estimate, and measurably lost to rollout: ~53% vs 83% win rate held-out). This one never
    // replaces the rollout -- it's blended (g_params.policyNetWeight) into the SAME hand-tuned
    // per-action heuristic (nativeScoreAction) that already drives every rollout step, the same
    // relationship a PUCT policy prior has to search: nudging which actions the rollout/tree
    // favor, not answering "how good is this state" on its own.
    // Reuses ValueNetLayer's exact forward-pass shape (W/b/applyTanh) -- same MLP shape, different
    // input semantics, so no reason for a second struct definition.
    struct PolicyNet {
        bool loaded = false;
        std::vector<double> mu, sd;  // sized NATIVE_LEAF_FEATURE_DIM + NATIVE_ACTION_FEATURE_DIM (not a
                                      // fixed std::array like ValueNet's, since this concatenated
                                      // dimension only exists at this one call site)
        std::vector<ValueNetLayer> layers;
    };
    PolicyNet g_policyNet;

    // True only when the policy-net blend can actually change a score. Callers use this to skip
    // computing nativeLeafFeatures entirely: those features exist ONLY to feed
    // nativePolicyNetScore, they cost a full monster loop with per-monster damage prediction, and
    // the rollout recomputes them once per step -- so with no net loaded (the default, and the
    // current production config) that is a full redundant damage loop per rollout step feeding a
    // function that returns 0.0 and is then multiplied by a 0.0 weight.
    bool nativePolicyNetActive() {
        return g_policyNet.loaded && g_params.policyNetWeight != 0.0;
    }

    double nativePolicyNetScore(const BattleContext &sim, const search::Action &a, const HeuristicContext &ctx,
                                 const std::array<double, NATIVE_LEAF_FEATURE_DIM> &stateFeatures) {
        if (!g_policyNet.loaded) {
            return 0.0;
        }
        const auto &sf = stateFeatures;
        const auto af = nativeActionFeatures(sim, a, ctx);
        std::vector<double> x(NATIVE_LEAF_FEATURE_DIM + NATIVE_ACTION_FEATURE_DIM);
        for (int i = 0; i < NATIVE_LEAF_FEATURE_DIM; ++i) {
            x[i] = (sf[i] - g_policyNet.mu[i]) / g_policyNet.sd[i];
        }
        for (int i = 0; i < NATIVE_ACTION_FEATURE_DIM; ++i) {
            const int j = NATIVE_LEAF_FEATURE_DIM + i;
            x[j] = (af[i] - g_policyNet.mu[j]) / g_policyNet.sd[j];
        }
        for (const auto &layer : g_policyNet.layers) {
            const int outDim = static_cast<int>(layer.b.size());
            std::vector<double> y(outDim);
            for (int o = 0; o < outDim; ++o) {
                double acc = layer.b[o];
                const auto &row = layer.W[o];
                for (int i = 0; i < static_cast<int>(x.size()); ++i) {
                    acc += row[i] * x[i];
                }
                y[o] = layer.applyTanh ? std::tanh(acc) : acc;
            }
            x = std::move(y);
        }
        return x[0];
    }

    // Leaf-evaluation mode: how a newly-expanded MCTS leaf gets its value.
    // Profiling this session established that a full rollout to terminal
    // (ROLLOUT, the original/default) is ~90% of total search time -- each
    // simulation is basically one rollout, so the rollout IS the search
    // cost, not the tree machinery around it. VALUE skips the playout
    // entirely and returns the static potential (nativeExpectimaxPotential,
    // the same value function already computed for dense rewards) directly
    // -- ~10x cheaper per leaf, but a coarser value estimate that can miss
    // tactical sequences a rollout would find. TRUNCATED is the middle
    // ground: play g_truncatedRolloutSteps actions, then apply the static
    // potential -- keeps near-term tactics, cuts most of the playout length.
    //
    // Thread/process-safety: g_leafEvalMode/g_truncatedRolloutSteps are the
    // same kind of process-global mutable state as g_params -- see
    // set_search_params's docstring for the identical never-change-mid-flight
    // rule. Set once before an evaluation, or use separate processes.
    // VALUENET: trained MLP leaf estimate (nativeValueNetEstimate), the
    // nonlinear successor to VALUE's linear estimate. Requires load_value_net
    // first (g_valueNet.loaded) -- set_leaf_eval_mode enforces that.
    enum class LeafEvalMode { ROLLOUT, VALUE, TRUNCATED, VALUENET };
    LeafEvalMode g_leafEvalMode = LeafEvalMode::ROLLOUT;
    int g_truncatedRolloutSteps = 3;

    double nativeLeafValue(const BattleContext &bc, int maxTurn = std::numeric_limits<int>::max(),
                            std::vector<std::uint32_t> *raveTrace = nullptr) {
        if (g_leafEvalMode == LeafEvalMode::ROLLOUT) {
            return nativeHeuristicPlayout(bc, maxTurn, raveTrace);
        }
        if (g_leafEvalMode == LeafEvalMode::VALUENET) {
            if (bc.outcome != Outcome::UNDECIDED) {
                return NATIVE_W_SHAPE * nativeExpectimaxTerminalReward(bc, bc.turn);
            }
            return nativeValueNetEstimate(bc);  // already W_SHAPE-scaled, see its comment
        }
        if (g_leafEvalMode == LeafEvalMode::VALUE) {
            if (bc.outcome != Outcome::UNDECIDED) {
                return NATIVE_W_SHAPE * nativeExpectimaxTerminalReward(bc, bc.turn);
            }
            return NATIVE_W_SHAPE * nativeLeafValueEstimate(bc);
        }
        // TRUNCATED: short rollout, then the enriched static estimate (or
        // terminal if the fight ends within the truncation window).
        BattleContext sim(bc);
        for (int step = 0; step < g_truncatedRolloutSteps; ++step) {
            if (sim.outcome != Outcome::UNDECIDED) {
                return NATIVE_W_SHAPE * nativeExpectimaxTerminalReward(sim, sim.turn);
            }
            if (sim.turn >= maxTurn) {
                return NATIVE_W_SHAPE * nativeLeafValueEstimate(sim);
            }
            const search::Action action = nativeHeuristicPickFast(sim);
            action.execute(sim);
        }
        if (sim.outcome != Outcome::UNDECIDED) {
            return NATIVE_W_SHAPE * nativeExpectimaxTerminalReward(sim, sim.turn);
        }
        return NATIVE_W_SHAPE * nativeLeafValueEstimate(sim);
    }

    std::uint64_t nativeRngCounterSum(const BattleContext &bc) {
        return static_cast<std::uint64_t>(bc.aiRng.counter) + static_cast<std::uint64_t>(bc.cardRandomRng.counter)
             + static_cast<std::uint64_t>(bc.miscRng.counter) + static_cast<std::uint64_t>(bc.shuffleRng.counter);
    }

    void nativeDecorrelateRng(BattleContext &bc) {
        bc.aiRng = sts::Random(bc.aiRng.nextLong());
        bc.cardRandomRng = sts::Random(bc.cardRandomRng.nextLong());
        bc.miscRng = sts::Random(bc.miscRng.nextLong());
        bc.shuffleRng = sts::Random(bc.shuffleRng.nextLong());
    }

    void nativeSeedRng(BattleContext &bc, std::uint64_t base) {
        bc.aiRng = sts::Random(base);
        bc.cardRandomRng = sts::Random(base + 1);
        bc.miscRng = sts::Random(base + 2);
        bc.shuffleRng = sts::Random(base + 3);
    }

    // Fixed-size per-monster record matching state_key_bundle's own
    // per-monster tuple field-for-field: (curHp, block, strength,
    // vulnerable, weak, halfDead, moveHistory[0], moveHistory[1], miscInfo,
    // 6 status values). A plain struct (not a flattened int array) so that
    // NativeStateKey's own equality below can never structurally collide
    // two different-shaped states (different monster counts, hand/pile
    // sizes) regardless of what the individual integers happen to be --
    // std::vector::operator== checks size before elements, the same
    // guarantee Python's nested-tuple equality already gave state_key_bundle.
    struct NativeMonsterKey {
        int curHp, block, strength, vulnerable, weak, halfDead;
        int moveHistory0, moveHistory1, miscInfo;
        std::array<int, 6> statuses;
        bool operator==(const NativeMonsterKey &o) const {
            return curHp == o.curHp && block == o.block && strength == o.strength
                && vulnerable == o.vulnerable && weak == o.weak && halfDead == o.halfDead
                && moveHistory0 == o.moveHistory0 && moveHistory1 == o.moveHistory1
                && miscInfo == o.miscInfo && statuses == o.statuses;
        }
    };

    struct NativeStateKey {
        int playerHp = 0, playerBlock = 0, playerEnergy = 0, turn = 0;
        std::array<int, 19> pStatuses{};
        std::vector<NativeMonsterKey> monsters;
        std::vector<int> hand;
        std::vector<int> draw;
        std::vector<int> discard;

        bool operator==(const NativeStateKey &o) const {
            return playerHp == o.playerHp && playerBlock == o.playerBlock
                && playerEnergy == o.playerEnergy && turn == o.turn
                && pStatuses == o.pStatuses && monsters == o.monsters
                && hand == o.hand && draw == o.draw && discard == o.discard;
        }
    };

    struct NativeStateKeyHash {
        static void mix(std::size_t &h, std::size_t v) {
            h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        }
        std::size_t operator()(const NativeStateKey &k) const {
            std::size_t h = 0;
            mix(h, std::hash<int>()(k.playerHp));
            mix(h, std::hash<int>()(k.playerBlock));
            mix(h, std::hash<int>()(k.playerEnergy));
            mix(h, std::hash<int>()(k.turn));
            for (int v : k.pStatuses) mix(h, std::hash<int>()(v));
            for (const auto &m : k.monsters) {
                mix(h, std::hash<int>()(m.curHp));
                mix(h, std::hash<int>()(m.block));
                mix(h, std::hash<int>()(m.strength));
                mix(h, std::hash<int>()(m.vulnerable));
                mix(h, std::hash<int>()(m.weak));
                mix(h, std::hash<int>()(m.halfDead));
                mix(h, std::hash<int>()(m.moveHistory0));
                mix(h, std::hash<int>()(m.moveHistory1));
                mix(h, std::hash<int>()(m.miscInfo));
                for (int s : m.statuses) mix(h, std::hash<int>()(s));
            }
            mix(h, std::hash<std::size_t>()(k.hand.size()));
            for (int v : k.hand) mix(h, std::hash<int>()(v));
            mix(h, std::hash<std::size_t>()(k.draw.size()));
            for (int v : k.draw) mix(h, std::hash<int>()(v));
            mix(h, std::hash<std::size_t>()(k.discard.size()));
            for (int v : k.discard) mix(h, std::hash<int>()(v));
            return h;
        }
    };

    NativeStateKey nativeStateKey(const BattleContext &bc) {
        static const PlayerStatus PLAYER_STATUS_IDS[] = {
            PlayerStatus::ARTIFACT, PlayerStatus::BARRICADE, PlayerStatus::METALLICIZE,
            PlayerStatus::RITUAL, PlayerStatus::RAGE, PlayerStatus::RUPTURE,
            PlayerStatus::COMBUST, PlayerStatus::DEMON_FORM, PlayerStatus::DARK_EMBRACE,
            PlayerStatus::EVOLVE, PlayerStatus::FEEL_NO_PAIN, PlayerStatus::FIRE_BREATHING,
            PlayerStatus::JUGGERNAUT, PlayerStatus::PANACHE, PlayerStatus::ENVENOM,
            PlayerStatus::FLAME_BARRIER, PlayerStatus::BRUTALITY, PlayerStatus::REGEN,
            PlayerStatus::CORRUPTION,
        };
        NativeStateKey key;
        key.playerHp = bc.player.curHp;
        key.playerBlock = bc.player.block;
        key.playerEnergy = bc.player.energy;
        key.turn = bc.turn;
        for (int i = 0; i < 19; ++i) {
            key.pStatuses[i] = bc.player.getStatusRuntime(PLAYER_STATUS_IDS[i]);
        }
        key.monsters.reserve(bc.monsters.monsterCount);
        for (int i = 0; i < bc.monsters.monsterCount; ++i) {
            const Monster &m = bc.monsters.arr[i];
            NativeMonsterKey mk;
            mk.curHp = m.curHp;
            mk.block = m.block;
            mk.strength = m.strength;
            mk.vulnerable = m.vulnerable;
            mk.weak = m.weak;
            mk.halfDead = m.halfDead;
            mk.moveHistory0 = static_cast<int>(m.moveHistory[0]);
            mk.moveHistory1 = static_cast<int>(m.moveHistory[1]);
            mk.miscInfo = m.miscInfo;
            mk.statuses = {
                m.getStatus<MS::POISON>(), m.getStatus<MS::PLATED_ARMOR>(),
                m.getStatus<MS::ARTIFACT>(), m.getStatus<MS::METALLICIZE>(),
                m.getStatus<MS::MODE_SHIFT>(), m.getStatus<MS::TIME_WARP>(),
            };
            key.monsters.push_back(mk);
        }
        key.hand.reserve(bc.cards.cardsInHand);
        for (int i = 0; i < bc.cards.cardsInHand; ++i) {
            key.hand.push_back(static_cast<int>(bc.cards.hand[i].id));
        }
        std::sort(key.hand.begin(), key.hand.end());
        key.draw.reserve(bc.cards.drawPile.size());
        for (const auto &c : bc.cards.drawPile) {
            key.draw.push_back(static_cast<int>(c.id));
        }
        key.discard.reserve(bc.cards.discardPile.size());
        for (const auto &c : bc.cards.discardPile) {
            key.discard.push_back(static_cast<int>(c.id));
        }
        std::sort(key.discard.begin(), key.discard.end());
        return key;
    }

    struct MctsNode {
        BattleContext bc;
        std::vector<search::Action> actions;
        std::vector<std::int64_t> N;
        std::vector<double> W;
        std::vector<int> visitOrder;  // indices into actions/N/W, descending heuristic-score order -- see nativeHeuristicVisitOrder
        std::vector<double> priors;  // softmax-over-heuristic-score PUCT prior, same index space as actions/N/W -- see nativeHeuristicPriors
        // Flat, index-sized-to-actions.size() storage instead of unordered_map<int, ...> --
        // idx is always a small dense integer (0..actions.size()-1, typically <15), so a hash
        // map here was pure overhead (hash + bucket lookup) versus direct array indexing for
        // every simulate() call. Sized and zero/null-initialized in nativeExpandLeaf, alongside
        // N/W/visitOrder.
        std::vector<MctsNode *> children;
        std::vector<std::vector<MctsNode *>> chanceChildren;
        std::vector<int> chanceSamplesDrawn;
        // RAVE / AMAF ("all moves as first") statistics, same index space as actions/N/W. Where
        // N[i]/W[i] count only simulations that took action i AT THIS NODE, amafN[i]/amafW[i]
        // count every simulation in which action i was played ANYWHERE at or below this node --
        // including deep in the rollout. One simulation therefore updates many actions' AMAF
        // stats instead of just the one on the path, which is what makes value estimates converge
        // far faster per simulation at the small budgets this project runs. Only populated when
        // g_useRave is on; see nativeSelectIdx for how they are blended in.
        std::vector<std::int64_t> amafN;
        std::vector<double> amafW;
        bool hasKey = false;
        NativeStateKey key;
        bool isTerminal = false;
        double terminalValue = 0.0;
        int visitCount = 0;
        bool expanded = false;

        explicit MctsNode(BattleContext bcIn) : bc(std::move(bcIn)) {}
    };

    // All nodes for one choose_action call are owned here, scoped to that
    // one call's lifetime -- mirrors the Python original exactly ("a fresh
    // tree every decision", no cross-call persistence anywhere). Raw
    // MctsNode* pointers are handed out and stored freely in
    // children/chanceChildren/the transposition table because everything
    // referencing them dies together when this arena (a local variable in
    // nativeRunMctsSearch) goes out of scope.
    class MctsArena {
        static constexpr std::size_t NODES_PER_BLOCK = 32;
        struct Block {
            using Storage = std::aligned_storage_t<
                sizeof(MctsNode), alignof(MctsNode)>;
            std::array<Storage, NODES_PER_BLOCK> storage;
            std::size_t size = 0;

            ~Block() {
                for (std::size_t i = 0; i < size; ++i) {
                    std::launder(reinterpret_cast<MctsNode *>(
                        &storage[i]))->~MctsNode();
                }
            }

            MctsNode *emplace(BattleContext &&bc) {
                void *slot = &storage[size];
                MctsNode *node = new (slot) MctsNode(std::move(bc));
                ++size;
                return node;
            }
        };

    public:
        MctsNode *newNode(BattleContext bc) {
            if (blocks_.empty()
                    || blocks_.back()->size == NODES_PER_BLOCK) {
                blocks_.push_back(std::make_unique<Block>());
            }
            return blocks_.back()->emplace(std::move(bc));
        }
    private:
        std::vector<std::unique_ptr<Block>> blocks_;
    };

    int nativeSelectIdx(const MctsNode &node) {
        const int n = static_cast<int>(node.N.size());
        // Cold-start phase: every edge needs one mandatory first visit
        // before UCB1 differentiation means anything. Try them in
        // heuristic-score order (node.visitOrder), not raw enumeration
        // order -- see nativeHeuristicVisitOrder's own comment.
        for (int idx : node.visitOrder) {
            if (node.N[idx] == 0) {
                return idx;
            }
        }
        std::int64_t total = 0;
        for (int i = 0; i < n; ++i) {
            total += node.N[i];
        }
        const double logParent = std::log(static_cast<double>(total) + 1.0);
        const double sqrtTotal = std::sqrt(static_cast<double>(total) + 1.0);
        int bestIdx = 0;
        double bestScore = -std::numeric_limits<double>::infinity();
        for (int i = 0; i < n; ++i) {
            const double ni = static_cast<double>(node.N[i]);
            // Every idx reaching this loop has N[i] > 0 (unvisited edges
            // return above), so a chance-classified action already has its
            // chanceChildren entry -- see g_params.cUcbChance's own comment.
            const bool destIsChance = node.actions[i].getActionType() == search::ActionType::END_TURN
                                    || !node.chanceChildren[i].empty();
            const double cUcb = destIsChance ? g_params.cUcbChance : g_params.cUcb;
            // PUCT term added ON TOP of the existing UCB1 exploration term (not a replacement --
            // see g_params.cPuct's own comment). Standard PUCT shape: prior probability times
            // sqrt(parent visits), decaying as this edge's own visit count grows. g_params.cPuct
            // defaults to 0.0, so this term vanishes entirely until explicitly tuned on.
            const double puctTerm = g_params.cPuct * node.priors[i] * sqrtTotal / (1.0 + ni);
            // Exploitation term: normally this edge's own mean value, but with RAVE on it is
            // blended with the AMAF mean via the Gelly/Silver MC-RAVE schedule. beta starts near 1
            // (AMAF dominates, since it has far more samples early) and decays toward 0 as this
            // edge accumulates real visits of its own, so the biased-but-plentiful AMAF estimate
            // is used exactly while the unbiased-but-scarce direct estimate is too noisy to act on.
            double exploit = node.W[i] / ni;
            if (g_useRave && !node.amafN.empty() && node.amafN[i] > 0) {
                const double an = static_cast<double>(node.amafN[i]);
                const double b = g_params.raveBias;
                const double beta = an / (ni + an + 4.0 * ni * an * b * b);
                exploit = (1.0 - beta) * exploit + beta * (node.amafW[i] / an);
            }
            const double score = exploit + cUcb * std::sqrt(logParent / (ni + 1.0)) + puctTerm;
            if (score > bestScore) {
                bestScore = score;
                bestIdx = i;
            }
        }
        return bestIdx;
    }

    void nativeDedupActions(const BattleContext &bc, std::vector<search::Action> &actions) {
        if (g_params.mergeDuplicateActions == 0.0 || actions.size() < 2) {
            return;
        }
        // Packed identity key. CARD: every CardInstance field except uniqueId,
        // plus the target -- twins identical in all of those reach identical
        // successor states up to uniqueId bookkeeping. POTION: potion id and
        // target. Card-select/scry actions are positional into piles and are
        // deliberately left alone.
        std::vector<std::uint64_t> seen;
        seen.reserve(actions.size());
        std::vector<search::Action> kept;
        kept.reserve(actions.size());
        for (const auto &action : actions) {
            std::uint64_t key;
            const auto type = action.getActionType();
            if (type == search::ActionType::CARD) {
                const CardInstance &c = bc.cards.hand[action.getSourceIdx()];
                key = (static_cast<std::uint64_t>(1) << 60)
                    ^ (static_cast<std::uint64_t>(static_cast<std::uint16_t>(c.id)) << 44)
                    ^ (static_cast<std::uint64_t>(static_cast<std::uint16_t>(c.specialData)) << 28)
                    ^ (static_cast<std::uint64_t>(static_cast<std::uint8_t>(c.cost)) << 20)
                    ^ (static_cast<std::uint64_t>(static_cast<std::uint8_t>(c.costForTurn)) << 12)
                    ^ (static_cast<std::uint64_t>(c.upgraded) << 11)
                    ^ (static_cast<std::uint64_t>(c.freeToPlayOnce) << 10)
                    ^ (static_cast<std::uint64_t>(c.retain) << 9)
                    ^ static_cast<std::uint64_t>(action.getTargetIdx() & 0xF);
            } else if (type == search::ActionType::POTION) {
                key = (static_cast<std::uint64_t>(2) << 60)
                    ^ (static_cast<std::uint64_t>(static_cast<std::uint16_t>(
                           bc.potions[action.getSourceIdx()])) << 8)
                    ^ static_cast<std::uint64_t>(action.getTargetIdx() & 0xF);
            } else {
                kept.push_back(action);
                continue;
            }
            if (std::find(seen.begin(), seen.end(), key) == seen.end()) {
                seen.push_back(key);
                kept.push_back(action);
            }
        }
        actions = std::move(kept);
    }

    double nativeExpandLeaf(MctsNode *node, int maxTurn, std::vector<std::uint32_t> *raveTrace = nullptr) {
        node->actions = sts::py::getLegalActions(node->bc);
        nativeDedupActions(node->bc, node->actions);
        node->N.assign(node->actions.size(), 0);
        node->W.assign(node->actions.size(), 0.0);
        node->children.assign(node->actions.size(), nullptr);
        node->chanceChildren.assign(node->actions.size(), {});
        node->chanceSamplesDrawn.assign(node->actions.size(), 0);
        {
            const std::vector<double> scores = nativeHeuristicScores(node->bc, node->actions);
            node->visitOrder = nativeHeuristicVisitOrder(scores);
            node->priors = nativeHeuristicPriors(scores);
        }
        if (g_useRave) {
            node->amafN.assign(node->actions.size(), 0);
            node->amafW.assign(node->actions.size(), 0.0);
        }
        node->expanded = true;
        return nativeLeafValue(node->bc, maxTurn, raveTrace);
    }

    // Root-candidate pairing state. Single-threaded search, set by the
    // sequential-halving loop before each descent: pairIndex is the visited
    // count of the candidate about to be simulated, so the k-th simulation of
    // every candidate shares one determinization stream. -1 = descent is not
    // attributable to one root candidate (plain MCTS path), pairing inert.
    std::uint64_t g_pairSeedBase = 0;
    int g_pairIndex = -1;
    // Diagnostics for choosing escalationQgap: the top-2 survivor value gap and
    // the escalation verdicts of the most recent root search.
    double g_lastRootValueGap = -1.0;
    bool g_lastSearchDangerous = false;
    bool g_lastSearchEscalated = false;
    // Chance-sibling merge telemetry (merge_chance_outcomes): samples drawn
    // and dedup hits across the lifetime of the process. A "hit" is a DPW
    // sample that reached an information set an existing sibling already
    // represents; the visit is routed there and the sample is discarded.
    std::int64_t g_chanceMergeSamples = 0;
    std::int64_t g_chanceMergeHits = 0;

    bool nativePairingActive() {
        return g_params.pairedDeterminization != 0.0 && g_pairIndex >= 0
            && nativeHonestDrawOrder();
    }

    std::uint64_t nativePairSeed(int turn, int localSampleIndex) {
        std::size_t h = std::hash<std::uint64_t>()(g_pairSeedBase);
        NativeStateKeyHash::mix(h, std::hash<int>()(g_pairIndex));
        NativeStateKeyHash::mix(h, std::hash<int>()(turn));
        NativeStateKeyHash::mix(h, std::hash<int>()(localSampleIndex));
        return static_cast<std::uint64_t>(h);
    }

    std::size_t nativeCrnSeed(std::uint64_t crnBase, const NativeStateKey &stateKey, int localSampleIndex) {
        NativeStateKeyHash hasher;
        std::size_t h = hasher(stateKey);
        NativeStateKeyHash::mix(h, std::hash<std::uint64_t>()(crnBase));
        NativeStateKeyHash::mix(h, std::hash<int>()(localSampleIndex));
        return h;
    }

    NativeStateKey nativeInfoSetKey(const BattleContext &bc) {
        NativeStateKey key = nativeStateKey(bc);
        const int level = static_cast<int>(g_params.mergeChanceOutcomes);
        std::sort(key.draw.begin(), key.draw.end());
        if (level >= 2) {
            std::sort(key.hand.begin(), key.hand.end());
        }
        if (level >= 3) {
            std::sort(key.discard.begin(), key.discard.end());
        }
        return key;
    }

    MctsNode *nativeDpwChanceChild(MctsArena &arena, MctsNode *node, int idx, const search::Action &action,
                                    bool useCrn, std::uint64_t crnBase, std::mt19937_64 &rng) {
        auto &siblings = node->chanceChildren[idx];
        const std::int64_t n = node->N[idx];
        const double wc = (nativeHonestDrawOrder() && g_params.honestWcChance >= 0.0)
            ? g_params.honestWcChance : g_params.wcChance;
        const double wa = (nativeHonestDrawOrder() && g_params.honestWaChance >= 0.0)
            ? g_params.honestWaChance : g_params.waChance;
        const int wideningCap = static_cast<int>(std::ceil(wc * std::pow(static_cast<double>(n + 1), wa)));
        if (static_cast<int>(siblings.size()) < wideningCap) {
            const int localSampleIndex = node->chanceSamplesDrawn[idx];
            node->chanceSamplesDrawn[idx] = localSampleIndex + 1;

            std::uint64_t shuffleSeed = 0;
            if (useCrn) {
                const std::size_t seed = nativeCrnSeed(crnBase, nativeStateKey(node->bc), localSampleIndex);
                nativeSeedRng(node->bc, static_cast<std::uint64_t>(seed));
                shuffleSeed = static_cast<std::uint64_t>(seed);
            } else {
                nativeDecorrelateRng(node->bc);
                shuffleSeed = rng();
            }
            // Keyed by candidate-visit rather than by state, so sibling root
            // candidates draw correlated orders. Deliberately NOT keyed on the
            // state: keying on the state is what makes candidates independent.
            if (nativePairingActive()) {
                shuffleSeed = nativePairSeed(node->bc.turn, localSampleIndex);
            }
            BattleContext sample(node->bc);
            // Each DPW sample of this chance outcome gets its OWN draw order, so widening
            // covers draw orders alongside every other stochastic outcome and the tree
            // averages over them in place. Derived from the same seed the rest of this
            // sample uses, so paired comparisons keep their common random numbers.
            std::mt19937_64 shuffleRng(shuffleSeed);
            if (nativeHonestDrawOrder() && !nativeActionBindsPileOrder(action)) {
                nativeShuffleDrawPile(sample, shuffleRng);
            }
            const std::size_t drawBeforeSample = sample.cards.drawPile.size();
            action.execute(sample);
            // Lazy: the cards this action drew are now known, but what remains
            // must not be. Re-permuting makes the NEXT draw independent of this
            // sample's order, which is the whole difference between an ordered
            // pile and a canonical one.
            if (g_params.honestDrawOrder >= 2.0
                && sample.cards.drawPile.size() < drawBeforeSample) {
                nativeShuffleDrawPile(sample, shuffleRng);
            }

            MctsNode *child = nullptr;
            const bool mergeSiblings = g_useStateMerging
                || g_params.mergeChanceOutcomes != 0.0;
            if (mergeSiblings) {
                ++g_chanceMergeSamples;
                const NativeStateKey sampleKey =
                    g_params.mergeChanceOutcomes != 0.0
                        ? nativeInfoSetKey(sample) : nativeStateKey(sample);
                for (MctsNode *existing : siblings) {
                    if (existing->hasKey && existing->key == sampleKey) {
                        child = existing;
                        ++g_chanceMergeHits;
                        break;
                    }
                }
            }
            if (child == nullptr) {
                child = arena.newNode(std::move(sample));
                if (mergeSiblings) {
                    child->hasKey = true;
                    child->key = g_params.mergeChanceOutcomes != 0.0
                        ? nativeInfoSetKey(child->bc)
                        : nativeStateKey(child->bc);
                }
                if (child->bc.outcome != Outcome::UNDECIDED) {
                    child->isTerminal = true;
                    child->terminalValue = NATIVE_W_SHAPE * nativeExpectimaxTerminalReward(child->bc, child->bc.turn);
                }
                siblings.push_back(child);
            }
            return child;
        }
        double totalWeight = 0.0;
        for (MctsNode *c : siblings) {
            totalWeight += static_cast<double>(c->visitCount) + 1.0;
        }
        std::uniform_real_distribution<double> dist(0.0, totalWeight);
        double draw = dist(rng);
        for (MctsNode *child : siblings) {
            draw -= static_cast<double>(child->visitCount) + 1.0;
            if (draw <= 0.0) {
                return child;
            }
        }
        return siblings.back();
    }

    // forcedIdx: when >= 0, THIS node takes that action index instead of consulting
    // nativeSelectIdx. Only ever passed at the root, by the sequential-halving driver
    // (nativeRunMctsSearchSeqHalving), which owns root-level budget allocation itself; recursive
    // calls always pass -1 so every deeper level uses normal UCB1 selection.
    double nativeSimulate(MctsArena &arena, MctsNode *node,
                           std::unordered_map<NativeStateKey, MctsNode *, NativeStateKeyHash> &transTable,
                           int callDepth, bool useCrn, std::uint64_t crnBase, std::mt19937_64 &rng,
                           int maxTurn, int forcedIdx = -1,
                           std::vector<std::uint32_t> *raveTrace = nullptr) {
        node->visitCount += 1;
        if (node->isTerminal) {
            return node->terminalValue;
        }
        // See NATIVE_MAX_TURNS_PER_SEARCH's own comment: this turn-based cutoff is
        // independent of callDepth (which caps ACTIONS, not turns) -- it's what
        // actually bounds a branch that keeps making progress action-wise (so
        // never hits the depth cap) but never advances past a certain point in
        // real fight-time, letting per-turn effects (e.g. Dual Wield) compound
        // across however many actions it takes to reach that many turns.
        if (callDepth >= NATIVE_MAX_CALL_DEPTH || node->bc.turn >= maxTurn) {
            return nativeLeafValue(node->bc, maxTurn);
        }
        if (!node->expanded) {
            return nativeExpandLeaf(node, maxTurn, raveTrace);
        }

        const int idx = (forcedIdx >= 0 && forcedIdx < static_cast<int>(node->actions.size()))
                ? forcedIdx : nativeSelectIdx(*node);
        // Captured BEFORE this node's own action is appended, so the AMAF window below covers
        // this action and everything played deeper, and excludes anything played above.
        const std::size_t traceStart = raveTrace != nullptr ? raveTrace->size() : 0;
        if (raveTrace != nullptr) {
            raveTrace->push_back(node->actions[idx].bits);
        }
        if (nativeMastActive()) {
            nativeMastRecord(node->bc, node->actions[idx]);
        }
        const search::Action action = node->actions[idx];
        BattleContext &bc = node->bc;
        double value;

        const bool isChanceIdx = action.getActionType() == search::ActionType::END_TURN
                                || !node->chanceChildren[idx].empty();
        if (isChanceIdx) {
            MctsNode *child = nativeDpwChanceChild(arena, node, idx, action, useCrn, crnBase, rng);
            const double r = nativeExpectimaxDenseReward(bc, child->bc);
            if (child->isTerminal) {
                value = r + child->terminalValue;
            } else {
                value = r + NATIVE_GAMMA * nativeSimulate(arena, child, transTable, callDepth + 1, useCrn, crnBase, rng, maxTurn, -1, raveTrace);
            }
        } else if (node->children[idx] != nullptr) {
            MctsNode *child = node->children[idx];
            const double r = nativeExpectimaxDenseReward(bc, child->bc);
            value = r + NATIVE_GAMMA * nativeSimulate(arena, child, transTable, callDepth + 1, useCrn, crnBase, rng, maxTurn, -1, raveTrace);
        } else {
            BattleContext childBc(bc);
            const std::uint64_t counterBefore = nativeRngCounterSum(bc);
            // Drawing from an ORDERED pile consumes no RNG -- it pops the top -- so the
            // counter probe below classifies every mid-turn draw card (Battle Trance,
            // Pommel Strike, Shrug It Off, Acrobatics, Warcry...) as deterministic, caches
            // one child, and never samples another. That is exactly how draw-order
            // clairvoyance enters the tree: only END_TURN was ever treated as a chance node.
            // With honestDrawOrder on, the pile is permuted before the action runs and any
            // action that actually DREW is reclassified as a chance node, so DPW widening
            // samples several draw orders for it and the tree averages over them.
            const std::size_t drawPileBefore = childBc.cards.drawPile.size();
            if (nativeHonestDrawOrder() && !nativeActionBindsPileOrder(action)) {
                nativeShuffleDrawPile(childBc, rng);
            }
            action.execute(childBc);
            bool consumedRng = nativeRngCounterSum(childBc) != counterBefore;
            if (nativeHonestDrawOrder() && childBc.cards.drawPile.size() < drawPileBefore) {
                consumedRng = true;
                if (g_params.honestDrawOrder >= 2.0) {
                    nativeShuffleDrawPile(childBc, rng);
                }
            }
            const double r = nativeExpectimaxDenseReward(bc, childBc);
            MctsNode *child = nullptr;
            if (consumedRng) {
                child = arena.newNode(std::move(childBc));
                if (child->bc.outcome != Outcome::UNDECIDED) {
                    child->isTerminal = true;
                    child->terminalValue = NATIVE_W_SHAPE * nativeExpectimaxTerminalReward(child->bc, child->bc.turn);
                }
                if (g_useStateMerging || g_params.mergeChanceOutcomes != 0.0) {
                    child->hasKey = true;
                    child->key = g_params.mergeChanceOutcomes != 0.0
                        ? nativeInfoSetKey(child->bc)
                        : nativeStateKey(child->bc);
                }
                node->chanceChildren[idx] = {child};
                node->chanceSamplesDrawn[idx] = 1;
            } else {
                const bool terminal = childBc.outcome != Outcome::UNDECIDED;
                if (g_useStateMerging && !terminal) {
                    NativeStateKey key = nativeStateKey(childBc);
                    auto it = transTable.find(key);
                    if (it != transTable.end()) {
                        child = it->second;
                    } else {
                        child = arena.newNode(std::move(childBc));
                        transTable.emplace(std::move(key), child);
                    }
                } else {
                    child = arena.newNode(std::move(childBc));
                }
                if (terminal) {
                    child->isTerminal = true;
                    child->terminalValue = NATIVE_W_SHAPE * nativeExpectimaxTerminalReward(child->bc, child->bc.turn);
                }
                node->children[idx] = child;
            }
            if (child->isTerminal) {
                value = r + child->terminalValue;
            } else {
                value = r + NATIVE_GAMMA * nativeSimulate(arena, child, transTable, callDepth + 1, useCrn, crnBase, rng, maxTurn, -1, raveTrace);
            }
        }

        node->N[idx] += 1;
        node->W[idx] += value;

        // Max-Monte-Carlo backup, blended -- see TunableParams::backupMaxWeight. Computed
        // AFTER this action's own statistics are updated so the max can include it, and
        // kept separate from `value`, which stays the sampled return: RAVE below credits
        // what was actually played, not what the node now believes about its best line.
        double backedUp = value;
        if (g_params.backupMaxWeight != 0.0) {
            double bestQ = -std::numeric_limits<double>::infinity();
            const int nActions = static_cast<int>(node->N.size());
            for (int i = 0; i < nActions; ++i) {
                if (node->N[i] > 0) {
                    const double q = node->W[i] / static_cast<double>(node->N[i]);
                    if (q > bestQ) {
                        bestQ = q;
                    }
                }
            }
            // bestQ is finite by construction -- idx was just visited -- but an unvisited
            // node would leave it at -inf, and silently returning that would poison the
            // whole branch rather than fail.
            if (bestQ > -std::numeric_limits<double>::infinity()) {
                backedUp = (1.0 - g_params.backupMaxWeight) * value
                    + g_params.backupMaxWeight * bestQ;
            }
        }

        // AMAF update. traceStart was captured before this node's action was taken, so
        // [traceStart, end) is exactly the set of actions played AT or BELOW this node in this
        // simulation -- the "all moves as first" set. Each of this node's legal actions appearing
        // anywhere in that window is credited with this simulation's value, so one simulation
        // informs many action estimates rather than only the one it happened to take here.
        if (raveTrace != nullptr && !node->amafN.empty()) {
            const std::size_t traceEnd = raveTrace->size();
            const int nActions = static_cast<int>(node->actions.size());
            for (int i = 0; i < nActions; ++i) {
                const std::uint32_t bits = node->actions[i].bits;
                for (std::size_t t = traceStart; t < traceEnd; ++t) {
                    if ((*raveTrace)[t] == bits) {
                        node->amafN[i] += 1;
                        node->amafW[i] += value;
                        break;  // credit once per simulation, not once per repetition
                    }
                }
            }
        }
        return backedUp;
    }

    // Sequential halving over ROOT actions (Karnin/Koren/Somekh; the allocation rule Gumbel
    // MuZero also uses at the root). Motivation: at this project's real budget -- 150 sims over a
    // typical 5-8 root actions -- plain UCB1 keeps re-sampling actions it has already established
    // are bad, because the exploration term never fully stops pulling them. Sequential halving
    // instead splits the budget into ceil(log2(A)) phases and drops the worst half of the
    // surviving candidates after each, so late phases spend the remaining budget only on genuine
    // contenders. This targets exactly the small-budget regime; it is not a port of anything in
    // Silver Automaton (their root uses ordinary UCT with a tuned exploration constant).
    //
    // Only ROOT allocation changes. Every level below the root still descends by UCB1 through the
    // unchanged nativeSimulate, and the value backed up is unchanged, so this composes with the
    // existing DPW/chance-node/transposition machinery rather than replacing any of it.
    //
    // Final pick is by mean value among survivors rather than by visit count: visit counts here
    // are an artifact of the phase schedule (every survivor is deliberately given equal pulls
    // within a phase), so they carry no preference information the way UCB1 visit counts do.
    // Tree-reuse state. Single-threaded search: the arena outlives calls only
    // so the previous root's subtrees stay valid; everything is discarded the
    // moment a match fails.
    std::unique_ptr<MctsArena> g_reuseArena;
    MctsNode *g_reusePrevRoot = nullptr;
    int g_reusePrevChosen = -1;
    std::int64_t g_reuseHits = 0, g_reuseMisses = 0;

    void nativeResetSearchTree() {
        g_reuseArena.reset();
        g_reusePrevRoot = nullptr;
        g_reusePrevChosen = -1;
    }

    // Identity signature for grafting root actions across two orderings of
    // the same information set -- same encoding as nativeDedupActions.
    std::uint64_t nativeActionIdentity(const BattleContext &bc,
                                       const search::Action &action) {
        const auto type = action.getActionType();
        if (type == search::ActionType::CARD) {
            const CardInstance &c = bc.cards.hand[action.getSourceIdx()];
            return (static_cast<std::uint64_t>(1) << 60)
                ^ (static_cast<std::uint64_t>(static_cast<std::uint16_t>(c.id)) << 44)
                ^ (static_cast<std::uint64_t>(static_cast<std::uint16_t>(c.specialData)) << 28)
                ^ (static_cast<std::uint64_t>(static_cast<std::uint8_t>(c.cost)) << 20)
                ^ (static_cast<std::uint64_t>(static_cast<std::uint8_t>(c.costForTurn)) << 12)
                ^ (static_cast<std::uint64_t>(c.upgraded) << 11)
                ^ (static_cast<std::uint64_t>(c.freeToPlayOnce) << 10)
                ^ (static_cast<std::uint64_t>(c.retain) << 9)
                ^ static_cast<std::uint64_t>(action.getTargetIdx() & 0xF);
        }
        if (type == search::ActionType::POTION) {
            return (static_cast<std::uint64_t>(2) << 60)
                ^ (static_cast<std::uint64_t>(static_cast<std::uint16_t>(
                       bc.potions[action.getSourceIdx()])) << 8)
                ^ static_cast<std::uint64_t>(action.getTargetIdx() & 0xF);
        }
        if (type == search::ActionType::END_TURN) {
            return (static_cast<std::uint64_t>(3) << 60);
        }
        return 0;  // card-select / scry: positional, never grafted
    }

    MctsNode *nativeFindReuseRoot(const BattleContext &bc) {
        if (g_reusePrevRoot == nullptr || g_reusePrevChosen < 0
            || g_reusePrevChosen >= static_cast<int>(g_reusePrevRoot->actions.size())) {
            return nullptr;
        }
        // The played action's resulting real state, against the stored
        // deterministic child and every chance sample. Key comparison uses the
        // info-set key under honest draws -- the real pile order is exactly
        // the thing the honest tree never conditioned on.
        const NativeStateKey want = nativeHonestDrawOrder()
            ? nativeInfoSetKey(bc) : nativeStateKey(bc);
        MctsNode *det = g_reusePrevRoot->children[g_reusePrevChosen];
        if (det != nullptr) {
            const NativeStateKey got = nativeHonestDrawOrder()
                ? nativeInfoSetKey(det->bc) : nativeStateKey(det->bc);
            if (got == want) {
                return det;
            }
        }
        for (MctsNode *cand : g_reusePrevRoot->chanceChildren[g_reusePrevChosen]) {
            const NativeStateKey got = nativeHonestDrawOrder()
                ? nativeInfoSetKey(cand->bc) : nativeStateKey(cand->bc);
            if (got == want) {
                return cand;
            }
        }
        return nullptr;
    }

    std::pair<search::Action, std::vector<std::int64_t>> nativeRunMctsSearchSeqHalving(
            const BattleContext &bc, int nSimulations, bool useCrn, std::uint64_t crnBase,
            bool useSearchSeed, std::uint64_t searchSeed) {
        // The rollout policy samples (Gumbel-max) whenever rolloutTemperature > 0, so
        // its RNG has to be tied to this call's seed or the search is not reproducible
        // and sibling candidates lose common random numbers.
        nativeSeedGumbel(useSearchSeed ? searchSeed : std::random_device{}());
        g_rootMaxHp = bc.player.maxHp;
        g_rootGold = bc.player.gold;
        const bool reuse = g_params.treeReuse != 0.0;
        MctsArena localArena;
        MctsNode *reused = nullptr;
        if (reuse) {
            reused = nativeFindReuseRoot(bc);
            if (reused != nullptr) {
                ++g_reuseHits;
            } else {
                ++g_reuseMisses;
                // Miss (or battle boundary): everything stored is stale.
                g_reuseArena.reset();
                g_reusePrevRoot = nullptr;
                g_reusePrevChosen = -1;
            }
            if (!g_reuseArena) {
                g_reuseArena = std::make_unique<MctsArena>();
            }
        } else if (g_reuseArena) {
            nativeResetSearchTree();
        }
        MctsArena &arena = reuse ? *g_reuseArena : localArena;
        MctsNode *root = arena.newNode(BattleContext(bc));
        std::unordered_map<NativeStateKey, MctsNode *, NativeStateKeyHash> transTable;
        std::random_device rd;
        std::mt19937_64 rng(useSearchSeed ? searchSeed : rd());
        // One determinization-pairing base per SEARCH CALL: candidates inside
        // this call share it (that is the pairing), successive decisions do
        // not (that would replay the same futures all fight).
        g_pairSeedBase = useCrn ? crnBase
            : (useSearchSeed ? searchSeed : rng());
        g_pairIndex = -1;
        const int maxTurn = bc.turn + static_cast<int>(g_params.searchMaxTurns);

        // Initialize the root so its action list exists before allocation decisions are made.
        // Calling nativeExpandLeaf here would run and discard one full rollout.
        root->actions = sts::py::getLegalActions(root->bc);
        nativeDedupActions(root->bc, root->actions);
        root->N.assign(root->actions.size(), 0);
        root->W.assign(root->actions.size(), 0.0);
        root->children.assign(root->actions.size(), nullptr);
        root->chanceChildren.assign(root->actions.size(), {});
        root->chanceSamplesDrawn.assign(root->actions.size(), 0);
        if (reused != nullptr && reused->expanded) {
            // Graft the matched subtrees onto the fresh root by action
            // IDENTITY. The fresh root's indices bind to the real state; the
            // stored subtrees keep their own internally consistent worlds.
            // Duplicate identities (two identical Strikes) graft first-come,
            // which loses nothing: post-dedup the fresh list has no twins.
            for (std::size_t fi = 0; fi < root->actions.size(); ++fi) {
                const std::uint64_t wantId =
                    nativeActionIdentity(root->bc, root->actions[fi]);
                if (wantId == 0) {
                    continue;
                }
                for (std::size_t si = 0; si < reused->actions.size(); ++si) {
                    if (nativeActionIdentity(reused->bc, reused->actions[si])
                            != wantId) {
                        continue;
                    }
                    root->children[fi] = reused->children[si];
                    root->chanceChildren[fi] = reused->chanceChildren[si];
                    root->chanceSamplesDrawn[fi] = reused->chanceSamplesDrawn[si];
                    break;
                }
            }
        }
        {
            const std::vector<double> scores = nativeHeuristicScores(root->bc, root->actions);
            root->visitOrder = nativeHeuristicVisitOrder(scores);
            root->priors = nativeHeuristicPriors(scores);
        }
        if (g_useRave) {
            root->amafN.assign(root->actions.size(), 0);
            root->amafW.assign(root->actions.size(), 0.0);
        }
        root->expanded = true;
        std::vector<std::uint32_t> raveTrace;
        // Per-search, not per-fight: MAST's whole value is that it reflects the state
        // distribution of THIS decision, and carrying a table across decisions would make
        // it a stale global prior of the kind cardPickRateWeight already provides.
        nativeMastReset();
        const int nActions = static_cast<int>(root->actions.size());
        if (nActions == 0) {
            return {search::Action(search::ActionType::END_TURN), {}};
        }
        if (nActions == 1) {
            return {root->actions[0], std::vector<std::int64_t>(root->N.begin(), root->N.end())};
        }

        // Root candidate set. With seqHalvingCandidates set, this is Gumbel-Top-k: adding
        // independent Gumbel noise to log-priors and taking the top m draws exactly m
        // actions WITHOUT replacement from the prior (Danihelka et al., "Policy Improvement
        // by Planning with Gumbel"), which is how that work spends a small budget.
        //
        // Halving over EVERY action -- what this did before, and what the 0 default still
        // does -- spends its first phase eliminating half the candidates on a couple of
        // simulations each. At 100 simulations and 12 legal actions the schedule is
        // 2/4/8/14 per arm, so 24% of the budget buys two samples per candidate before the
        // first cut. At m = 4 the same budget gives 12 then 26, and the first elimination
        // rests on 6x the evidence. This is the cost the visitOrder comment upstream
        // describes ("burns 10+ simulations trying every one once"), addressed by not
        // visiting most of them rather than by ordering them.
        //
        // Note for callers reading the returned visit vector: actions outside the candidate
        // set keep N = 0 because they were never searched. That is honest, but it means the
        // visit distribution is no longer a ranking over all legal actions. The production
        // path already copes -- expectimax_search.choose_action_native discards visit totals
        // under seq halving and emits a one-hot on the chosen action, on the grounds that
        // they are "a budget-allocation artifact", which this only makes more true.
        // expectimax_search.root_parallel_search does NOT: it sums visit vectors across
        // trees and takes the argmax, and with a candidate cap each tree samples a DIFFERENT
        // subset, so the sum mixes arms that were never compared against each other. It is
        // used only by the legacy combat-distillation scripts (distillation.py,
        // train_distillation_expectimax.py), not by the whole-run pipeline -- but do not
        // combine a candidate cap with root-parallel search without fixing that aggregation.
        std::vector<int> survivors;
        const int candidateCap = static_cast<int>(g_params.seqHalvingCandidates);
        const bool havePriors = static_cast<int>(root->priors.size()) == nActions;
        if (candidateCap <= 0 || candidateCap >= nActions) {
            survivors.resize(nActions);
            for (int i = 0; i < nActions; ++i) {
                survivors[i] = i;
            }
        } else {
            std::vector<std::pair<double, int>> keys;
            keys.reserve(nActions);
            for (int i = 0; i < nActions; ++i) {
                const double prior = havePriors ? root->priors[i] : 1.0 / nActions;
                keys.emplace_back(std::log(std::max(prior, 1e-12)) + nativeGumbelNoise(), i);
            }
            // The heuristic's own best action is forced in rather than sampled. Gumbel-Top-k
            // is a sample, and these priors come from a hand-tuned heuristic rather than a
            // trained net, so the strongest candidate can genuinely be missed -- and unlike
            // the training-target setting the trick was designed for, nothing downstream
            // recovers when it is. Costs one slot and removes the failure mode.
            if (havePriors) {
                int best = 0;
                for (int i = 1; i < nActions; ++i) {
                    if (root->priors[i] > root->priors[best]) {
                        best = i;
                    }
                }
                keys[best].first = std::numeric_limits<double>::infinity();
            }
            std::partial_sort(
                keys.begin(), keys.begin() + candidateCap, keys.end(),
                [](const std::pair<double, int> &a, const std::pair<double, int> &b) {
                    return a.first > b.first;
                });
            survivors.reserve(candidateCap);
            for (int i = 0; i < candidateCap; ++i) {
                survivors.push_back(keys[i].second);
            }
        }

        int used = 0;
        while (survivors.size() > 1 && used < nSimulations) {
            const int remaining = nSimulations - used;
            // Budget for THIS phase, split evenly across survivors. Uses the phases still to run
            // rather than the original count, so an early phase can't starve the later ones.
            int phasesLeft = 1;
            {
                std::size_t s = survivors.size();
                phasesLeft = 0;
                while (s > 1) { s = (s + 1) / 2; ++phasesLeft; }
                if (phasesLeft < 1) phasesLeft = 1;
            }
            int perAction = remaining / (static_cast<int>(survivors.size()) * phasesLeft);
            if (perAction < 1) {
                perAction = 1;
            }
            for (int idx : survivors) {
                for (int j = 0; j < perAction && used < nSimulations; ++j) {
                    g_pairIndex = static_cast<int>(root->N[idx]);
                    raveTrace.clear();
                    g_mastTrace.clear();
                    const double value = nativeSimulate(
                        arena, root, transTable, 0, useCrn, crnBase, rng, maxTurn, idx,
                        g_useRave ? &raveTrace : nullptr);
                    if (nativeMastActive()) {
                        nativeMastUpdate(value);
                    }
                    ++used;
                }
            }
            // Keep the better-scoring half by mean value. Unvisited entries sort last.
            std::sort(survivors.begin(), survivors.end(), [&](int a, int b) {
                const double qa = root->N[a] > 0 ? root->W[a] / root->N[a] : -std::numeric_limits<double>::infinity();
                const double qb = root->N[b] > 0 ? root->W[b] / root->N[b] : -std::numeric_limits<double>::infinity();
                return qa > qb;
            });
            survivors.resize((survivors.size() + 1) / 2);
        }

        // Spend any leftover budget on the finalists, then pick the best mean value.
        while (used < nSimulations) {
            for (int idx : survivors) {
                if (used >= nSimulations) break;
                g_pairIndex = static_cast<int>(root->N[idx]);
                raveTrace.clear();
                g_mastTrace.clear();
                const double value = nativeSimulate(
                    arena, root, transTable, 0, useCrn, crnBase, rng, maxTurn, idx,
                    g_useRave ? &raveTrace : nullptr);
                if (nativeMastActive()) {
                    nativeMastUpdate(value);
                }
                ++used;
            }
        }
        g_pairIndex = -1;
        // Top-2 survivor gap, recorded every search (diagnostic), acted on when
        // escalation is enabled and this call is still the base-budget stage.
        {
            // Over every VISITED root action, not `survivors` -- the halving
            // loop has already resized survivors down to one by this point, so
            // the runner-up lives in the eliminated set. The gap between the
            // last survivor and the best eliminated candidate is exactly how
            // contested the final cut was.
            double q1 = -std::numeric_limits<double>::infinity();
            double q2 = -std::numeric_limits<double>::infinity();
            int visited = 0;
            for (std::size_t idx = 0; idx < root->actions.size(); ++idx) {
                if (root->N[idx] <= 0) {
                    continue;
                }
                ++visited;
                const double q = root->W[idx] / root->N[idx];
                if (q > q1) { q2 = q1; q1 = q; }
                else if (q > q2) { q2 = q; }
            }
            g_lastRootValueGap = visited >= 2 ? q1 - q2 : -1.0;
        }
        if (g_params.escalationSims > 0.0
            && nSimulations < static_cast<int>(g_params.escalationSims)) {
            const HeuristicContext ctx = nativeComputeHeuristicContext(bc);
            bool heartPresent = false;
            for (int i = 0; i < bc.monsters.monsterCount; ++i) {
                if (bc.monsters.arr[i].id == MonsterId::CORRUPT_HEART
                    && bc.monsters.arr[i].curHp > 0) {
                    heartPresent = true;
                }
            }
            const bool dangerous = heartPresent
                || ctx.unblocked >= g_params.escalationDangerFrac
                    * std::max(1, static_cast<int>(bc.player.curHp));
            const bool contested = g_lastRootValueGap >= 0.0
                && g_lastRootValueGap < g_params.escalationQgap;
            g_lastSearchDangerous = dangerous;
            g_lastSearchEscalated = dangerous && contested;
            if (g_lastSearchEscalated) {
                // Fresh full search at the big budget -- exactly the shape the
                // savable-death probe validated. The recursive call fails this
                // nSimulations guard, so it cannot escalate again.
                return nativeRunMctsSearchSeqHalving(
                    bc, static_cast<int>(g_params.escalationSims),
                    useCrn, crnBase, useSearchSeed, searchSeed);
            }
        } else {
            g_lastSearchDangerous = false;
            g_lastSearchEscalated = false;
        }
        int bestIdx = survivors[0];
        double bestQ = -std::numeric_limits<double>::infinity();
        for (int idx : survivors) {
            const double q = root->N[idx] > 0 ? root->W[idx] / root->N[idx] : -std::numeric_limits<double>::infinity();
            if (q > bestQ) {
                bestQ = q;
                bestIdx = idx;
            }
        }
        if (reuse) {
            g_reusePrevRoot = root;
            g_reusePrevChosen = bestIdx;
        }
        return {root->actions[bestIdx], std::vector<std::int64_t>(root->N.begin(), root->N.end())};
    }

    std::pair<search::Action, std::vector<std::int64_t>> nativeRunMctsSearch(
            const BattleContext &bc, int nSimulations, bool useCrn, std::uint64_t crnBase,
            bool useSearchSeed = false, std::uint64_t searchSeed = 0) {
        if (g_useSeqHalving) {
            return nativeRunMctsSearchSeqHalving(bc, nSimulations, useCrn, crnBase, useSearchSeed, searchSeed);
        }
        // The rollout policy samples (Gumbel-max) whenever rolloutTemperature > 0, so
        // its RNG has to be tied to this call's seed or the search is not reproducible
        // and sibling candidates lose common random numbers.
        nativeSeedGumbel(useSearchSeed ? searchSeed : std::random_device{}());
        g_rootMaxHp = bc.player.maxHp;
        g_rootGold = bc.player.gold;
        const bool reuse = g_params.treeReuse != 0.0;
        MctsArena localArena;
        MctsNode *reused = nullptr;
        if (reuse) {
            reused = nativeFindReuseRoot(bc);
            if (reused != nullptr) {
                ++g_reuseHits;
            } else {
                ++g_reuseMisses;
                // Miss (or battle boundary): everything stored is stale.
                g_reuseArena.reset();
                g_reusePrevRoot = nullptr;
                g_reusePrevChosen = -1;
            }
            if (!g_reuseArena) {
                g_reuseArena = std::make_unique<MctsArena>();
            }
        } else if (g_reuseArena) {
            nativeResetSearchTree();
        }
        MctsArena &arena = reuse ? *g_reuseArena : localArena;
        MctsNode *root = arena.newNode(BattleContext(bc));
        std::unordered_map<NativeStateKey, MctsNode *, NativeStateKeyHash> transTable;
        std::random_device rd;
        std::mt19937_64 rng(useSearchSeed ? searchSeed : rd());
        // One determinization-pairing base per SEARCH CALL: candidates inside
        // this call share it (that is the pairing), successive decisions do
        // not (that would replay the same futures all fight).
        g_pairSeedBase = useCrn ? crnBase
            : (useSearchSeed ? searchSeed : rng());
        g_pairIndex = -1;
        const int maxTurn = bc.turn + static_cast<int>(g_params.searchMaxTurns);
        // One reused buffer rather than a fresh vector per simulation: cleared each time, so it
        // keeps its capacity and the RAVE path costs no per-simulation allocation.
        std::vector<std::uint32_t> raveTrace;
        nativeMastReset();
        for (int i = 0; i < nSimulations; ++i) {
            raveTrace.clear();
            g_mastTrace.clear();
            const double value = nativeSimulate(
                arena, root, transTable, 0, useCrn, crnBase, rng, maxTurn, -1,
                g_useRave ? &raveTrace : nullptr);
            if (nativeMastActive()) {
                nativeMastUpdate(value);
            }
        }
        int bestIdx = 0;
        std::int64_t bestN = -1;
        for (std::size_t i = 0; i < root->N.size(); ++i) {
            if (root->N[i] > bestN) {
                bestN = root->N[i];
                bestIdx = static_cast<int>(i);
            }
        }
        if (reuse) {
            g_reusePrevRoot = root;
            g_reusePrevChosen = bestIdx;
        }
        return {root->actions[bestIdx], std::vector<std::int64_t>(root->N.begin(), root->N.end())};
    }

    // Whole-run hybrid playout: our own tuned/per-card-aware native MCTS
    // (nativeRunMctsSearch, reading g_params) drives EVERY combat decision,
    // while Silverbot's own already-working ScumSearchAgent2::
    // stepOutOfCombatPolicy handles every non-combat decision (map path,
    // shop, rest site, events, card rewards, boss relic choice) completely
    // unchanged. `agent` is used ONLY for its stepOutOfCombatPolicy/rng
    // state here -- its own playoutBattle/BattleScumSearcher2 combat search
    // is never called. Building a from-scratch overworld policy tuned to
    // OUR own priorities is a much larger, separate undertaking; this reuses
    // Silverbot's mature, already-tested logic for everything except the one
    // piece this project actually improves on (combat), to get a genuinely
    // working whole-run simulator quickly. REQUIRES agent.pause_on_card_reward
    // == false (the default on a fresh Agent()) -- with it true,
    // stepOutOfCombatPolicy can return without advancing screenState,
    // which would spin this loop forever since it doesn't check `paused`.
    void nativeHybridPlayout(search::ScumSearchAgent2 &agent, GameContext &gc, int nSimulations) {
        while (gc.outcome == GameOutcome::UNDECIDED) {
            if (gc.screenState == ScreenState::BATTLE) {
                BattleContext bc;
                bc.init(gc);
                while (bc.outcome == Outcome::UNDECIDED) {
                    auto result = nativeRunMctsSearch(bc, nSimulations, false, 0);
                    result.first.execute(bc);
                }
                bc.exitBattle(gc);
                continue;
            }
            agent.stepOutOfCombatPolicy(gc);
        }
    }

    // Plays ONE already-constructed battle to completion using our own tuned native MCTS
    // (nativeRunMctsSearch, reading g_params), entirely in C++ -- the single-battle inner loop
    // nativeHybridPlayout above uses, factored out standalone so a caller measuring whole-fight
    // wall-clock cost doesn't pay Python round-trip overhead once per DECISION the way a Python-
    // side `while bc.outcome == UNDECIDED: action, _ = run_mcts_search(bc, n); action.execute(bc)`
    // loop would -- added specifically to keep an engine-speed comparison against another engine's
    // own native playout_battle honest (same call shape: one Python call in, mutates bc in place,
    // returns once the fight is over).
    struct NativePlayoutStats {
        std::uint64_t decisions = 0;
        std::uint64_t searchedDecisions = 0;
        std::uint64_t simulations = 0;
        std::uint64_t stallFallbackDecisions = 0;
        std::uint64_t stallProgressOverrideDecisions = 0;
        std::uint64_t softTempoOverrideDecisions = 0;
        std::uint64_t stallRecoverySearchDecisions = 0;
        std::uint64_t maxConsecutiveStallFallbacks = 0;
        int firstStallTurn = -1;
        int lastStallTurn = -1;
        int firstStallPlayerHp = -1;
        int firstStallMonsterHp = -1;
        int firstTempoOverrideTurn = -1;
        int firstTempoOverridePlayerHp = -1;
        int firstTempoOverrideMonsterHp = -1;
    };

    NativePlayoutStats nativePlayoutBattle(
            BattleContext &bc, int nSimulations,
            bool useSearchSeed = false, std::uint64_t searchSeedBase = 0) {
        NativePlayoutStats stats;
        const auto livingMonsterHp = [](const BattleContext &state) {
            int hp = 0;
            for (int i = 0; i < state.monsters.monsterCount; ++i) {
                const Monster &monster = state.monsters.arr[i];
                if (monster.curHp > 0 || monster.halfDead) {
                    hp += monster.halfDead ? monster.maxHp : monster.curHp;
                }
            }
            return hp;
        };
        // Block is temporary HP for the purpose of progress detection.  In
        // particular, Spheric Guardian can sit at one visible HP behind a
        // large block stack; attacks that chip that block are meaningful
        // progress and must not trigger a defensive-loop recovery.
        const auto livingMonsterDurability = [](const BattleContext &state) {
            int durability = 0;
            for (int i = 0; i < state.monsters.monsterCount; ++i) {
                const Monster &monster = state.monsters.arr[i];
                if (monster.curHp > 0 || monster.halfDead) {
                    durability += (monster.halfDead ? monster.maxHp : monster.curHp)
                            + std::max(0, monster.block);
                }
            }
            return durability;
        };
        // The generic rollout heuristic is normally a good recovery policy,
        // but a few altered-deck states can repeatedly prefer setup/block
        // after MCTS has already gone twenty turns without reducing monster
        // HP.  On that exceptional path only, choose a legal action that
        // makes immediate HP progress *and* is safe against the current
        // incoming damage.  Previewing is cheap here because hard stalls are
        // rare; the hot rollout path remains allocation-free.
        const auto safeProgressAction = [&](int currentMonsterDurability)
                -> std::optional<search::Action> {
            const auto legal = sts::py::getLegalActions(bc);
            std::optional<search::Action> best;
            int bestMonsterDurability = currentMonsterDurability;
            int bestPlayerHp = std::numeric_limits<int>::min();
            for (const auto &action : legal) {
                if (action.getActionType() == search::ActionType::END_TURN) {
                    continue;
                }
                BattleContext preview(bc);
                action.execute(preview);
                if (preview.outcome == Outcome::PLAYER_LOSS) {
                    continue;
                }
                const int remainingMonsterDurability = livingMonsterDurability(preview);
                if (remainingMonsterDurability >= currentMonsterDurability) {
                    continue;
                }
                const bool fightFinished = remainingMonsterDurability == 0
                        || preview.outcome == Outcome::PLAYER_VICTORY;
                if (!fightFinished
                        && nativeComputeHeuristicContext(preview).unblocked > 0.0) {
                    continue;
                }
                if (!best || remainingMonsterDurability < bestMonsterDurability
                        || (remainingMonsterDurability == bestMonsterDurability
                            && preview.player.curHp > bestPlayerHp)) {
                    best = action;
                    bestMonsterDurability = remainingMonsterDurability;
                    bestPlayerHp = preview.player.curHp;
                }
            }
            return best;
        };
        constexpr int SOFT_TEMPO_TURNS = 12;
        constexpr int HARD_STALL_TURNS = 20;
        constexpr std::uint64_t MAX_CONSECUTIVE_STALL_FALLBACKS = 3;
        const bool softTempoEligibleEncounter =
                bc.encounter == MonsterEncounter::TWO_LOUSE
                || bc.encounter == MonsterEncounter::SMALL_SLIMES
                || bc.encounter == MonsterEncounter::GREMLIN_GANG
                || bc.encounter == MonsterEncounter::LARGE_SLIME
                || bc.encounter == MonsterEncounter::LOTS_OF_SLIMES
                || bc.encounter == MonsterEncounter::EXORDIUM_WILDLIFE
                || bc.encounter == MonsterEncounter::THREE_LOUSE;
        int bestMonsterDurability = livingMonsterDurability(bc);
        int lastProgressTurn = bc.turn;
        std::uint64_t consecutiveStallFallbacks = 0;
        while (bc.outcome == Outcome::UNDECIDED) {
            const int currentMonsterDurability = livingMonsterDurability(bc);
            if (currentMonsterDurability < bestMonsterDurability) {
                bestMonsterDurability = currentMonsterDurability;
                lastProgressTurn = bc.turn;
                consecutiveStallFallbacks = 0;
            }
            const int stalledTurns = bc.turn - lastProgressTurn;
            const std::uint64_t decisionIndex = stats.decisions++;
            // A real combat should not go twenty full turns without removing
            // any monster HP. If the search objective enters a defensive
            // attractor, use the established rollout policy until damage
            // resumes instead of converting a healthy, winnable fight into
            // BattleContext's synthetic turn-501 loss. Three consecutive
            // overrides are the maximum before one MCTS recovery decision is
            // allowed through; this prevents an unbounded heuristic-only
            // suffix when the heuristic itself needs several setup actions.
            if (stalledTurns >= HARD_STALL_TURNS
                    && consecutiveStallFallbacks < MAX_CONSECUTIVE_STALL_FALLBACKS) {
                ++stats.stallFallbackDecisions;
                if (stats.firstStallTurn < 0) {
                    stats.firstStallTurn = bc.turn;
                    stats.firstStallPlayerHp = bc.player.curHp;
                    stats.firstStallMonsterHp = livingMonsterHp(bc);
                }
                stats.lastStallTurn = bc.turn;
                ++consecutiveStallFallbacks;
                stats.maxConsecutiveStallFallbacks = std::max(
                    stats.maxConsecutiveStallFallbacks, consecutiveStallFallbacks);
                const auto progressAction = safeProgressAction(currentMonsterDurability);
                if (progressAction) {
                    ++stats.stallProgressOverrideDecisions;
                    progressAction->execute(bc);
                } else {
                    nativeHeuristicPickFast(bc).execute(bc);
                }
                continue;
            }
            if (stalledTurns >= HARD_STALL_TURNS) {
                ++stats.stallRecoverySearchDecisions;
                consecutiveStallFallbacks = 0;
            }
            // A distinct deterministic stream per combat decision. The odd
            // golden-ratio increment avoids correlated nearby mt19937 seeds
            // while keeping an identical battle exactly reproducible.
            const std::uint64_t decisionSeed =
                    searchSeedBase + 0x9E3779B97F4A7C15ULL * decisionIndex;
            auto result = nativeRunMctsSearch(
                    bc, nSimulations, false, 0,
                    useSearchSeed, decisionSeed);
            std::uint64_t decisionSimulations = 0;
            for (const std::int64_t visits : result.second) {
                decisionSimulations += static_cast<std::uint64_t>(visits);
            }
            stats.simulations += decisionSimulations;
            stats.searchedDecisions += decisionSimulations > 0;

            search::Action selectedAction = result.first;
            // Soft tempo recovery: MCTS still receives its full budget. Once
            // a fight has made no HP progress for twelve turns, replace a
            // non-damaging root choice with the rollout heuristic only when
            // (a) the heuristic action immediately lowers living-monster HP
            // and (b) current block already covers predicted incoming damage.
            // This is deliberately narrower than a global turn penalty: it
            // cannot alter normal fights, sacrifice block on a dangerous
            // turn, or prefer a merely different setup action.
            if (softTempoEligibleEncounter
                    && stalledTurns >= SOFT_TEMPO_TURNS
                    && stalledTurns < HARD_STALL_TURNS) {
                const HeuristicContext ctx = nativeComputeHeuristicContext(bc);
                if (ctx.unblocked <= 0.0) {
                    const search::Action heuristicAction = nativeHeuristicPickFast(bc);
                    BattleContext mctsPreview(bc);
                    selectedAction.execute(mctsPreview);
                    BattleContext heuristicPreview(bc);
                    heuristicAction.execute(heuristicPreview);
                    const int mctsMonsterDurability = livingMonsterDurability(mctsPreview);
                    const int heuristicMonsterDurability = livingMonsterDurability(heuristicPreview);
                    if (mctsMonsterDurability >= currentMonsterDurability
                            && heuristicMonsterDurability < currentMonsterDurability
                            && heuristicPreview.player.curHp >= mctsPreview.player.curHp) {
                        selectedAction = heuristicAction;
                        ++stats.softTempoOverrideDecisions;
                        if (stats.firstTempoOverrideTurn < 0) {
                            stats.firstTempoOverrideTurn = bc.turn;
                            stats.firstTempoOverridePlayerHp = bc.player.curHp;
                            stats.firstTempoOverrideMonsterHp = livingMonsterHp(bc);
                        }
                    }
                }
            }
            selectedAction.execute(bc);
        }
        return stats;
    }

    // Cross-decision tree reuse (keeping the arena/transposition table alive
    // across decisions instead of a fresh tree every time, mirroring Silver
    // Automaton's BattleSearcher::rerootAt) has now caused THREE crashes
    // this session, including one AFTER adding defensive guards (search()
    // throwing on a terminal/unexpanded root instead of touching an empty
    // actions/N array) specifically meant to catch the leading hypothesis --
    // the guarded version still segfaulted immediately, with no exception
    // ever caught, refuting that hypothesis and confirming there is a
    // SECOND, still-unidentified memory-safety bug in this feature, not
    // just the one found by inspection. This is deliberately left
    // unimplemented rather than guessed at further. If revisited, it needs
    // the same dedicated isolation-flag methodology (a debug harness with
    // toggles, like az_search_debug.py used) that found the original two
    // crashes' real root cause this session -- not more inspection-based
    // patches under time pressure.

    // --- Bridge: construct a BattleContext from EXACT, externally-reported state ---
    // (live-game combat_state via CommunicationMod/spirecomm), for suggesting/playing
    // moves in the real running game. CommunicationMod is a debug/AI interface, not a
    // screen-reader -- it reports the true draw pile/hand/discard/exhaust contents
    // exactly, so there is no hidden-information/determinization problem here, unlike
    // what a human-observable-only bot would face.

    // Every PlayerStatus except INVALID, dispatched by canonical enum-name string to
    // the correct template instantiation. setStatusValueNoChecks<s> already special-
    // cases ARTIFACT/DEXTERITY/FOCUS/STRENGTH to their plain int fields internally
    // (confirmed in Player.h), so a uniform call here is correct for every status,
    // not just the statusMap-backed ones.
    void setPlayerStatusByName(Player &player, const std::string &name, int amount) {
        static const std::unordered_map<std::string, PlayerStatus> nameMap = []() {
            std::unordered_map<std::string, PlayerStatus> m;
            for (int i = 0; i < static_cast<int>(sizeof(playerStatusEnumStrings) / sizeof(char*)); ++i) {
                m[playerStatusEnumStrings[i]] = static_cast<PlayerStatus>(i);
            }
            return m;
        }();
        auto it = nameMap.find(name);
        if (it == nameMap.end()) {
            return;  // unrecognized status name -- caller already logs/skips these, not fatal here
        }
        switch (it->second) {
            case PS::DOUBLE_DAMAGE: player.setStatusValueNoChecks<PS::DOUBLE_DAMAGE>(amount); player.setHasStatus<PS::DOUBLE_DAMAGE>(true); break;
            case PS::DRAW_REDUCTION: player.setStatusValueNoChecks<PS::DRAW_REDUCTION>(amount); player.setHasStatus<PS::DRAW_REDUCTION>(true); break;
            case PS::FRAIL: player.setStatusValueNoChecks<PS::FRAIL>(amount); player.setHasStatus<PS::FRAIL>(true); break;
            case PS::INTANGIBLE: player.setStatusValueNoChecks<PS::INTANGIBLE>(amount); player.setHasStatus<PS::INTANGIBLE>(true); break;
            case PS::VULNERABLE: player.setStatusValueNoChecks<PS::VULNERABLE>(amount); player.setHasStatus<PS::VULNERABLE>(true); break;
            case PS::WEAK: player.setStatusValueNoChecks<PS::WEAK>(amount); player.setHasStatus<PS::WEAK>(true); break;
            case PS::BIAS: player.setStatusValueNoChecks<PS::BIAS>(amount); player.setHasStatus<PS::BIAS>(true); break;
            case PS::CONFUSED: player.setStatusValueNoChecks<PS::CONFUSED>(amount); player.setHasStatus<PS::CONFUSED>(true); break;
            case PS::CONSTRICTED: player.setStatusValueNoChecks<PS::CONSTRICTED>(amount); player.setHasStatus<PS::CONSTRICTED>(true); break;
            case PS::ENTANGLED: player.setStatusValueNoChecks<PS::ENTANGLED>(amount); player.setHasStatus<PS::ENTANGLED>(true); break;
            case PS::FASTING: player.setStatusValueNoChecks<PS::FASTING>(amount); player.setHasStatus<PS::FASTING>(true); break;
            case PS::HEX: player.setStatusValueNoChecks<PS::HEX>(amount); player.setHasStatus<PS::HEX>(true); break;
            case PS::LOSE_DEXTERITY: player.setStatusValueNoChecks<PS::LOSE_DEXTERITY>(amount); player.setHasStatus<PS::LOSE_DEXTERITY>(true); break;
            case PS::LOSE_STRENGTH: player.setStatusValueNoChecks<PS::LOSE_STRENGTH>(amount); player.setHasStatus<PS::LOSE_STRENGTH>(true); break;
            case PS::NO_BLOCK: player.setStatusValueNoChecks<PS::NO_BLOCK>(amount); player.setHasStatus<PS::NO_BLOCK>(true); break;
            case PS::NO_DRAW: player.setStatusValueNoChecks<PS::NO_DRAW>(amount); player.setHasStatus<PS::NO_DRAW>(true); break;
            case PS::WRAITH_FORM: player.setStatusValueNoChecks<PS::WRAITH_FORM>(amount); player.setHasStatus<PS::WRAITH_FORM>(true); break;
            case PS::BARRICADE: player.setStatusValueNoChecks<PS::BARRICADE>(amount); player.setHasStatus<PS::BARRICADE>(true); break;
            case PS::BLASPHEMER: player.setStatusValueNoChecks<PS::BLASPHEMER>(amount); player.setHasStatus<PS::BLASPHEMER>(true); break;
            case PS::CORRUPTION: player.setStatusValueNoChecks<PS::CORRUPTION>(amount); player.setHasStatus<PS::CORRUPTION>(true); break;
            case PS::ELECTRO: player.setStatusValueNoChecks<PS::ELECTRO>(amount); player.setHasStatus<PS::ELECTRO>(true); break;
            case PS::SURROUNDED: player.setStatusValueNoChecks<PS::SURROUNDED>(amount); player.setHasStatus<PS::SURROUNDED>(true); break;
            case PS::MASTER_REALITY: player.setStatusValueNoChecks<PS::MASTER_REALITY>(amount); player.setHasStatus<PS::MASTER_REALITY>(true); break;
            case PS::PEN_NIB: player.setStatusValueNoChecks<PS::PEN_NIB>(amount); player.setHasStatus<PS::PEN_NIB>(true); break;
            case PS::WRATH_NEXT_TURN: player.setStatusValueNoChecks<PS::WRATH_NEXT_TURN>(amount); player.setHasStatus<PS::WRATH_NEXT_TURN>(true); break;
            case PS::AMPLIFY: player.setStatusValueNoChecks<PS::AMPLIFY>(amount); player.setHasStatus<PS::AMPLIFY>(true); break;
            case PS::BLUR: player.setStatusValueNoChecks<PS::BLUR>(amount); player.setHasStatus<PS::BLUR>(true); break;
            case PS::BUFFER: player.setStatusValueNoChecks<PS::BUFFER>(amount); player.setHasStatus<PS::BUFFER>(true); break;
            case PS::COLLECT: player.setStatusValueNoChecks<PS::COLLECT>(amount); player.setHasStatus<PS::COLLECT>(true); break;
            case PS::DOUBLE_TAP: player.setStatusValueNoChecks<PS::DOUBLE_TAP>(amount); player.setHasStatus<PS::DOUBLE_TAP>(true); break;
            case PS::DUPLICATION: player.setStatusValueNoChecks<PS::DUPLICATION>(amount); player.setHasStatus<PS::DUPLICATION>(true); break;
            case PS::ECHO_FORM: player.setStatusValueNoChecks<PS::ECHO_FORM>(amount); player.setHasStatus<PS::ECHO_FORM>(true); break;
            case PS::FREE_ATTACK_POWER: player.setStatusValueNoChecks<PS::FREE_ATTACK_POWER>(amount); player.setHasStatus<PS::FREE_ATTACK_POWER>(true); break;
            case PS::REBOUND: player.setStatusValueNoChecks<PS::REBOUND>(amount); player.setHasStatus<PS::REBOUND>(true); break;
            case PS::MANTRA: player.setStatusValueNoChecks<PS::MANTRA>(amount); player.setHasStatus<PS::MANTRA>(true); break;
            case PS::ACCURACY: player.setStatusValueNoChecks<PS::ACCURACY>(amount); player.setHasStatus<PS::ACCURACY>(true); break;
            case PS::AFTER_IMAGE: player.setStatusValueNoChecks<PS::AFTER_IMAGE>(amount); player.setHasStatus<PS::AFTER_IMAGE>(true); break;
            case PS::BATTLE_HYMN: player.setStatusValueNoChecks<PS::BATTLE_HYMN>(amount); player.setHasStatus<PS::BATTLE_HYMN>(true); break;
            case PS::BRUTALITY: player.setStatusValueNoChecks<PS::BRUTALITY>(amount); player.setHasStatus<PS::BRUTALITY>(true); break;
            case PS::BURST: player.setStatusValueNoChecks<PS::BURST>(amount); player.setHasStatus<PS::BURST>(true); break;
            case PS::COMBUST: player.setStatusValueNoChecks<PS::COMBUST>(amount); player.setHasStatus<PS::COMBUST>(true); break;
            case PS::CREATIVE_AI: player.setStatusValueNoChecks<PS::CREATIVE_AI>(amount); player.setHasStatus<PS::CREATIVE_AI>(true); break;
            case PS::DARK_EMBRACE: player.setStatusValueNoChecks<PS::DARK_EMBRACE>(amount); player.setHasStatus<PS::DARK_EMBRACE>(true); break;
            case PS::DEMON_FORM: player.setStatusValueNoChecks<PS::DEMON_FORM>(amount); player.setHasStatus<PS::DEMON_FORM>(true); break;
            case PS::DEVA: player.setStatusValueNoChecks<PS::DEVA>(amount); player.setHasStatus<PS::DEVA>(true); break;
            case PS::DEVOTION: player.setStatusValueNoChecks<PS::DEVOTION>(amount); player.setHasStatus<PS::DEVOTION>(true); break;
            case PS::DRAW_CARD_NEXT_TURN: player.setStatusValueNoChecks<PS::DRAW_CARD_NEXT_TURN>(amount); player.setHasStatus<PS::DRAW_CARD_NEXT_TURN>(true); break;
            case PS::ENERGIZED: player.setStatusValueNoChecks<PS::ENERGIZED>(amount); player.setHasStatus<PS::ENERGIZED>(true); break;
            case PS::ENVENOM: player.setStatusValueNoChecks<PS::ENVENOM>(amount); player.setHasStatus<PS::ENVENOM>(true); break;
            case PS::ESTABLISHMENT: player.setStatusValueNoChecks<PS::ESTABLISHMENT>(amount); player.setHasStatus<PS::ESTABLISHMENT>(true); break;
            case PS::EVOLVE: player.setStatusValueNoChecks<PS::EVOLVE>(amount); player.setHasStatus<PS::EVOLVE>(true); break;
            case PS::FEEL_NO_PAIN: player.setStatusValueNoChecks<PS::FEEL_NO_PAIN>(amount); player.setHasStatus<PS::FEEL_NO_PAIN>(true); break;
            case PS::FIRE_BREATHING: player.setStatusValueNoChecks<PS::FIRE_BREATHING>(amount); player.setHasStatus<PS::FIRE_BREATHING>(true); break;
            case PS::FLAME_BARRIER: player.setStatusValueNoChecks<PS::FLAME_BARRIER>(amount); player.setHasStatus<PS::FLAME_BARRIER>(true); break;
            case PS::FOCUS: player.setStatusValueNoChecks<PS::FOCUS>(amount); player.setHasStatus<PS::FOCUS>(true); break;
            case PS::FORESIGHT: player.setStatusValueNoChecks<PS::FORESIGHT>(amount); player.setHasStatus<PS::FORESIGHT>(true); break;
            case PS::HELLO_WORLD: player.setStatusValueNoChecks<PS::HELLO_WORLD>(amount); player.setHasStatus<PS::HELLO_WORLD>(true); break;
            case PS::INFINITE_BLADES: player.setStatusValueNoChecks<PS::INFINITE_BLADES>(amount); player.setHasStatus<PS::INFINITE_BLADES>(true); break;
            case PS::JUGGERNAUT: player.setStatusValueNoChecks<PS::JUGGERNAUT>(amount); player.setHasStatus<PS::JUGGERNAUT>(true); break;
            case PS::LIKE_WATER: player.setStatusValueNoChecks<PS::LIKE_WATER>(amount); player.setHasStatus<PS::LIKE_WATER>(true); break;
            case PS::LOOP: player.setStatusValueNoChecks<PS::LOOP>(amount); player.setHasStatus<PS::LOOP>(true); break;
            case PS::MAGNETISM: player.setStatusValueNoChecks<PS::MAGNETISM>(amount); player.setHasStatus<PS::MAGNETISM>(true); break;
            case PS::MAYHEM: player.setStatusValueNoChecks<PS::MAYHEM>(amount); player.setHasStatus<PS::MAYHEM>(true); break;
            case PS::METALLICIZE: player.setStatusValueNoChecks<PS::METALLICIZE>(amount); player.setHasStatus<PS::METALLICIZE>(true); break;
            case PS::NEXT_TURN_BLOCK: player.setStatusValueNoChecks<PS::NEXT_TURN_BLOCK>(amount); player.setHasStatus<PS::NEXT_TURN_BLOCK>(true); break;
            case PS::NOXIOUS_FUMES: player.setStatusValueNoChecks<PS::NOXIOUS_FUMES>(amount); player.setHasStatus<PS::NOXIOUS_FUMES>(true); break;
            case PS::OMEGA: player.setStatusValueNoChecks<PS::OMEGA>(amount); player.setHasStatus<PS::OMEGA>(true); break;
            case PS::PANACHE: player.setStatusValueNoChecks<PS::PANACHE>(amount); player.setHasStatus<PS::PANACHE>(true); break;
            case PS::PHANTASMAL: player.setStatusValueNoChecks<PS::PHANTASMAL>(amount); player.setHasStatus<PS::PHANTASMAL>(true); break;
            case PS::PLATED_ARMOR: player.setStatusValueNoChecks<PS::PLATED_ARMOR>(amount); player.setHasStatus<PS::PLATED_ARMOR>(true); break;
            case PS::RAGE: player.setStatusValueNoChecks<PS::RAGE>(amount); player.setHasStatus<PS::RAGE>(true); break;
            case PS::REGEN: player.setStatusValueNoChecks<PS::REGEN>(amount); player.setHasStatus<PS::REGEN>(true); break;
            case PS::RITUAL: player.setStatusValueNoChecks<PS::RITUAL>(amount); player.setHasStatus<PS::RITUAL>(true); break;
            case PS::RUPTURE: player.setStatusValueNoChecks<PS::RUPTURE>(amount); player.setHasStatus<PS::RUPTURE>(true); break;
            case PS::SADISTIC: player.setStatusValueNoChecks<PS::SADISTIC>(amount); player.setHasStatus<PS::SADISTIC>(true); break;
            case PS::STATIC_DISCHARGE: player.setStatusValueNoChecks<PS::STATIC_DISCHARGE>(amount); player.setHasStatus<PS::STATIC_DISCHARGE>(true); break;
            case PS::THORNS: player.setStatusValueNoChecks<PS::THORNS>(amount); player.setHasStatus<PS::THORNS>(true); break;
            case PS::THOUSAND_CUTS: player.setStatusValueNoChecks<PS::THOUSAND_CUTS>(amount); player.setHasStatus<PS::THOUSAND_CUTS>(true); break;
            case PS::TOOLS_OF_THE_TRADE: player.setStatusValueNoChecks<PS::TOOLS_OF_THE_TRADE>(amount); player.setHasStatus<PS::TOOLS_OF_THE_TRADE>(true); break;
            case PS::VIGOR: player.setStatusValueNoChecks<PS::VIGOR>(amount); player.setHasStatus<PS::VIGOR>(true); break;
            case PS::WAVE_OF_THE_HAND: player.setStatusValueNoChecks<PS::WAVE_OF_THE_HAND>(amount); player.setHasStatus<PS::WAVE_OF_THE_HAND>(true); break;
            case PS::WELL_LAID_PLANS: player.setStatusValueNoChecks<PS::WELL_LAID_PLANS>(amount); player.setHasStatus<PS::WELL_LAID_PLANS>(true); break;
            case PS::EQUILIBRIUM: player.setStatusValueNoChecks<PS::EQUILIBRIUM>(amount); player.setHasStatus<PS::EQUILIBRIUM>(true); break;
            case PS::ARTIFACT: player.setStatusValueNoChecks<PS::ARTIFACT>(amount); player.setHasStatus<PS::ARTIFACT>(true); break;
            case PS::DEXTERITY: player.setStatusValueNoChecks<PS::DEXTERITY>(amount); player.setHasStatus<PS::DEXTERITY>(true); break;
            case PS::STRENGTH: player.setStatusValueNoChecks<PS::STRENGTH>(amount); player.setHasStatus<PS::STRENGTH>(true); break;
            case PS::THE_BOMB: player.setStatusValueNoChecks<PS::THE_BOMB>(amount); player.setHasStatus<PS::THE_BOMB>(true); break;
            default: break;  // INVALID or anything not covered above -- no-op
        }
    }

    // Every MonsterStatus except INVALID. NOTE: deliberately does NOT use the existing
    // monsterStatusEnumStrings[] array for the name->ordinal lookup -- it's misaligned
    // with the actual MonsterStatus enum declaration starting at REACTIVE/SHARP_HIDE
    // (confirmed by direct comparison; flagged separately for a dedicated fix). This
    // builds its own name map directly from the enum's own spelling instead.
    void setMonsterStatusByName(Monster &m, const std::string &name, int amount) {
        static const std::unordered_map<std::string, MonsterStatus> nameMap = {
            {"ARTIFACT", MonsterStatus::ARTIFACT},
            {"BLOCK_RETURN", MonsterStatus::BLOCK_RETURN},
            {"CHOKED", MonsterStatus::CHOKED},
            {"CORPSE_EXPLOSION", MonsterStatus::CORPSE_EXPLOSION},
            {"LOCK_ON", MonsterStatus::LOCK_ON},
            {"MARK", MonsterStatus::MARK},
            {"METALLICIZE", MonsterStatus::METALLICIZE},
            {"PLATED_ARMOR", MonsterStatus::PLATED_ARMOR},
            {"POISON", MonsterStatus::POISON},
            {"REGEN", MonsterStatus::REGEN},
            {"SHACKLED", MonsterStatus::SHACKLED},
            {"STRENGTH", MonsterStatus::STRENGTH},
            {"VULNERABLE", MonsterStatus::VULNERABLE},
            {"WEAK", MonsterStatus::WEAK},
            {"ANGRY", MonsterStatus::ANGRY},
            {"BEAT_OF_DEATH", MonsterStatus::BEAT_OF_DEATH},
            {"CURIOSITY", MonsterStatus::CURIOSITY},
            {"CURL_UP", MonsterStatus::CURL_UP},
            {"ENRAGE", MonsterStatus::ENRAGE},
            {"FADING", MonsterStatus::FADING},
            {"FLIGHT", MonsterStatus::FLIGHT},
            {"GENERIC_STRENGTH_UP", MonsterStatus::GENERIC_STRENGTH_UP},
            {"INTANGIBLE", MonsterStatus::INTANGIBLE},
            {"MALLEABLE", MonsterStatus::MALLEABLE},
            {"MODE_SHIFT", MonsterStatus::MODE_SHIFT},
            {"RITUAL", MonsterStatus::RITUAL},
            {"SLOW", MonsterStatus::SLOW},
            {"SPORE_CLOUD", MonsterStatus::SPORE_CLOUD},
            {"THIEVERY", MonsterStatus::THIEVERY},
            {"THORNS", MonsterStatus::THORNS},
            {"TIME_WARP", MonsterStatus::TIME_WARP},
            {"INVINCIBLE", MonsterStatus::INVINCIBLE},
            {"REACTIVE", MonsterStatus::REACTIVE},
            {"SHARP_HIDE", MonsterStatus::SHARP_HIDE},
            {"ASLEEP", MonsterStatus::ASLEEP},
            {"BARRICADE", MonsterStatus::BARRICADE},
            {"MINION", MonsterStatus::MINION},
            {"MINION_LEADER", MonsterStatus::MINION_LEADER},
            {"PAINFUL_STABS", MonsterStatus::PAINFUL_STABS},
            {"REGROW", MonsterStatus::REGROW},
            {"SHIFTING", MonsterStatus::SHIFTING},
            {"STASIS", MonsterStatus::STASIS},
        };
        auto it = nameMap.find(name);
        if (it == nameMap.end()) {
            return;
        }
        switch (it->second) {
            case MS::ARTIFACT: m.setStatus<MS::ARTIFACT>(amount); m.setHasStatus<MS::ARTIFACT>(true); break;
            case MS::BLOCK_RETURN: m.setStatus<MS::BLOCK_RETURN>(amount); m.setHasStatus<MS::BLOCK_RETURN>(true); break;
            case MS::CHOKED: m.setStatus<MS::CHOKED>(amount); m.setHasStatus<MS::CHOKED>(true); break;
            case MS::CORPSE_EXPLOSION: m.setStatus<MS::CORPSE_EXPLOSION>(amount); m.setHasStatus<MS::CORPSE_EXPLOSION>(true); break;
            case MS::LOCK_ON: m.setStatus<MS::LOCK_ON>(amount); m.setHasStatus<MS::LOCK_ON>(true); break;
            case MS::MARK: m.setStatus<MS::MARK>(amount); m.setHasStatus<MS::MARK>(true); break;
            case MS::METALLICIZE: m.setStatus<MS::METALLICIZE>(amount); m.setHasStatus<MS::METALLICIZE>(true); break;
            case MS::PLATED_ARMOR: m.setStatus<MS::PLATED_ARMOR>(amount); m.setHasStatus<MS::PLATED_ARMOR>(true); break;
            case MS::POISON: m.setStatus<MS::POISON>(amount); m.setHasStatus<MS::POISON>(true); break;
            case MS::REGEN: m.setStatus<MS::REGEN>(amount); m.setHasStatus<MS::REGEN>(true); break;
            case MS::SHACKLED: m.setStatus<MS::SHACKLED>(amount); m.setHasStatus<MS::SHACKLED>(true); break;
            case MS::STRENGTH: m.setStatus<MS::STRENGTH>(amount); m.setHasStatus<MS::STRENGTH>(true); break;
            case MS::VULNERABLE: m.setStatus<MS::VULNERABLE>(amount); m.setHasStatus<MS::VULNERABLE>(true); break;
            case MS::WEAK: m.setStatus<MS::WEAK>(amount); m.setHasStatus<MS::WEAK>(true); break;
            case MS::ANGRY: m.setStatus<MS::ANGRY>(amount); m.setHasStatus<MS::ANGRY>(true); break;
            case MS::BEAT_OF_DEATH: m.setStatus<MS::BEAT_OF_DEATH>(amount); m.setHasStatus<MS::BEAT_OF_DEATH>(true); break;
            case MS::CURIOSITY: m.setStatus<MS::CURIOSITY>(amount); m.setHasStatus<MS::CURIOSITY>(true); break;
            case MS::CURL_UP: m.setStatus<MS::CURL_UP>(amount); m.setHasStatus<MS::CURL_UP>(true); break;
            case MS::ENRAGE: m.setStatus<MS::ENRAGE>(amount); m.setHasStatus<MS::ENRAGE>(true); break;
            case MS::FADING: m.setStatus<MS::FADING>(amount); m.setHasStatus<MS::FADING>(true); break;
            case MS::FLIGHT: m.setStatus<MS::FLIGHT>(amount); m.setHasStatus<MS::FLIGHT>(true); break;
            case MS::GENERIC_STRENGTH_UP: m.setStatus<MS::GENERIC_STRENGTH_UP>(amount); m.setHasStatus<MS::GENERIC_STRENGTH_UP>(true); break;
            case MS::INTANGIBLE: m.setStatus<MS::INTANGIBLE>(amount); m.setHasStatus<MS::INTANGIBLE>(true); break;
            case MS::MALLEABLE: m.setStatus<MS::MALLEABLE>(amount); m.setHasStatus<MS::MALLEABLE>(true); break;
            case MS::MODE_SHIFT: m.setStatus<MS::MODE_SHIFT>(amount); m.setHasStatus<MS::MODE_SHIFT>(true); break;
            case MS::RITUAL: m.setStatus<MS::RITUAL>(amount); m.setHasStatus<MS::RITUAL>(true); break;
            case MS::SLOW: m.setStatus<MS::SLOW>(amount); m.setHasStatus<MS::SLOW>(true); break;
            case MS::SPORE_CLOUD: m.setStatus<MS::SPORE_CLOUD>(amount); m.setHasStatus<MS::SPORE_CLOUD>(true); break;
            case MS::THIEVERY: m.setStatus<MS::THIEVERY>(amount); m.setHasStatus<MS::THIEVERY>(true); break;
            case MS::THORNS: m.setStatus<MS::THORNS>(amount); m.setHasStatus<MS::THORNS>(true); break;
            case MS::TIME_WARP: m.setStatus<MS::TIME_WARP>(amount); m.setHasStatus<MS::TIME_WARP>(true); break;
            case MS::INVINCIBLE: m.setStatus<MS::INVINCIBLE>(amount); m.setHasStatus<MS::INVINCIBLE>(true); break;
            case MS::REACTIVE: m.setStatus<MS::REACTIVE>(amount); m.setHasStatus<MS::REACTIVE>(true); break;
            case MS::SHARP_HIDE: m.setStatus<MS::SHARP_HIDE>(amount); m.setHasStatus<MS::SHARP_HIDE>(true); break;
            case MS::ASLEEP: m.setStatus<MS::ASLEEP>(amount); m.setHasStatus<MS::ASLEEP>(true); break;
            case MS::BARRICADE: m.setStatus<MS::BARRICADE>(amount); m.setHasStatus<MS::BARRICADE>(true); break;
            case MS::MINION: m.setStatus<MS::MINION>(amount); m.setHasStatus<MS::MINION>(true); break;
            case MS::MINION_LEADER: m.setStatus<MS::MINION_LEADER>(amount); m.setHasStatus<MS::MINION_LEADER>(true); break;
            case MS::PAINFUL_STABS: m.setStatus<MS::PAINFUL_STABS>(amount); m.setHasStatus<MS::PAINFUL_STABS>(true); break;
            case MS::REGROW: m.setStatus<MS::REGROW>(amount); m.setHasStatus<MS::REGROW>(true); break;
            case MS::SHIFTING: m.setStatus<MS::SHIFTING>(amount); m.setHasStatus<MS::SHIFTING>(true); break;
            case MS::STASIS: m.setStatus<MS::STASIS>(amount); m.setHasStatus<MS::STASIS>(true); break;
            default: break;
        }
    }

    // Canonical MonsterMoveId name -> enum, for setting a monster's currently-known
    // queued move (moveHistory[0]) from a live game's reported move -- see
    // Monster::getMoveBaseDamage, a pure function of moveHistory[0]/ascension with no
    // RNG dependency, so setting this correctly makes incoming-damage prediction exact
    // for the immediate turn. Building the actual per-monster spirecomm-move_id-to-name
    // calibration table is the Python bridge's job (best-effort, expanded via live
    // testing) -- this is just the mechanical name->enum lookup.
    MonsterMoveId lookupMonsterMoveId(const std::string &name) {
        static const std::unordered_map<std::string, MonsterMoveId> nameMap = {
        {"GENERIC_ESCAPE_MOVE", MMID::GENERIC_ESCAPE_MOVE},
        {"ACID_SLIME_L_CORROSIVE_SPIT", MMID::ACID_SLIME_L_CORROSIVE_SPIT},
        {"ACID_SLIME_L_LICK", MMID::ACID_SLIME_L_LICK},
        {"ACID_SLIME_L_TACKLE", MMID::ACID_SLIME_L_TACKLE},
        {"ACID_SLIME_L_SPLIT", MMID::ACID_SLIME_L_SPLIT},
        {"ACID_SLIME_M_CORROSIVE_SPIT", MMID::ACID_SLIME_M_CORROSIVE_SPIT},
        {"ACID_SLIME_M_LICK", MMID::ACID_SLIME_M_LICK},
        {"ACID_SLIME_M_TACKLE", MMID::ACID_SLIME_M_TACKLE},
        {"ACID_SLIME_S_LICK", MMID::ACID_SLIME_S_LICK},
        {"ACID_SLIME_S_TACKLE", MMID::ACID_SLIME_S_TACKLE},
        {"AWAKENED_ONE_SLASH", MMID::AWAKENED_ONE_SLASH},
        {"AWAKENED_ONE_SOUL_STRIKE", MMID::AWAKENED_ONE_SOUL_STRIKE},
        {"AWAKENED_ONE_REBIRTH", MMID::AWAKENED_ONE_REBIRTH},
        {"AWAKENED_ONE_DARK_ECHO", MMID::AWAKENED_ONE_DARK_ECHO},
        {"AWAKENED_ONE_SLUDGE", MMID::AWAKENED_ONE_SLUDGE},
        {"AWAKENED_ONE_TACKLE", MMID::AWAKENED_ONE_TACKLE},
        {"BEAR_BEAR_HUG", MMID::BEAR_BEAR_HUG},
        {"BEAR_LUNGE", MMID::BEAR_LUNGE},
        {"BEAR_MAUL", MMID::BEAR_MAUL},
        {"BLUE_SLAVER_STAB", MMID::BLUE_SLAVER_STAB},
        {"BLUE_SLAVER_RAKE", MMID::BLUE_SLAVER_RAKE},
        {"BOOK_OF_STABBING_MULTI_STAB", MMID::BOOK_OF_STABBING_MULTI_STAB},
        {"BOOK_OF_STABBING_SINGLE_STAB", MMID::BOOK_OF_STABBING_SINGLE_STAB},
        {"BRONZE_AUTOMATON_BOOST", MMID::BRONZE_AUTOMATON_BOOST},
        {"BRONZE_AUTOMATON_FLAIL", MMID::BRONZE_AUTOMATON_FLAIL},
        {"BRONZE_AUTOMATON_HYPER_BEAM", MMID::BRONZE_AUTOMATON_HYPER_BEAM},
        {"BRONZE_AUTOMATON_SPAWN_ORBS", MMID::BRONZE_AUTOMATON_SPAWN_ORBS},
        {"BRONZE_AUTOMATON_STUNNED", MMID::BRONZE_AUTOMATON_STUNNED},
        {"BRONZE_ORB_BEAM", MMID::BRONZE_ORB_BEAM},
        {"BRONZE_ORB_STASIS", MMID::BRONZE_ORB_STASIS},
        {"BRONZE_ORB_SUPPORT_BEAM", MMID::BRONZE_ORB_SUPPORT_BEAM},
        {"BYRD_CAW", MMID::BYRD_CAW},
        {"BYRD_FLY", MMID::BYRD_FLY},
        {"BYRD_HEADBUTT", MMID::BYRD_HEADBUTT},
        {"BYRD_PECK", MMID::BYRD_PECK},
        {"BYRD_STUNNED", MMID::BYRD_STUNNED},
        {"BYRD_SWOOP", MMID::BYRD_SWOOP},
        {"CENTURION_SLASH", MMID::CENTURION_SLASH},
        {"CENTURION_FURY", MMID::CENTURION_FURY},
        {"CENTURION_DEFEND", MMID::CENTURION_DEFEND},
        {"CHOSEN_POKE", MMID::CHOSEN_POKE},
        {"CHOSEN_ZAP", MMID::CHOSEN_ZAP},
        {"CHOSEN_DEBILITATE", MMID::CHOSEN_DEBILITATE},
        {"CHOSEN_DRAIN", MMID::CHOSEN_DRAIN},
        {"CHOSEN_HEX", MMID::CHOSEN_HEX},
        {"CORRUPT_HEART_DEBILITATE", MMID::CORRUPT_HEART_DEBILITATE},
        {"CORRUPT_HEART_BLOOD_SHOTS", MMID::CORRUPT_HEART_BLOOD_SHOTS},
        {"CORRUPT_HEART_ECHO", MMID::CORRUPT_HEART_ECHO},
        {"CORRUPT_HEART_BUFF", MMID::CORRUPT_HEART_BUFF},
        {"CULTIST_INCANTATION", MMID::CULTIST_INCANTATION},
        {"CULTIST_DARK_STRIKE", MMID::CULTIST_DARK_STRIKE},
        {"DAGGER_STAB", MMID::DAGGER_STAB},
        {"DAGGER_EXPLODE", MMID::DAGGER_EXPLODE},
        {"DARKLING_NIP", MMID::DARKLING_NIP},
        {"DARKLING_CHOMP", MMID::DARKLING_CHOMP},
        {"DARKLING_HARDEN", MMID::DARKLING_HARDEN},
        {"DARKLING_REINCARNATE", MMID::DARKLING_REINCARNATE},
        {"DARKLING_REGROW", MMID::DARKLING_REGROW},
        {"DECA_SQUARE_OF_PROTECTION", MMID::DECA_SQUARE_OF_PROTECTION},
        {"DECA_BEAM", MMID::DECA_BEAM},
        {"DONU_CIRCLE_OF_POWER", MMID::DONU_CIRCLE_OF_POWER},
        {"DONU_BEAM", MMID::DONU_BEAM},
        {"EXPLODER_SLAM", MMID::EXPLODER_SLAM},
        {"EXPLODER_EXPLODE", MMID::EXPLODER_EXPLODE},
        {"FAT_GREMLIN_SMASH", MMID::FAT_GREMLIN_SMASH},
        {"FUNGI_BEAST_BITE", MMID::FUNGI_BEAST_BITE},
        {"FUNGI_BEAST_GROW", MMID::FUNGI_BEAST_GROW},
        {"GIANT_HEAD_COUNT", MMID::GIANT_HEAD_COUNT},
        {"GIANT_HEAD_GLARE", MMID::GIANT_HEAD_GLARE},
        {"GIANT_HEAD_IT_IS_TIME", MMID::GIANT_HEAD_IT_IS_TIME},
        {"GREEN_LOUSE_BITE", MMID::GREEN_LOUSE_BITE},
        {"GREEN_LOUSE_SPIT_WEB", MMID::GREEN_LOUSE_SPIT_WEB},
        {"GREMLIN_LEADER_ENCOURAGE", MMID::GREMLIN_LEADER_ENCOURAGE},
        {"GREMLIN_LEADER_RALLY", MMID::GREMLIN_LEADER_RALLY},
        {"GREMLIN_LEADER_STAB", MMID::GREMLIN_LEADER_STAB},
        {"GREMLIN_NOB_BELLOW", MMID::GREMLIN_NOB_BELLOW},
        {"GREMLIN_NOB_RUSH", MMID::GREMLIN_NOB_RUSH},
        {"GREMLIN_NOB_SKULL_BASH", MMID::GREMLIN_NOB_SKULL_BASH},
        {"GREMLIN_WIZARD_CHARGING", MMID::GREMLIN_WIZARD_CHARGING},
        {"GREMLIN_WIZARD_ULTIMATE_BLAST", MMID::GREMLIN_WIZARD_ULTIMATE_BLAST},
        {"HEXAGHOST_ACTIVATE", MMID::HEXAGHOST_ACTIVATE},
        {"HEXAGHOST_DIVIDER", MMID::HEXAGHOST_DIVIDER},
        {"HEXAGHOST_INFERNO", MMID::HEXAGHOST_INFERNO},
        {"HEXAGHOST_SEAR", MMID::HEXAGHOST_SEAR},
        {"HEXAGHOST_TACKLE", MMID::HEXAGHOST_TACKLE},
        {"HEXAGHOST_INFLAME", MMID::HEXAGHOST_INFLAME},
        {"JAW_WORM_CHOMP", MMID::JAW_WORM_CHOMP},
        {"JAW_WORM_THRASH", MMID::JAW_WORM_THRASH},
        {"JAW_WORM_BELLOW", MMID::JAW_WORM_BELLOW},
        {"LAGAVULIN_ATTACK", MMID::LAGAVULIN_ATTACK},
        {"LAGAVULIN_SIPHON_SOUL", MMID::LAGAVULIN_SIPHON_SOUL},
        {"LAGAVULIN_SLEEP", MMID::LAGAVULIN_SLEEP},
        {"LOOTER_MUG", MMID::LOOTER_MUG},
        {"LOOTER_LUNGE", MMID::LOOTER_LUNGE},
        {"LOOTER_SMOKE_BOMB", MMID::LOOTER_SMOKE_BOMB},
        {"LOOTER_ESCAPE", MMID::LOOTER_ESCAPE},
        {"MAD_GREMLIN_SCRATCH", MMID::MAD_GREMLIN_SCRATCH},
        {"MUGGER_MUG", MMID::MUGGER_MUG},
        {"MUGGER_LUNGE", MMID::MUGGER_LUNGE},
        {"MUGGER_SMOKE_BOMB", MMID::MUGGER_SMOKE_BOMB},
        {"MUGGER_ESCAPE", MMID::MUGGER_ESCAPE},
        {"MYSTIC_HEAL", MMID::MYSTIC_HEAL},
        {"MYSTIC_BUFF", MMID::MYSTIC_BUFF},
        {"MYSTIC_ATTACK_DEBUFF", MMID::MYSTIC_ATTACK_DEBUFF},
        {"NEMESIS_DEBUFF", MMID::NEMESIS_DEBUFF},
        {"NEMESIS_ATTACK", MMID::NEMESIS_ATTACK},
        {"NEMESIS_SCYTHE", MMID::NEMESIS_SCYTHE},
        {"ORB_WALKER_LASER", MMID::ORB_WALKER_LASER},
        {"ORB_WALKER_CLAW", MMID::ORB_WALKER_CLAW},
        {"POINTY_ATTACK", MMID::POINTY_ATTACK},
        {"RED_LOUSE_BITE", MMID::RED_LOUSE_BITE},
        {"RED_LOUSE_GROW", MMID::RED_LOUSE_GROW},
        {"RED_SLAVER_STAB", MMID::RED_SLAVER_STAB},
        {"RED_SLAVER_SCRAPE", MMID::RED_SLAVER_SCRAPE},
        {"RED_SLAVER_ENTANGLE", MMID::RED_SLAVER_ENTANGLE},
        {"REPTOMANCER_SUMMON", MMID::REPTOMANCER_SUMMON},
        {"REPTOMANCER_SNAKE_STRIKE", MMID::REPTOMANCER_SNAKE_STRIKE},
        {"REPTOMANCER_BIG_BITE", MMID::REPTOMANCER_BIG_BITE},
        {"REPULSOR_BASH", MMID::REPULSOR_BASH},
        {"REPULSOR_REPULSE", MMID::REPULSOR_REPULSE},
        {"ROMEO_MOCK", MMID::ROMEO_MOCK},
        {"ROMEO_AGONIZING_SLASH", MMID::ROMEO_AGONIZING_SLASH},
        {"ROMEO_CROSS_SLASH", MMID::ROMEO_CROSS_SLASH},
        {"SENTRY_BEAM", MMID::SENTRY_BEAM},
        {"SENTRY_BOLT", MMID::SENTRY_BOLT},
        {"SHELLED_PARASITE_DOUBLE_STRIKE", MMID::SHELLED_PARASITE_DOUBLE_STRIKE},
        {"SHELLED_PARASITE_FELL", MMID::SHELLED_PARASITE_FELL},
        {"SHELLED_PARASITE_STUNNED", MMID::SHELLED_PARASITE_STUNNED},
        {"SHELLED_PARASITE_SUCK", MMID::SHELLED_PARASITE_SUCK},
        {"SHIELD_GREMLIN_PROTECT", MMID::SHIELD_GREMLIN_PROTECT},
        {"SHIELD_GREMLIN_SHIELD_BASH", MMID::SHIELD_GREMLIN_SHIELD_BASH},
        {"SLIME_BOSS_GOOP_SPRAY", MMID::SLIME_BOSS_GOOP_SPRAY},
        {"SLIME_BOSS_PREPARING", MMID::SLIME_BOSS_PREPARING},
        {"SLIME_BOSS_SLAM", MMID::SLIME_BOSS_SLAM},
        {"SLIME_BOSS_SPLIT", MMID::SLIME_BOSS_SPLIT},
        {"SNAKE_PLANT_CHOMP", MMID::SNAKE_PLANT_CHOMP},
        {"SNAKE_PLANT_ENFEEBLING_SPORES", MMID::SNAKE_PLANT_ENFEEBLING_SPORES},
        {"SNEAKY_GREMLIN_PUNCTURE", MMID::SNEAKY_GREMLIN_PUNCTURE},
        {"SNECKO_PERPLEXING_GLARE", MMID::SNECKO_PERPLEXING_GLARE},
        {"SNECKO_TAIL_WHIP", MMID::SNECKO_TAIL_WHIP},
        {"SNECKO_BITE", MMID::SNECKO_BITE},
        {"SPHERIC_GUARDIAN_SLAM", MMID::SPHERIC_GUARDIAN_SLAM},
        {"SPHERIC_GUARDIAN_ACTIVATE", MMID::SPHERIC_GUARDIAN_ACTIVATE},
        {"SPHERIC_GUARDIAN_HARDEN", MMID::SPHERIC_GUARDIAN_HARDEN},
        {"SPHERIC_GUARDIAN_ATTACK_DEBUFF", MMID::SPHERIC_GUARDIAN_ATTACK_DEBUFF},
        {"SPIKER_CUT", MMID::SPIKER_CUT},
        {"SPIKER_SPIKE", MMID::SPIKER_SPIKE},
        {"SPIKE_SLIME_L_FLAME_TACKLE", MMID::SPIKE_SLIME_L_FLAME_TACKLE},
        {"SPIKE_SLIME_L_LICK", MMID::SPIKE_SLIME_L_LICK},
        {"SPIKE_SLIME_L_SPLIT", MMID::SPIKE_SLIME_L_SPLIT},
        {"SPIKE_SLIME_M_FLAME_TACKLE", MMID::SPIKE_SLIME_M_FLAME_TACKLE},
        {"SPIKE_SLIME_M_LICK", MMID::SPIKE_SLIME_M_LICK},
        {"SPIKE_SLIME_S_TACKLE", MMID::SPIKE_SLIME_S_TACKLE},
        {"SPIRE_GROWTH_QUICK_TACKLE", MMID::SPIRE_GROWTH_QUICK_TACKLE},
        {"SPIRE_GROWTH_SMASH", MMID::SPIRE_GROWTH_SMASH},
        {"SPIRE_GROWTH_CONSTRICT", MMID::SPIRE_GROWTH_CONSTRICT},
        {"SPIRE_SHIELD_BASH", MMID::SPIRE_SHIELD_BASH},
        {"SPIRE_SHIELD_FORTIFY", MMID::SPIRE_SHIELD_FORTIFY},
        {"SPIRE_SHIELD_SMASH", MMID::SPIRE_SHIELD_SMASH},
        {"SPIRE_SPEAR_BURN_STRIKE", MMID::SPIRE_SPEAR_BURN_STRIKE},
        {"SPIRE_SPEAR_PIERCER", MMID::SPIRE_SPEAR_PIERCER},
        {"SPIRE_SPEAR_SKEWER", MMID::SPIRE_SPEAR_SKEWER},
        {"TASKMASTER_SCOURING_WHIP", MMID::TASKMASTER_SCOURING_WHIP},
        {"TORCH_HEAD_TACKLE", MMID::TORCH_HEAD_TACKLE},
        {"THE_CHAMP_DEFENSIVE_STANCE", MMID::THE_CHAMP_DEFENSIVE_STANCE},
        {"THE_CHAMP_FACE_SLAP", MMID::THE_CHAMP_FACE_SLAP},
        {"THE_CHAMP_TAUNT", MMID::THE_CHAMP_TAUNT},
        {"THE_CHAMP_HEAVY_SLASH", MMID::THE_CHAMP_HEAVY_SLASH},
        {"THE_CHAMP_GLOAT", MMID::THE_CHAMP_GLOAT},
        {"THE_CHAMP_EXECUTE", MMID::THE_CHAMP_EXECUTE},
        {"THE_CHAMP_ANGER", MMID::THE_CHAMP_ANGER},
        {"THE_COLLECTOR_BUFF", MMID::THE_COLLECTOR_BUFF},
        {"THE_COLLECTOR_FIREBALL", MMID::THE_COLLECTOR_FIREBALL},
        {"THE_COLLECTOR_MEGA_DEBUFF", MMID::THE_COLLECTOR_MEGA_DEBUFF},
        {"THE_COLLECTOR_SPAWN", MMID::THE_COLLECTOR_SPAWN},
        {"THE_GUARDIAN_CHARGING_UP", MMID::THE_GUARDIAN_CHARGING_UP},
        {"THE_GUARDIAN_FIERCE_BASH", MMID::THE_GUARDIAN_FIERCE_BASH},
        {"THE_GUARDIAN_VENT_STEAM", MMID::THE_GUARDIAN_VENT_STEAM},
        {"THE_GUARDIAN_WHIRLWIND", MMID::THE_GUARDIAN_WHIRLWIND},
        {"THE_GUARDIAN_DEFENSIVE_MODE", MMID::THE_GUARDIAN_DEFENSIVE_MODE},
        {"THE_GUARDIAN_ROLL_ATTACK", MMID::THE_GUARDIAN_ROLL_ATTACK},
        {"THE_GUARDIAN_TWIN_SLAM", MMID::THE_GUARDIAN_TWIN_SLAM},
        {"THE_MAW_ROAR", MMID::THE_MAW_ROAR},
        {"THE_MAW_DROOL", MMID::THE_MAW_DROOL},
        {"THE_MAW_SLAM", MMID::THE_MAW_SLAM},
        {"THE_MAW_NOM", MMID::THE_MAW_NOM},
        {"TIME_EATER_REVERBERATE", MMID::TIME_EATER_REVERBERATE},
        {"TIME_EATER_HEAD_SLAM", MMID::TIME_EATER_HEAD_SLAM},
        {"TIME_EATER_RIPPLE", MMID::TIME_EATER_RIPPLE},
        {"TIME_EATER_HASTE", MMID::TIME_EATER_HASTE},
        {"TRANSIENT_ATTACK", MMID::TRANSIENT_ATTACK},
        {"WRITHING_MASS_IMPLANT", MMID::WRITHING_MASS_IMPLANT},
        {"WRITHING_MASS_FLAIL", MMID::WRITHING_MASS_FLAIL},
        {"WRITHING_MASS_WITHER", MMID::WRITHING_MASS_WITHER},
        {"WRITHING_MASS_MULTI_STRIKE", MMID::WRITHING_MASS_MULTI_STRIKE},
        {"WRITHING_MASS_STRONG_STRIKE", MMID::WRITHING_MASS_STRONG_STRIKE}
        };
        auto it = nameMap.find(name);
        return it == nameMap.end() ? MonsterMoveId::INVALID : it->second;
    }

    CardId lookupCardIdByStringId(const std::string &stringId) {
        static const std::unordered_map<std::string, CardId> nameMap = []() {
            std::unordered_map<std::string, CardId> m;
            for (int i = 0; i < static_cast<int>(sizeof(cardStringIds) / sizeof(char*)); ++i) {
                m[cardStringIds[i]] = static_cast<CardId>(i);
            }
            return m;
        }();
        auto it = nameMap.find(stringId);
        if (it == nameMap.end()) {
            throw std::runtime_error("lookupCardIdByStringId: unrecognized card_string_id: " + stringId);
        }
        return it->second;
    }

    MonsterId lookupMonsterIdByName(const std::string &name) {
        static const std::unordered_map<std::string, MonsterId> nameMap = []() {
            std::unordered_map<std::string, MonsterId> m;
            for (int i = 0; i < static_cast<int>(sizeof(monsterIdStrings) / sizeof(char*)); ++i) {
                m[monsterIdStrings[i]] = static_cast<MonsterId>(i);
            }
            return m;
        }();
        auto it = nameMap.find(name);
        if (it == nameMap.end()) {
            throw std::runtime_error("lookupMonsterIdByName: unrecognized monster id name: " + name);
        }
        return it->second;
    }

    // Per-monster explicit-state spec: everything nativeBuildBattleContext needs to
    // reconstruct one live monster exactly (see its own comment for the overall design).
    struct NativeMonsterSpec {
        std::string monsterIdName;  // canonical MonsterId enum-name, e.g. "GREMLIN_NOB"
        int curHp = 0;
        int maxHp = 0;
        int block = 0;
        bool halfDead = false;
        std::vector<std::pair<std::string, int>> statuses;  // (canonical MonsterStatus enum-name, amount)
        std::string moveName;  // canonical MonsterMoveId enum-name, or "" if unmapped/unknown
    };

    // Every RelicId with ordinal < 128, dispatched at runtime to the correct compile-
    // time template instantiation of Player::setHasRelic<r>(true) -- same pattern as
    // setPlayerStatusByName/setMonsterStatusByName above. RelicId is already a bound
    // pybind11 enum (unlike PlayerStatus/MonsterStatus/CardId/MonsterId), so
    // nativeBuildBattleContext takes a std::vector<RelicId> directly -- no string-name
    // lookup needed here, the Python bridge does that translation.
    //
    // PRE-EXISTING ENGINE LIMITATION, not introduced by this function: Player::
    // relicBits0/relicBits1 (Player.h) are each a plain uint64_t, covering only
    // ordinals 0-127, even though RelicId has 181 values -- setHasRelic<r>/
    // hasRelicRuntime(r) both shift a uint64_t by (int)r-64 for r>=64, which is
    // undefined behavior once (int)r-64 reaches 64+ (i.e. r>=128). Relics at
    // ordinal>=128 (includes VAJRA, a real combat-relevant relic: +1 Strength every
    // fight) are skipped here with a warning rather than invoking that UB. Fixing
    // this properly needs a THIRD relicBits word (or a wider bitset type) in
    // Player.h itself, touching every read/write site -- a deliberate, separate
    // change, not bundled into this bridge fix.
    //
    // KNOWN LIMITATION (separate from the above): this sets only relic PRESENCE, not
    // any relic-specific runtime counter/charge state accumulated over the run so far
    // (see test_relics.py for which relics this engine models with real mutable
    // state, e.g. Bronze Scales/Counter Relic). A freshly-granted relic and a
    // mid-charge one are indistinguishable here -- a real, bounded approximation,
    // improvable once a specific relic's counter behavior is confirmed to matter
    // against live capture data, same as the monster-move-name gap.
    void setPlayerRelicByEnum(Player &player, RelicId r) {
        if (static_cast<int>(r) >= 128) {
            return;  // see this function's own comment -- ordinal>=128 is unsafe (UB) in this engine
        }
        switch (r) {
            case RelicId::AKABEKO: player.setHasRelic<RelicId::AKABEKO>(true); break;
            case RelicId::ART_OF_WAR: player.setHasRelic<RelicId::ART_OF_WAR>(true); break;
            case RelicId::BIRD_FACED_URN: player.setHasRelic<RelicId::BIRD_FACED_URN>(true); break;
            case RelicId::BLOODY_IDOL: player.setHasRelic<RelicId::BLOODY_IDOL>(true); break;
            case RelicId::BLUE_CANDLE: player.setHasRelic<RelicId::BLUE_CANDLE>(true); break;
            case RelicId::BRIMSTONE: player.setHasRelic<RelicId::BRIMSTONE>(true); break;
            case RelicId::CALIPERS: player.setHasRelic<RelicId::CALIPERS>(true); break;
            case RelicId::CAPTAINS_WHEEL: player.setHasRelic<RelicId::CAPTAINS_WHEEL>(true); break;
            case RelicId::CENTENNIAL_PUZZLE: player.setHasRelic<RelicId::CENTENNIAL_PUZZLE>(true); break;
            case RelicId::CERAMIC_FISH: player.setHasRelic<RelicId::CERAMIC_FISH>(true); break;
            case RelicId::CHAMPION_BELT: player.setHasRelic<RelicId::CHAMPION_BELT>(true); break;
            case RelicId::CHARONS_ASHES: player.setHasRelic<RelicId::CHARONS_ASHES>(true); break;
            case RelicId::CHEMICAL_X: player.setHasRelic<RelicId::CHEMICAL_X>(true); break;
            case RelicId::CLOAK_CLASP: player.setHasRelic<RelicId::CLOAK_CLASP>(true); break;
            case RelicId::DARKSTONE_PERIAPT: player.setHasRelic<RelicId::DARKSTONE_PERIAPT>(true); break;
            case RelicId::DEAD_BRANCH: player.setHasRelic<RelicId::DEAD_BRANCH>(true); break;
            case RelicId::DUALITY: player.setHasRelic<RelicId::DUALITY>(true); break;
            case RelicId::ECTOPLASM: player.setHasRelic<RelicId::ECTOPLASM>(true); break;
            case RelicId::EMOTION_CHIP: player.setHasRelic<RelicId::EMOTION_CHIP>(true); break;
            case RelicId::FROZEN_CORE: player.setHasRelic<RelicId::FROZEN_CORE>(true); break;
            case RelicId::FROZEN_EYE: player.setHasRelic<RelicId::FROZEN_EYE>(true); break;
            case RelicId::GAMBLING_CHIP: player.setHasRelic<RelicId::GAMBLING_CHIP>(true); break;
            case RelicId::GINGER: player.setHasRelic<RelicId::GINGER>(true); break;
            case RelicId::GOLDEN_EYE: player.setHasRelic<RelicId::GOLDEN_EYE>(true); break;
            case RelicId::GREMLIN_HORN: player.setHasRelic<RelicId::GREMLIN_HORN>(true); break;
            case RelicId::HAND_DRILL: player.setHasRelic<RelicId::HAND_DRILL>(true); break;
            case RelicId::HAPPY_FLOWER: player.setHasRelic<RelicId::HAPPY_FLOWER>(true); break;
            case RelicId::HORN_CLEAT: player.setHasRelic<RelicId::HORN_CLEAT>(true); break;
            case RelicId::HOVERING_KITE: player.setHasRelic<RelicId::HOVERING_KITE>(true); break;
            case RelicId::ICE_CREAM: player.setHasRelic<RelicId::ICE_CREAM>(true); break;
            case RelicId::INCENSE_BURNER: player.setHasRelic<RelicId::INCENSE_BURNER>(true); break;
            case RelicId::INK_BOTTLE: player.setHasRelic<RelicId::INK_BOTTLE>(true); break;
            case RelicId::INSERTER: player.setHasRelic<RelicId::INSERTER>(true); break;
            case RelicId::KUNAI: player.setHasRelic<RelicId::KUNAI>(true); break;
            case RelicId::LETTER_OPENER: player.setHasRelic<RelicId::LETTER_OPENER>(true); break;
            case RelicId::LIZARD_TAIL: player.setHasRelic<RelicId::LIZARD_TAIL>(true); break;
            case RelicId::MAGIC_FLOWER: player.setHasRelic<RelicId::MAGIC_FLOWER>(true); break;
            case RelicId::MARK_OF_THE_BLOOM: player.setHasRelic<RelicId::MARK_OF_THE_BLOOM>(true); break;
            case RelicId::MEDICAL_KIT: player.setHasRelic<RelicId::MEDICAL_KIT>(true); break;
            case RelicId::MELANGE: player.setHasRelic<RelicId::MELANGE>(true); break;
            case RelicId::MERCURY_HOURGLASS: player.setHasRelic<RelicId::MERCURY_HOURGLASS>(true); break;
            case RelicId::MUMMIFIED_HAND: player.setHasRelic<RelicId::MUMMIFIED_HAND>(true); break;
            case RelicId::NECRONOMICON: player.setHasRelic<RelicId::NECRONOMICON>(true); break;
            case RelicId::NILRYS_CODEX: player.setHasRelic<RelicId::NILRYS_CODEX>(true); break;
            case RelicId::NUNCHAKU: player.setHasRelic<RelicId::NUNCHAKU>(true); break;
            case RelicId::ODD_MUSHROOM: player.setHasRelic<RelicId::ODD_MUSHROOM>(true); break;
            case RelicId::OMAMORI: player.setHasRelic<RelicId::OMAMORI>(true); break;
            case RelicId::ORANGE_PELLETS: player.setHasRelic<RelicId::ORANGE_PELLETS>(true); break;
            case RelicId::ORICHALCUM: player.setHasRelic<RelicId::ORICHALCUM>(true); break;
            case RelicId::ORNAMENTAL_FAN: player.setHasRelic<RelicId::ORNAMENTAL_FAN>(true); break;
            case RelicId::PAPER_KRANE: player.setHasRelic<RelicId::PAPER_KRANE>(true); break;
            case RelicId::PAPER_PHROG: player.setHasRelic<RelicId::PAPER_PHROG>(true); break;
            case RelicId::PEN_NIB: player.setHasRelic<RelicId::PEN_NIB>(true); break;
            case RelicId::PHILOSOPHERS_STONE: player.setHasRelic<RelicId::PHILOSOPHERS_STONE>(true); break;
            case RelicId::POCKETWATCH: player.setHasRelic<RelicId::POCKETWATCH>(true); break;
            case RelicId::RED_SKULL: player.setHasRelic<RelicId::RED_SKULL>(true); break;
            case RelicId::RUNIC_CUBE: player.setHasRelic<RelicId::RUNIC_CUBE>(true); break;
            case RelicId::RUNIC_DOME: player.setHasRelic<RelicId::RUNIC_DOME>(true); break;
            case RelicId::RUNIC_PYRAMID: player.setHasRelic<RelicId::RUNIC_PYRAMID>(true); break;
            case RelicId::SACRED_BARK: player.setHasRelic<RelicId::SACRED_BARK>(true); break;
            case RelicId::SELF_FORMING_CLAY: player.setHasRelic<RelicId::SELF_FORMING_CLAY>(true); break;
            case RelicId::SHURIKEN: player.setHasRelic<RelicId::SHURIKEN>(true); break;
            case RelicId::SNECKO_EYE: player.setHasRelic<RelicId::SNECKO_EYE>(true); break;
            case RelicId::SNECKO_SKULL: player.setHasRelic<RelicId::SNECKO_SKULL>(true); break;
            case RelicId::SOZU: player.setHasRelic<RelicId::SOZU>(true); break;
            case RelicId::STONE_CALENDAR: player.setHasRelic<RelicId::STONE_CALENDAR>(true); break;
            case RelicId::STRANGE_SPOON: player.setHasRelic<RelicId::STRANGE_SPOON>(true); break;
            case RelicId::STRIKE_DUMMY: player.setHasRelic<RelicId::STRIKE_DUMMY>(true); break;
            case RelicId::SUNDIAL: player.setHasRelic<RelicId::SUNDIAL>(true); break;
            case RelicId::THE_ABACUS: player.setHasRelic<RelicId::THE_ABACUS>(true); break;
            case RelicId::THE_BOOT: player.setHasRelic<RelicId::THE_BOOT>(true); break;
            case RelicId::THE_SPECIMEN: player.setHasRelic<RelicId::THE_SPECIMEN>(true); break;
            case RelicId::TINGSHA: player.setHasRelic<RelicId::TINGSHA>(true); break;
            case RelicId::TOOLBOX: player.setHasRelic<RelicId::TOOLBOX>(true); break;
            case RelicId::TORII: player.setHasRelic<RelicId::TORII>(true); break;
            case RelicId::TOUGH_BANDAGES: player.setHasRelic<RelicId::TOUGH_BANDAGES>(true); break;
            case RelicId::TOY_ORNITHOPTER: player.setHasRelic<RelicId::TOY_ORNITHOPTER>(true); break;
            case RelicId::TUNGSTEN_ROD: player.setHasRelic<RelicId::TUNGSTEN_ROD>(true); break;
            case RelicId::TURNIP: player.setHasRelic<RelicId::TURNIP>(true); break;
            case RelicId::TWISTED_FUNNEL: player.setHasRelic<RelicId::TWISTED_FUNNEL>(true); break;
            case RelicId::UNCEASING_TOP: player.setHasRelic<RelicId::UNCEASING_TOP>(true); break;
            case RelicId::VELVET_CHOKER: player.setHasRelic<RelicId::VELVET_CHOKER>(true); break;
            case RelicId::VIOLET_LOTUS: player.setHasRelic<RelicId::VIOLET_LOTUS>(true); break;
            case RelicId::WARPED_TONGS: player.setHasRelic<RelicId::WARPED_TONGS>(true); break;
            case RelicId::WRIST_BLADE: player.setHasRelic<RelicId::WRIST_BLADE>(true); break;
            case RelicId::BLACK_BLOOD: player.setHasRelic<RelicId::BLACK_BLOOD>(true); break;
            case RelicId::BURNING_BLOOD: player.setHasRelic<RelicId::BURNING_BLOOD>(true); break;
            case RelicId::MEAT_ON_THE_BONE: player.setHasRelic<RelicId::MEAT_ON_THE_BONE>(true); break;
            case RelicId::FACE_OF_CLERIC: player.setHasRelic<RelicId::FACE_OF_CLERIC>(true); break;
            case RelicId::ANCHOR: player.setHasRelic<RelicId::ANCHOR>(true); break;
            case RelicId::ANCIENT_TEA_SET: player.setHasRelic<RelicId::ANCIENT_TEA_SET>(true); break;
            case RelicId::BAG_OF_MARBLES: player.setHasRelic<RelicId::BAG_OF_MARBLES>(true); break;
            case RelicId::BAG_OF_PREPARATION: player.setHasRelic<RelicId::BAG_OF_PREPARATION>(true); break;
            case RelicId::BLOOD_VIAL: player.setHasRelic<RelicId::BLOOD_VIAL>(true); break;
            case RelicId::BOTTLED_FLAME: player.setHasRelic<RelicId::BOTTLED_FLAME>(true); break;
            case RelicId::BOTTLED_LIGHTNING: player.setHasRelic<RelicId::BOTTLED_LIGHTNING>(true); break;
            case RelicId::BOTTLED_TORNADO: player.setHasRelic<RelicId::BOTTLED_TORNADO>(true); break;
            case RelicId::BRONZE_SCALES: player.setHasRelic<RelicId::BRONZE_SCALES>(true); break;
            case RelicId::BUSTED_CROWN: player.setHasRelic<RelicId::BUSTED_CROWN>(true); break;
            case RelicId::CLOCKWORK_SOUVENIR: player.setHasRelic<RelicId::CLOCKWORK_SOUVENIR>(true); break;
            case RelicId::COFFEE_DRIPPER: player.setHasRelic<RelicId::COFFEE_DRIPPER>(true); break;
            case RelicId::CRACKED_CORE: player.setHasRelic<RelicId::CRACKED_CORE>(true); break;
            case RelicId::CURSED_KEY: player.setHasRelic<RelicId::CURSED_KEY>(true); break;
            case RelicId::DAMARU: player.setHasRelic<RelicId::DAMARU>(true); break;
            case RelicId::DATA_DISK: player.setHasRelic<RelicId::DATA_DISK>(true); break;
            case RelicId::DU_VU_DOLL: player.setHasRelic<RelicId::DU_VU_DOLL>(true); break;
            case RelicId::ENCHIRIDION: player.setHasRelic<RelicId::ENCHIRIDION>(true); break;
            case RelicId::FOSSILIZED_HELIX: player.setHasRelic<RelicId::FOSSILIZED_HELIX>(true); break;
            case RelicId::FUSION_HAMMER: player.setHasRelic<RelicId::FUSION_HAMMER>(true); break;
            case RelicId::GIRYA: player.setHasRelic<RelicId::GIRYA>(true); break;
            case RelicId::GOLD_PLATED_CABLES: player.setHasRelic<RelicId::GOLD_PLATED_CABLES>(true); break;
            case RelicId::GREMLIN_VISAGE: player.setHasRelic<RelicId::GREMLIN_VISAGE>(true); break;
            case RelicId::HOLY_WATER: player.setHasRelic<RelicId::HOLY_WATER>(true); break;
            case RelicId::LANTERN: player.setHasRelic<RelicId::LANTERN>(true); break;
            case RelicId::MARK_OF_PAIN: player.setHasRelic<RelicId::MARK_OF_PAIN>(true); break;
            case RelicId::MUTAGENIC_STRENGTH: player.setHasRelic<RelicId::MUTAGENIC_STRENGTH>(true); break;
            case RelicId::NEOWS_LAMENT: player.setHasRelic<RelicId::NEOWS_LAMENT>(true); break;
            case RelicId::NINJA_SCROLL: player.setHasRelic<RelicId::NINJA_SCROLL>(true); break;
            case RelicId::NUCLEAR_BATTERY: player.setHasRelic<RelicId::NUCLEAR_BATTERY>(true); break;
            case RelicId::ODDLY_SMOOTH_STONE: player.setHasRelic<RelicId::ODDLY_SMOOTH_STONE>(true); break;
            case RelicId::PANTOGRAPH: player.setHasRelic<RelicId::PANTOGRAPH>(true); break;
            case RelicId::PRESERVED_INSECT: player.setHasRelic<RelicId::PRESERVED_INSECT>(true); break;
            case RelicId::PURE_WATER: player.setHasRelic<RelicId::PURE_WATER>(true); break;
            case RelicId::RED_MASK: player.setHasRelic<RelicId::RED_MASK>(true); break;
            case RelicId::RING_OF_THE_SERPENT: player.setHasRelic<RelicId::RING_OF_THE_SERPENT>(true); break;
            case RelicId::RING_OF_THE_SNAKE: player.setHasRelic<RelicId::RING_OF_THE_SNAKE>(true); break;
            case RelicId::RUNIC_CAPACITOR: player.setHasRelic<RelicId::RUNIC_CAPACITOR>(true); break;
            case RelicId::SLAVERS_COLLAR: player.setHasRelic<RelicId::SLAVERS_COLLAR>(true); break;
            default: break;
        }
    }

    // (card_string_id, upgrade_count) -- an int count, not a bool, specifically so
    // Searing Blow's multi-upgrade damage scaling (specialData) can be reconstructed
    // exactly, matching CardInstance's own Card-based constructor special-case below.
    CardInstance buildCardInstance(const std::string &stringId, int upgradeCount) {
        const CardId id = lookupCardIdByStringId(stringId);
        CardInstance ci(id, upgradeCount > 0);
        if (id == CardId::SEARING_BLOW) {
            ci.specialData = static_cast<std::int16_t>(upgradeCount);
        }
        return ci;
    }

    BattleContext nativeBuildBattleContext(
            int playerHp, int playerMaxHp, int playerBlock, int playerEnergy,
            std::vector<std::pair<std::string, int>> playerStatuses,
            std::vector<NativeMonsterSpec> monsters,
            std::vector<std::pair<std::string, int>> handCards,
            std::vector<std::pair<std::string, int>> drawPileCards,
            std::vector<std::pair<std::string, int>> discardPileCards,
            std::vector<std::pair<std::string, int>> exhaustPileCards,
            std::vector<Potion> potionSlots, std::vector<RelicId> relics,
            int turn, int ascension, std::uint64_t rngSeed) {
        BattleContext bc;
        bc.outcome = Outcome::UNDECIDED;
        bc.inputState = InputState::PLAYER_NORMAL;
        bc.turn = turn;
        bc.ascension = ascension;
        nativeSeedRng(bc, rngSeed);  // synthetic -- the real game's RNG state isn't observable
                               // through spirecomm; this only affects OUR OWN hypothetical
                               // future exploration during search, not the already-known
                               // current state being reconstructed here.

        // Potion is already a bound pybind11 enum (unlike PlayerStatus/MonsterStatus/CardId/
        // MonsterId), so no string-name lookup is needed here -- the Python bridge passes
        // sts.Potion values directly, translated from spirecomm's own potion_id strings on
        // the Python side. getLegalActions reads bc.potions[i] PER SLOT (not dense-packed --
        // confirmed in bindings-util.cpp: EMPTY_POTION_SLOT/INVALID at a slot means "skip",
        // any other value means "real potion here"), so slot i here must correspond exactly
        // to spirecomm's potion slot i, empty slots included.
        bc.potionCapacity = std::min(static_cast<int>(potionSlots.size()), static_cast<int>(bc.potions.size()));
        bc.potionCount = 0;
        for (int i = 0; i < bc.potionCapacity; ++i) {
            bc.potions[i] = potionSlots[i];
            if (potionSlots[i] != Potion::INVALID && potionSlots[i] != Potion::EMPTY_POTION_SLOT) {
                ++bc.potionCount;
            }
        }

        bc.player.curHp = playerHp;
        bc.player.maxHp = playerMaxHp;
        bc.player.block = playerBlock;
        bc.player.energy = playerEnergy;
        for (const auto &[name, amount] : playerStatuses) {
            setPlayerStatusByName(bc.player, name, amount);
        }
        // Applied AFTER curHp/maxHp above (not before, unlike new_battle()'s own
        // gc.obtain_relic()-then-hp-override flow) -- here playerMaxHp is already the
        // real, live-reported max HP (already reflecting any relic-granted HP bumps
        // like Mango/Strawberry/Pear from the ACTUAL game history), so there is no
        // "override must win over relic-granted HP" ordering concern the way
        // new_battle's own tier-calibrated-HP-assignment comment describes -- see
        // setPlayerRelicByEnum's own comment for what this does and doesn't model.
        for (RelicId r : relics) {
            setPlayerRelicByEnum(bc.player, r);
        }

        bc.monsters.monsterCount = static_cast<int>(monsters.size());
        bc.monsters.monstersAlive = 0;
        for (int i = 0; i < static_cast<int>(monsters.size()); ++i) {
            const auto &spec = monsters[i];
            Monster &m = bc.monsters.arr[i];
            m = Monster();
            m.id = lookupMonsterIdByName(spec.monsterIdName);
            m.idx = i;
            m.curHp = spec.curHp;
            m.maxHp = spec.maxHp;
            m.block = spec.block;
            m.halfDead = spec.halfDead;
            for (const auto &[name, amount] : spec.statuses) {
                setMonsterStatusByName(m, name, amount);
            }
            if (!spec.moveName.empty()) {
                const MonsterMoveId mv = lookupMonsterMoveId(spec.moveName);
                if (mv != MonsterMoveId::INVALID) {
                    m.moveHistory[1] = m.moveHistory[0];
                    m.moveHistory[0] = mv;
                }
                // unrecognized move name -- moveHistory stays INVALID here; the
                // fallback pass below rolls a real move for it (see that pass's
                // own comment for why INVALID can't just be left as-is).
            }
            if (spec.curHp > 0 && !spec.halfDead) {
                ++bc.monsters.monstersAlive;
            }
        }

        // Any monster whose move wasn't set above (empty/unmapped spirecomm move
        // name) still needs a REAL move -- unlike a fresh new_battle() fight, where
        // every monster gets rollMove() called immediately after spawning, so
        // moveHistory[0] is NEVER actually INVALID during normal play. Downstream
        // code (e.g. MonsterSpecific.cpp's move-damage switches) assumes this
        // invariant and asserts/UB's on a genuinely-INVALID move rather than
        // handling it gracefully -- confirmed by a real assertion failure hit
        // while building this function. Roll a plausible move via the engine's
        // own AI model instead, the same mechanism already used for every FUTURE
        // (not-yet-happened) monster turn our own search explores -- an honest,
        // bounded approximation for a genuinely unmapped CURRENT move, not a new
        // source of unreliability beyond what search already has for turn 2+.
        //
        // TORCH_HEAD is a special case: it has no AI move-roll logic at all
        // (MonsterSpecific.cpp's getMoveForRoll has an explicit empty case for it,
        // "// setting in collector spawn move" -- it's a minion spawned mid-fight
        // by The Collector with its move always set directly to
        // MMID::TORCH_HEAD_TACKLE, never rolled). Calling rollMove() on it hits
        // getMoveForRoll's unconditional assert(false) fallthrough -- confirmed via
        // a real crash replaying live-captured data from a Collector fight. Since
        // it only ever has this one move, set it directly instead of rolling.
        for (int i = 0; i < static_cast<int>(monsters.size()); ++i) {
            Monster &m = bc.monsters.arr[i];
            if (m.moveHistory[0] == MonsterMoveId::INVALID && m.curHp > 0 && !m.halfDead) {
                if (m.id == MonsterId::TORCH_HEAD) {
                    m.moveHistory[1] = m.moveHistory[0];
                    m.moveHistory[0] = MonsterMoveId::TORCH_HEAD_TACKLE;
                } else {
                    m.rollMove(bc);
                }
            }
        }

        // Every card needs a distinct uniqueId. BattleContext::useCard removes the
        // played card from hand via CardManager::removeFromHandById(c.uniqueId),
        // matching against each hand card's own id -- so cards left at
        // CardInstance's -1 default are never removed and can be replayed without
        // limit. Numbering starts at 0 and runs across all four piles, matching
        // what the normal GameContext->new_battle path produces.
        std::int16_t nextUniqueId = 0;
        auto buildPile = [&nextUniqueId](const std::vector<std::pair<std::string, int>> &cards) {
            std::vector<CardInstance> pile;
            pile.reserve(cards.size());
            for (const auto &[stringId, upgradeCount] : cards) {
                CardInstance card = buildCardInstance(stringId, upgradeCount);
                card.uniqueId = nextUniqueId++;
                pile.push_back(card);
            }
            return pile;
        };
        bc.cards.drawPile = buildPile(drawPileCards);
        bc.cards.discardPile = buildPile(discardPileCards);
        bc.cards.exhaustPile = buildPile(exhaustPileCards);
        bc.cards.cardsInHand = static_cast<int>(handCards.size());
        for (int i = 0; i < static_cast<int>(handCards.size()) && i < CardManager::MAX_HAND_SIZE; ++i) {
            bc.cards.hand[i] = buildCardInstance(handCards[i].first, handCards[i].second);
            bc.cards.hand[i].uniqueId = nextUniqueId++;
        }
        bc.cards.nextUniqueCardId = nextUniqueId;

        return bc;
    }
}

PYBIND11_MODULE(slaythespire, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    // Heart1's input spaces size their embedding tables with len(Enum).  Match
    // Silverbot's extended enum metaclass without changing enum ordinals.
    auto &internals = pybind11::detail::get_internals();
    auto pybind11_metaclass = pybind11::reinterpret_borrow<pybind11::object>((PyObject*)internals.default_metaclass);
    auto standard_metaclass = pybind11::reinterpret_borrow<pybind11::object>((PyObject *)&PyType_Type);
    pybind11::dict enum_attributes;
    enum_attributes["__len__"] = pybind11::cpp_function(
        [](pybind11::object cls) { return pybind11::len(cls.attr("__entries")); },
        pybind11::is_method(pybind11::none()));
    auto enum_metaclass = standard_metaclass(std::string("pybind11_ext_enum"),
        pybind11::make_tuple(pybind11_metaclass), enum_attributes);
    m.def("play", &sts::py::play, "play Slay the Spire Console");
    m.def("get_seed_str", &SeedHelper::getString, "gets the integral representation of seed string used in the game ui");
    m.def("get_seed_long", &SeedHelper::getLong, "gets the seed string representation of an integral seed");
    m.def("getNNInterface", &sts::NNInterface::getInstance, "gets the NNInterface object");
    m.def("getFixedObservation", &sts::py::getFixedObservation);
    m.def("getFixedObservationMaximums", &sts::py::getFixedObservationMaximums);
    m.def("getNNRepresentation", &sts::py::getNNRepresentation,
          "Heart1-compatible structured overworld observation");
    m.attr("MAX_POTION_CAPACITY") = 5;
    m.def("get_card_color", [](CardId id) { return cardColors[static_cast<int>(id)]; },
          "gets the CardColor (class ownership) of a CardId");
    m.def("get_card_type", [](CardId id) { return cardTypes[static_cast<int>(id)]; },
          "gets the CardType of a CardId -- works for ANY card id including Status/Curse "
          "cards outside the 75-card Ironclad pool (e.g. Wound/Dazed/Slimed/Void shuffled "
          "in mid-fight), unlike cards.py's card_type() which only looks up the pool");
    m.def("is_boss_relic", [](RelicId id) { return getRelicTier(id) == RelicTier::BOSS; },
          "true if this relic is BOSS-rarity (the 3-choice pool offered after killing an "
          "act boss), using the engine's own authoritative relicTiers[] table rather than "
          "a hand-maintained list -- for callers that need to guarantee a real run's boss-"
          "relic count (1 by Act 2, 2 by Act 3) rather than relying on general-pool "
          "frequency sampling to include one by chance");

    pybind11::class_<NNInterface> nnInterface(m, "NNInterface");
    nnInterface.def("getObservation", &NNInterface::getObservation, "get observation array given a GameContext")
        .def("getObservationMaximums", &NNInterface::getObservationMaximums, "get the defined maximum values of the observation space")
        .def_property_readonly("observation_space_size", []() { return NNInterface::observation_space_size; });

    pybind11::class_<sts::py::NNCardsRepresentation>(m, "NNCardRepresentation")
        .def_readwrite("cards", &sts::py::NNCardsRepresentation::cards)
        .def_readwrite("upgrades", &sts::py::NNCardsRepresentation::upgrades)
        .def("as_dict", &sts::py::NNCardsRepresentation::as_dict);
    pybind11::class_<sts::py::NNRelicsRepresentation>(m, "NNRelicRepresentation")
        .def_readwrite("relics", &sts::py::NNRelicsRepresentation::relics)
        .def_readwrite("relic_counters", &sts::py::NNRelicsRepresentation::relicCounters)
        .def("as_dict", &sts::py::NNRelicsRepresentation::as_dict);
    pybind11::class_<sts::py::NNMapRepresentation>(m, "NNMapRepresentation")
        .def_readwrite("xs", &sts::py::NNMapRepresentation::xs)
        .def_readwrite("ys", &sts::py::NNMapRepresentation::ys)
        .def_readwrite("room_types", &sts::py::NNMapRepresentation::roomTypes)
        .def_readwrite("path_xs", &sts::py::NNMapRepresentation::pathXs)
        .def_readonly("burning_elite_x", &sts::py::NNMapRepresentation::burningEliteX)
        .def_readonly("burning_elite_y", &sts::py::NNMapRepresentation::burningEliteY)
        .def("as_dict", &sts::py::NNMapRepresentation::as_dict);
    pybind11::class_<sts::py::NNRepresentation>(m, "NNRepresentation")
        .def_readwrite("fixed_observation", &sts::py::NNRepresentation::fixedObservation)
        .def_readwrite("deck", &sts::py::NNRepresentation::deck)
        .def_readwrite("relics", &sts::py::NNRepresentation::relics)
        .def_readwrite("potions", &sts::py::NNRepresentation::potions)
        .def_readwrite("map", &sts::py::NNRepresentation::map)
        .def_readwrite("mapX", &sts::py::NNRepresentation::mapX)
        .def_readwrite("mapY", &sts::py::NNRepresentation::mapY)
        .def("as_dict", &sts::py::NNRepresentation::as_dict);

    pybind11::class_<search::ScumSearchAgent2> agent(m, "Agent");
    agent.def(pybind11::init<>());
    agent.def_readwrite("simulation_count_base", &search::ScumSearchAgent2::simulationCountBase, "number of simulations the agent uses for monte carlo tree search each turn")
        .def_readwrite("boss_simulation_multiplier", &search::ScumSearchAgent2::bossSimulationMultiplier, "bonus multiplier to the simulation count for boss fights")
        .def_readwrite("pause_on_card_reward", &search::ScumSearchAgent2::pauseOnCardReward, "causes the agent to pause so as to cede control to the user when it encounters a card reward choice")
        .def_readwrite("record_actions", &search::ScumSearchAgent2::recordActions)
        .def_readwrite("print_logs", &search::ScumSearchAgent2::printLogs, "when set to true, the agent prints state information as it makes actions")
        .def("playout", &search::ScumSearchAgent2::playout)
        .def("playout_battle", &search::ScumSearchAgent2::playoutBattle,
             "play a single isolated BattleContext (e.g. from new_battle) to completion via MCTS, mutating it in place")
        .def("step_out_of_combat_policy", &search::ScumSearchAgent2::stepOutOfCombatPolicy,
             "Execute one established heuristic overworld decision; used only as a behavior-cloning teacher")
        .def_property_readonly("game_action_history", [](const search::ScumSearchAgent2 &self) {
            return self.gameActionHistory;
        })
        .def("playout_hybrid", [](search::ScumSearchAgent2 &self, GameContext &gc, int nSimulations) {
            pybind11::gil_scoped_release release;
            nativeHybridPlayout(self, gc, nSimulations);
        }, pybind11::arg("gc"), pybind11::arg("n_simulations") = 200,
           "Play a FULL RUN to completion (win or death), mutating gc in place. Combat uses "
           "our own native MCTS (run_mcts_search/set_search_params, n_simulations per decision); "
           "every non-combat decision (map, shop, rest, events, card rewards) uses this agent's "
           "OWN stepOutOfCombatPolicy unchanged -- see nativeHybridPlayout's own comment for why. "
           "Requires pause_on_card_reward == false (the default) or this can spin forever.");

    pybind11::class_<GameContext> gameContext(m, "GameContext");
    gameContext.def(pybind11::init<CharacterClass, std::uint64_t, int>())
        .def("pick_reward_card", &sts::py::pickRewardCard, "choose to obtain the card at the specified index in the card reward list")
        .def("skip_reward_cards", &sts::py::skipRewardCards, "choose to skip the card reward (increases max_hp by 2 with singing bowl)")
        .def("get_card_reward", &sts::py::getCardReward, "return the current card reward list")
        .def_property_readonly("encounter", [](const GameContext &gc) { return gc.info.encounter; })
        .def_property_readonly("deck",
               [](const GameContext &gc) { return std::vector(gc.deck.cards.begin(), gc.deck.cards.end());},
               "returns a copy of the list of cards in the deck"
        )
        .def("obtain_card",
             [](GameContext &gc, Card card) { gc.deck.obtain(gc, card); },
             "add a card to the deck"
        )
        .def("obtain_relic",
             [](GameContext &gc, RelicId r) { gc.obtainRelic(r); },
             "add a relic before combat starts (mirrors obtain_card) -- GameContext::obtainRelic "
             "already exists in the engine, this was just never exposed to Python before"
        )
        .def("obtain_potion",
             [](GameContext &gc, Potion p) { gc.obtainPotion(p); },
             "add a potion before combat starts (mirrors obtain_relic) -- GameContext::obtainPotion "
             "already exists in the engine, this was just never exposed to Python before"
        )
        .def("remove_card",
            [](GameContext &gc, int idx) {
                if (idx < 0 || idx >= gc.deck.size()) {
                    std::cerr << "invalid remove deck remove idx" << std::endl;
                    return;
                }
                gc.deck.remove(gc, idx);
            },
             "remove a card at a idx in the deck"
        )
        .def("copy", [](const GameContext &gc) {
            GameContext copy(gc);
            // Map is normally immutable within an act, but act transition
            // assigns a newly generated map through *map. Sharing the pointer
            // therefore lets one long counterfactual overwrite its parent and
            // every sibling branch. Planning copies must own their map.
            if (gc.map) {
                copy.map = std::make_shared<Map>(*gc.map);
            }
            return copy;
        }, "deep value copy for counterfactual whole-run planning, including an independent map")
        .def_property_readonly("relics",
               [] (const GameContext &gc) { return std::vector(gc.relics.relics); },
               "returns a copy of the list of relics"
        )
        .def("__repr__", [](const GameContext &gc) {
            std::ostringstream oss;
            oss << "<" << gc << ">";
            return oss.str();
        }, "returns a string representation of the GameContext");

    gameContext.def_readwrite("outcome", &GameContext::outcome)
        .def_readwrite("act", &GameContext::act)
        .def_readwrite("ascension", &GameContext::ascension)
        .def_readwrite("floor_num", &GameContext::floorNum)
        .def_readwrite("screen_state", &GameContext::screenState)

        .def_readwrite("seed", &GameContext::seed)
        .def_readwrite("cur_map_node_x", &GameContext::curMapNodeX)
        .def_readwrite("cur_map_node_y", &GameContext::curMapNodeY)
        // The engine has always known which elite is burning
        // (BattleContext.cpp:69 applies the buff by comparing these; the
        // emerald key is awarded the same way at GameContext.cpp:1959) but
        // never exposed it, so neither a rule nor the policy could seek the
        // green key -- acquisition was pure stumble-and-take.
        .def_property_readonly("burning_elite_x", [](const GameContext &gc) {
            return gc.map ? gc.map->burningEliteX : -1; })
        .def_property_readonly("burning_elite_y", [](const GameContext &gc) {
            return gc.map ? gc.map->burningEliteY : -1; })
        .def_readwrite("cur_room", &GameContext::curRoom)
        .def_readwrite("cur_event", &GameContext::curEvent)
        .def_readwrite("boss", &GameContext::boss)

        .def_readwrite("cur_hp", &GameContext::curHp)
        .def_readwrite("max_hp", &GameContext::maxHp)
        .def_readwrite("gold", &GameContext::gold)

        .def_readwrite("blue_key", &GameContext::blueKey)
        .def_readwrite("green_key", &GameContext::greenKey)
        .def_readwrite("red_key", &GameContext::redKey)

        .def_readwrite("card_rarity_factor", &GameContext::cardRarityFactor)
        .def_readwrite("potion_chance", &GameContext::potionChance)
        .def_readwrite("monster_chance", &GameContext::monsterChance)
        .def_readwrite("shop_chance", &GameContext::shopChance)
        .def_readwrite("treasure_chance", &GameContext::treasureChance)

        .def_readwrite("shop_remove_count", &GameContext::shopRemoveCount)
        .def_readwrite("speedrun_pace", &GameContext::speedrunPace)
        .def_readwrite("note_for_yourself_card", &GameContext::noteForYourselfCard);

    // Heart1 only needs these public screen fields to turn its logits back
    // into GameActions.  The native run remains entirely in this engine.
    gameContext.def_readwrite("screen_state_info", &GameContext::info);

    pybind11::class_<Rewards>(m, "Rewards")
        .def_property_readonly("gold", [](const Rewards &r) {
            return std::vector<int>(r.gold.begin(), r.gold.begin() + r.goldRewardCount); })
        .def_property_readonly("cards", [](const Rewards &r) {
            std::vector<std::vector<Card>> ret;
            for (int i = 0; i < r.cardRewardCount; ++i)
                ret.emplace_back(r.cardRewards[i].begin(), r.cardRewards[i].end());
            return ret; })
        .def_property_readonly("relics", [](const Rewards &r) {
            return std::vector<RelicId>(r.relics.begin(), r.relics.begin() + r.relicCount); })
        .def_property_readonly("potions", [](const Rewards &r) {
            return std::vector<Potion>(r.potions.begin(), r.potions.begin() + r.potionCount); });

    pybind11::class_<Shop>(m, "Shop")
        .def_property_readonly("cards", [](const Shop &s) {
            return std::vector<Card>(s.cards, s.cards + 7); })
        .def_property_readonly("relics", [](const Shop &s) {
            return std::vector<RelicId>(s.relics, s.relics + 3); })
        .def_property_readonly("potions", [](const Shop &s) {
            return std::vector<Potion>(s.potions, s.potions + 3); })
        .def_property_readonly("prices", [](const Shop &s) {
            return std::vector<int>(s.prices, s.prices + 13); })
        .def_readonly("remove_cost", &Shop::removeCost);

    pybind11::class_<ScreenStateInfo>(m, "ScreenStateInfo")
        .def_property_readonly("boss_relics", [](const ScreenStateInfo &s) {
            return std::vector<RelicId>(s.bossRelics, s.bossRelics + 3); })
        .def_property_readonly("shop", [](const ScreenStateInfo &s) -> const Shop& { return s.shop; },
                               pybind11::return_value_policy::reference_internal)
        .def_property_readonly("to_select_cards", [](const ScreenStateInfo &s) {
            std::vector<Card> ret;
            for (const auto &entry : s.toSelectCards) ret.push_back(entry.card);
            return ret; })
        .def_readwrite("rewards_container", &ScreenStateInfo::rewardsContainer)
        .def_property_readonly("neowRewards", [](const ScreenStateInfo &s) {
            return std::vector<Neow::Option>(s.neowRewards.begin(), s.neowRewards.end()); })
        .def_readwrite("event_data", &ScreenStateInfo::eventData)
        .def_readwrite("select_screen_type", &ScreenStateInfo::selectScreenType)
        .def_readwrite("to_select_count", &ScreenStateInfo::toSelectCount)
        .def_readwrite("hpAmount0", &ScreenStateInfo::hpAmount0)
        .def_readwrite("hpAmount1", &ScreenStateInfo::hpAmount1)
        .def_readwrite("hpAmount2", &ScreenStateInfo::hpAmount2)
        .def_readwrite("goldLoss", &ScreenStateInfo::goldLoss)
        .def_readwrite("gold", &ScreenStateInfo::gold)
        .def_readwrite("cardIdx", &ScreenStateInfo::cardIdx)
        .def_readwrite("potionIdx", &ScreenStateInfo::potionIdx)
        .def_readwrite("relicIdx0", &ScreenStateInfo::relicIdx0)
        .def_readwrite("relicIdx1", &ScreenStateInfo::relicIdx1)
        .def_readwrite("skillCardDeckIdx", &ScreenStateInfo::skillCardDeckIdx)
        .def_readwrite("powerCardDeckIdx", &ScreenStateInfo::powerCardDeckIdx)
        .def_readwrite("attackCardDeckIdx", &ScreenStateInfo::attackCardDeckIdx);

    pybind11::class_<search::GameAction> gameAction(m, "GameAction");
    gameAction.def_static("getAllActionsInState", &search::GameAction::getAllActionsInState)
        .def(pybind11::init<std::uint32_t>())
        .def_readonly("bits", &search::GameAction::bits)
        .def_property_readonly("idx1", &search::GameAction::getIdx1)
        .def_property_readonly("idx2", &search::GameAction::getIdx2)
        .def_property_readonly("idx3", &search::GameAction::getIdx3)
        .def("execute", &search::GameAction::execute)
        .def("getDesc", [](const search::GameAction &a, const GameContext &gc) {
            std::ostringstream out; a.printDesc(out, gc); return out.str(); })
        .def("isValidAction", &search::GameAction::isValidAction)
        .def_property_readonly("rewards_action_type", &search::GameAction::getRewardsActionType)
        .def("__eq__", [](const search::GameAction &a, const search::GameAction &b) { return a.bits == b.bits; })
        .def("__hash__", [](const search::GameAction &a) { return std::hash<std::uint32_t>{}(a.bits); });
    pybind11::enum_<search::GameAction::RewardsActionType>(m, "RewardsActionType")
        .value("CARD", search::GameAction::RewardsActionType::CARD)
        .value("GOLD", search::GameAction::RewardsActionType::GOLD)
        .value("KEY", search::GameAction::RewardsActionType::KEY)
        .value("POTION", search::GameAction::RewardsActionType::POTION)
        .value("RELIC", search::GameAction::RewardsActionType::RELIC)
        .value("CARD_REMOVE", search::GameAction::RewardsActionType::CARD_REMOVE)
        .value("SKIP", search::GameAction::RewardsActionType::SKIP);
    pybind11::enum_<CardSelectScreenType>(m, "CardSelectScreenType", pybind11::metaclass(enum_metaclass))
        .value("INVALID", CardSelectScreenType::INVALID)
        .value("TRANSFORM", CardSelectScreenType::TRANSFORM)
        .value("TRANSFORM_UPGRADE", CardSelectScreenType::TRANSFORM_UPGRADE)
        .value("UPGRADE", CardSelectScreenType::UPGRADE)
        .value("REMOVE", CardSelectScreenType::REMOVE)
        .value("DUPLICATE", CardSelectScreenType::DUPLICATE)
        .value("OBTAIN", CardSelectScreenType::OBTAIN)
        .value("BOTTLE", CardSelectScreenType::BOTTLE)
        .value("BONFIRE_SPIRITS", CardSelectScreenType::BONFIRE_SPIRITS);
    pybind11::enum_<Neow::Bonus>(m, "NeowBonus")
        .value("THREE_CARDS", Neow::Bonus::THREE_CARDS).value("ONE_RANDOM_RARE_CARD", Neow::Bonus::ONE_RANDOM_RARE_CARD)
        .value("REMOVE_CARD", Neow::Bonus::REMOVE_CARD).value("UPGRADE_CARD", Neow::Bonus::UPGRADE_CARD)
        .value("TRANSFORM_CARD", Neow::Bonus::TRANSFORM_CARD).value("RANDOM_COLORLESS", Neow::Bonus::RANDOM_COLORLESS)
        .value("THREE_SMALL_POTIONS", Neow::Bonus::THREE_SMALL_POTIONS).value("RANDOM_COMMON_RELIC", Neow::Bonus::RANDOM_COMMON_RELIC)
        .value("TEN_PERCENT_HP_BONUS", Neow::Bonus::TEN_PERCENT_HP_BONUS).value("THREE_ENEMY_KILL", Neow::Bonus::THREE_ENEMY_KILL)
        .value("HUNDRED_GOLD", Neow::Bonus::HUNDRED_GOLD).value("RANDOM_COLORLESS_2", Neow::Bonus::RANDOM_COLORLESS_2)
        .value("REMOVE_TWO", Neow::Bonus::REMOVE_TWO).value("ONE_RARE_RELIC", Neow::Bonus::ONE_RARE_RELIC)
        .value("THREE_RARE_CARDS", Neow::Bonus::THREE_RARE_CARDS).value("TWO_FIFTY_GOLD", Neow::Bonus::TWO_FIFTY_GOLD)
        .value("TRANSFORM_TWO_CARDS", Neow::Bonus::TRANSFORM_TWO_CARDS).value("TWENTY_PERCENT_HP_BONUS", Neow::Bonus::TWENTY_PERCENT_HP_BONUS)
        .value("BOSS_RELIC", Neow::Bonus::BOSS_RELIC).value("INVALID", Neow::Bonus::INVALID);
    pybind11::enum_<Neow::Drawback>(m, "NeowDrawback")
        .value("INVALID", Neow::Drawback::INVALID).value("NONE", Neow::Drawback::NONE)
        .value("TEN_PERCENT_HP_LOSS", Neow::Drawback::TEN_PERCENT_HP_LOSS).value("NO_GOLD", Neow::Drawback::NO_GOLD)
        .value("CURSE", Neow::Drawback::CURSE).value("PERCENT_DAMAGE", Neow::Drawback::PERCENT_DAMAGE)
        .value("LOSE_STARTER_RELIC", Neow::Drawback::LOSE_STARTER_RELIC);
    pybind11::class_<Neow::Option>(m, "NeowOption")
        .def_readonly("r", &Neow::Option::r).def_readonly("d", &Neow::Option::d);
    pybind11::enum_<Event> eventEnum(m, "Event");
#define HEART1_EVENT(name) .value(#name, Event::name)
    eventEnum HEART1_EVENT(INVALID) HEART1_EVENT(MONSTER) HEART1_EVENT(REST) HEART1_EVENT(SHOP) HEART1_EVENT(TREASURE)
        HEART1_EVENT(NEOW) HEART1_EVENT(OMINOUS_FORGE) HEART1_EVENT(PLEADING_VAGRANT) HEART1_EVENT(ANCIENT_WRITING)
        HEART1_EVENT(OLD_BEGGAR) HEART1_EVENT(BIG_FISH) HEART1_EVENT(BONFIRE_SPIRITS) HEART1_EVENT(COLOSSEUM)
        HEART1_EVENT(CURSED_TOME) HEART1_EVENT(DEAD_ADVENTURER) HEART1_EVENT(DESIGNER_IN_SPIRE) HEART1_EVENT(AUGMENTER)
        HEART1_EVENT(DUPLICATOR) HEART1_EVENT(FACE_TRADER) HEART1_EVENT(FALLING) HEART1_EVENT(FORGOTTEN_ALTAR)
        HEART1_EVENT(THE_DIVINE_FOUNTAIN) HEART1_EVENT(GHOSTS) HEART1_EVENT(GOLDEN_IDOL) HEART1_EVENT(GOLDEN_SHRINE)
        HEART1_EVENT(WING_STATUE) HEART1_EVENT(KNOWING_SKULL) HEART1_EVENT(LAB) HEART1_EVENT(THE_SSSSSERPENT)
        HEART1_EVENT(LIVING_WALL) HEART1_EVENT(MASKED_BANDITS) HEART1_EVENT(MATCH_AND_KEEP) HEART1_EVENT(MINDBLOOM)
        HEART1_EVENT(HYPNOTIZING_COLORED_MUSHROOMS) HEART1_EVENT(MYSTERIOUS_SPHERE) HEART1_EVENT(THE_NEST) HEART1_EVENT(NLOTH)
        HEART1_EVENT(NOTE_FOR_YOURSELF) HEART1_EVENT(PURIFIER) HEART1_EVENT(SCRAP_OOZE) HEART1_EVENT(SECRET_PORTAL)
        HEART1_EVENT(SENSORY_STONE) HEART1_EVENT(SHINING_LIGHT) HEART1_EVENT(THE_CLERIC) HEART1_EVENT(THE_JOUST)
        HEART1_EVENT(THE_LIBRARY) HEART1_EVENT(THE_MAUSOLEUM) HEART1_EVENT(THE_MOAI_HEAD) HEART1_EVENT(THE_WOMAN_IN_BLUE)
        HEART1_EVENT(TOMB_OF_LORD_RED_MASK) HEART1_EVENT(TRANSMORGRIFIER) HEART1_EVENT(UPGRADE_SHRINE) HEART1_EVENT(VAMPIRES)
        HEART1_EVENT(WE_MEET_AGAIN) HEART1_EVENT(WHEEL_OF_CHANGE) HEART1_EVENT(WINDING_HALLS) HEART1_EVENT(WORLD_OF_GOOP);
#undef HEART1_EVENT

    pybind11::class_<RelicInstance> relic(m, "Relic");
    relic.def_readwrite("id", &RelicInstance::id)
        .def_readwrite("data", &RelicInstance::data);

    pybind11::class_<Map> map(m, "SpireMap");
    map.def(pybind11::init<std::uint64_t, int,int,bool>());
    map.def("get_room_type", &sts::py::getRoomType);
    map.def("has_edge", &sts::py::hasEdge);
    map.def("get_nn_rep", &sts::py::getNNMapRepresentation);
    map.def("__repr__", [](const Map &m) {
        return m.toString(true);
    });

    pybind11::class_<Card> card(m, "Card");
    card.def(pybind11::init<CardId>())
        .def("__repr__", [](const Card &c) {
            std::string s("<slaythespire.Card ");
            s += c.getName();
            if (c.isUpgraded()) {
                s += '+';
                if (c.id == sts::CardId::SEARING_BLOW) {
                    s += std::to_string(c.getUpgraded());
                }
            }
            return s += ">";
        }, "returns a string representation of a Card")
        .def("upgrade", &Card::upgrade)
        .def_readwrite("misc", &Card::misc, "value internal to the simulator used for things like ritual dagger damage");

    card.def_property_readonly("id", &Card::getId)
        .def_property_readonly("upgraded", &Card::isUpgraded)
        .def_property_readonly("upgrade_count", &Card::getUpgraded)
        .def_property_readonly("innate", &Card::isInnate)
        .def_property_readonly("transformable", &Card::canTransform)
        .def_property_readonly("upgradable", &Card::canUpgrade)
        .def_property_readonly("is_strikeCard", &Card::isStrikeCard)
        .def_property_readonly("is_starter_strike_or_defend", &Card::isStarterStrikeOrDefend)
        .def_property_readonly("rarity", &Card::getRarity)
        .def_property_readonly("type", &Card::getType);

    // --- isolated single-fight interface (RL training against just combat) ---

    pybind11::enum_<Outcome> battleOutcome(m, "BattleOutcome");
    battleOutcome.value("UNDECIDED", Outcome::UNDECIDED)
        .value("PLAYER_VICTORY", Outcome::PLAYER_VICTORY)
        .value("PLAYER_LOSS", Outcome::PLAYER_LOSS);

    // Only the two states getLegalActions() actually handles get real names;
    // everything else (potion/relic-triggered choice states -- see the
    // comment on getLegalActions in bindings-util.cpp) is unreachable from
    // this project's current scope (plain 75-card Ironclad pool, no potions/
    // relics), so it's exposed as a raw int for debugging rather than a full
    // ~25-value enum mapping that would mostly go untested.
    m.def("get_input_state_raw", [](const BattleContext &bc) { return static_cast<int>(bc.inputState); },
          "raw InputState ordinal, for debugging states getLegalActions() doesn't recognize yet");
    m.attr("INPUT_STATE_PLAYER_NORMAL") = static_cast<int>(InputState::PLAYER_NORMAL);
    m.attr("INPUT_STATE_CARD_SELECT") = static_cast<int>(InputState::CARD_SELECT);

    pybind11::enum_<search::ActionType> actionType(m, "ActionType");
    actionType.value("CARD", search::ActionType::CARD)
        .value("POTION", search::ActionType::POTION)
        .value("SINGLE_CARD_SELECT", search::ActionType::SINGLE_CARD_SELECT)
        .value("MULTI_CARD_SELECT", search::ActionType::MULTI_CARD_SELECT)
        .value("END_TURN", search::ActionType::END_TURN);

    pybind11::class_<search::Action> action(m, "Action");
    action.def(pybind11::init<search::ActionType>())
        .def(pybind11::init<search::ActionType, int>())
        .def(pybind11::init<search::ActionType, int, int>())
        .def("execute", &search::Action::execute)
        .def_property_readonly("bits", [](const search::Action &a) { return a.bits; })
        .def_property_readonly("action_type", &search::Action::getActionType)
        .def_property_readonly("source_idx", &search::Action::getSourceIdx)
        .def_property_readonly("target_idx", &search::Action::getTargetIdx)
        .def("__repr__", [](const search::Action &a) {
            std::ostringstream oss;
            oss << "<Action type=" << static_cast<int>(a.getActionType())
                << " src=" << a.getSourceIdx() << " tgt=" << a.getTargetIdx() << ">";
            return oss.str();
        });

    pybind11::class_<CardInstance> cardInstance(m, "CardInstance");
    cardInstance.def_readonly("id", &CardInstance::id)
        .def_readonly("cost", &CardInstance::cost)
        .def_readonly("cost_for_turn", &CardInstance::costForTurn)
        .def_readonly("upgraded", &CardInstance::upgraded)
        .def("__repr__", [](const CardInstance &c) {
            std::ostringstream oss;
            oss << "<CardInstance " << static_cast<int>(c.id)
                << (c.upgraded ? "+" : "") << " cost=" << (int)c.costForTurn << ">";
            return oss.str();
        });

    pybind11::class_<Monster> monster(m, "Monster");
    monster.def_readonly("cur_hp", &Monster::curHp)
        .def_readonly("max_hp", &Monster::maxHp)
        .def_readonly("block", &Monster::block)
        .def_readonly("strength", &Monster::strength)
        .def_readonly("vulnerable", &Monster::vulnerable)
        .def_readonly("weak", &Monster::weak)
        .def_readonly("half_dead", &Monster::halfDead)
        .def_property_readonly("move_history", [](const Monster &mo) {
            // Raw MMID ints (moveHistory[0]=most recent, [1]=one before that).
            // pybind11::make_tuple used explicitly (not returning a raw
            // std::pair and relying on stl.h's caster) to rule out a type-
            // caster issue as a variable when debugging why a state key
            // built from this crashed the process.
            return pybind11::make_tuple(static_cast<int>(mo.moveHistory[0]), static_cast<int>(mo.moveHistory[1]));
        })
        .def_readonly("misc_info", &Monster::miscInfo)
        .def_property_readonly("name", &Monster::getName)
        .def_property_readonly("is_targetable", &Monster::isTargetable)
        .def("__repr__", [](const Monster &mo) {
            std::ostringstream oss;
            oss << "<Monster " << mo.getName() << " hp=" << mo.curHp << "/" << mo.maxHp
                << " block=" << mo.block << ">";
            return oss.str();
        });

    pybind11::class_<sts::py::MoveCategory>(m, "MoveCategory")
        .def_readonly("self_buffs", &sts::py::MoveCategory::self_buffs)
        .def_readonly("buffs_ally", &sts::py::MoveCategory::buffs_ally)
        .def_readonly("debuffs_player", &sts::py::MoveCategory::debuffs_player);

    pybind11::class_<BattleContext> battleContext(m, "BattleContext");
    battleContext.def(pybind11::init<>())
        .def(pybind11::init<const BattleContext &>(), "copy constructor -- BattleContext already supports value semantics in C++ (default copy ctor), just wasn't exposed to Python before")
        .def_readonly("outcome", &BattleContext::outcome)
        .def_readonly("turn", &BattleContext::turn)
        .def_property_readonly("player_hp", [](const BattleContext &bc) { return bc.player.curHp; })
        .def_property_readonly("player_max_hp", [](const BattleContext &bc) { return bc.player.maxHp; })
        .def_property_readonly("player_block", [](const BattleContext &bc) { return bc.player.block; })
        .def_property_readonly("player_energy", [](const BattleContext &bc) { return bc.player.energy; })
        .def_property_readonly("player_strength", [](const BattleContext &bc) { return bc.player.strength; })
        .def_property_readonly("player_dexterity", [](const BattleContext &bc) { return bc.player.dexterity; })
        .def_property_readonly("player_cards_played_this_turn",
                                [](const BattleContext &bc) { return bc.player.cardsPlayedThisTurn; })
        .def_property_readonly("hand", [](const BattleContext &bc) {
            std::vector<CardInstance> h;
            for (int i = 0; i < bc.cards.cardsInHand; ++i) {
                h.push_back(bc.cards.hand[i]);
            }
            return h;
        })
        // draw_pile/discard_pile: previously unexposed -- the policy could
        // see its hand but had no way to know what was left in its own
        // deck (a scaling payoff still buried in the draw pile vs already
        // discarded, or the deck getting clogged with shuffled-in Voids/
        // Wounds/Dazed from a card or monster effect). Explicitly copied
        // into a std::vector (not returned directly) since CardManager's
        // drawPile/discardPile are fixed_list<CardInstance, MAX_GROUP_SIZE>
        // (sts_card_manager_use_fixed_list -- see sts_common.h), which
        // pybind11's stl.h has no caster for; this construction works
        // identically regardless of which container CardManager uses.
        .def_property_readonly("draw_pile", [](const BattleContext &bc) {
            return std::vector<CardInstance>(bc.cards.drawPile.begin(), bc.cards.drawPile.end());
        })
        .def("set_draw_pile_order", [](BattleContext &bc, const std::vector<int> &order) {
            // Permute the draw pile WITHOUT changing its contents. Exists to
            // measure what the search is worth when it cannot see the true draw
            // order: `run_mcts_search` roots its tree in a full copy of this
            // context, so today every simulation draws exactly what reality will
            // draw. Live play cannot do that -- CommunicationMod reports the
            // draw pile's contents, not its shuffle order -- so the honest cost
            // of that advantage needs a way to scramble order in place.
            // `draw_pile` hands back a COPY, so shuffling it Python-side is a
            // silent no-op.
            const auto size = static_cast<int>(bc.cards.drawPile.size());
            if (static_cast<int>(order.size()) != size) {
                throw std::invalid_argument(
                    "set_draw_pile_order needs exactly one index per draw-pile card");
            }
            std::vector<CardInstance> shuffled;
            shuffled.reserve(size);
            std::vector<bool> used(size, false);
            for (int index : order) {
                if (index < 0 || index >= size || used[index]) {
                    throw std::invalid_argument(
                        "set_draw_pile_order needs a permutation of 0..n-1");
                }
                used[index] = true;
                shuffled.push_back(bc.cards.drawPile[index]);
            }
            for (int i = 0; i < size; ++i) {
                bc.cards.drawPile[i] = shuffled[i];
            }
        }, pybind11::arg("order"),
           "Reorder the draw pile in place from a permutation of its indices; "
           "contents are unchanged. See _clairvoyance_cost.py.")
        .def_property_readonly("discard_pile", [](const BattleContext &bc) {
            return std::vector<CardInstance>(bc.cards.discardPile.begin(), bc.cards.discardPile.end());
        })
        .def_property_readonly("exhaust_pile", [](const BattleContext &bc) {
            return std::vector<CardInstance>(bc.cards.exhaustPile.begin(), bc.cards.exhaustPile.end());
        })
        .def_property_readonly("monsters", [](const BattleContext &bc) {
            std::vector<Monster> ms;
            for (int i = 0; i < bc.monsters.monsterCount; ++i) {
                ms.push_back(bc.monsters.arr[i]);
            }
            return ms;
        })
        .def("get_legal_actions", &sts::py::getLegalActions)
        .def("get_monster_move_damage", &sts::py::getMonsterMoveDamage)
        .def_static("reuse_diag", []() {
            return pybind11::make_tuple(g_reuseHits, g_reuseMisses);
        })
        .def_static("merge_diag", []() {
            return pybind11::make_tuple(g_chanceMergeSamples, g_chanceMergeHits);
        })
        .def_static("last_search_diag", []() {
            return pybind11::make_tuple(g_lastRootValueGap,
                                        g_lastSearchDangerous,
                                        g_lastSearchEscalated);
        })
        .def("get_player_status_value", &sts::py::getPlayerStatusValue)
        .def("get_monster_status_value", &sts::py::getMonsterStatusValue)
        .def("get_monster_misc_info", &sts::py::getMonsterMiscInfo)
        .def("classify_monster_move", &sts::py::classifyMonsterMove)
        .def("has_relic",
             [](const BattleContext &bc, RelicId r) { return bc.player.hasRelicRuntime(r); },
             "query whether the player currently has a given relic during combat "
             "(Player::hasRelicRuntime already exists, mirrors get_player_status_value)"
        )
        .def_property_readonly("potions", [](const BattleContext &bc) {
            // Raw slots, EMPTY_POTION_SLOT entries included (unlike hand/
            // draw_pile/discard_pile above, which have no concept of an
            // "empty slot") -- deliberately, so index i here always matches
            // slot i, which getLegalActions' POTION actions are source-
            // indexed against. Filtering empties out would desync that
            // correspondence the moment a middle slot (not just the last)
            // is empty. potionCapacity, not the fixed array size 5, is the
            // real slot count for this player.
            return std::vector<Potion>(bc.potions.begin(), bc.potions.begin() + bc.potionCapacity);
        })
        .def("decorrelate_rng", [](BattleContext &bc) {
            // The copy constructor above is a bit-for-bit struct copy,
            // RNG state included -- two clones of the same BattleContext
            // replay the *identical* deterministic sequence of enemy moves
            // and card draws if you execute the same action on both (verified
            // directly: three END_TURN executions on three fresh clones came
            // back with identical resulting HP and hand every time). A
            // search that clones a state to sample multiple hypothetical
            // futures needs those futures to actually diverge -- this
            // reseeds each gameplay-relevant RNG stream from one draw of
            // itself (mirrors the equivalent fix already made to the
            // Python-side engine's CombatState.clone() this session, same
            // reasoning: the exact draw sequence isn't part of any
            // correctness-relevant state, only decorrelation matters).
            // monsterHpRng/potionRng are deliberately left alone -- they're
            // only consumed once at battle start, not mid-combat.
            bc.aiRng = sts::Random(bc.aiRng.nextLong());
            bc.cardRandomRng = sts::Random(bc.cardRandomRng.nextLong());
            bc.miscRng = sts::Random(bc.miscRng.nextLong());
            bc.shuffleRng = sts::Random(bc.shuffleRng.nextLong());
        }, "reseed the gameplay-relevant RNG streams (aiRng/cardRandomRng/miscRng/shuffleRng) from draws of themselves, so a freshly-cloned BattleContext produces independent outcomes instead of exactly replaying its parent's future")
        .def("seed_rng", [](BattleContext &bc, std::uint64_t base) {
            // Common Random Numbers: unlike decorrelate_rng's self-referential
            // reseed (irreproducible -- depends on whatever the stream had
            // already consumed), this reseeds from an EXTERNALLY supplied
            // base, so calling this with the same base always produces the
            // same 4 streams. sts::Random's own constructor already runs the
            // seed through murmur3 twice (see Random.h), the same
            // decorrelation guarantee decorrelate_rng relies on -- base,
            // base+1, base+2, base+3 for the 4 streams are already
            // well-separated by that mixing, no extra salting needed.
            // Lets a caller compare two search runs (two policies, two
            // configs, before/after a change) under matched pseudo-luck
            // instead of independently-decorrelated noise -- mirrors Silver
            // Automaton's randomnessBase technique (see its README's
            // "Common random numbers" section), adapted to comparing
            // different search CONFIGURATIONS rather than sibling stochastic
            // actions within one search (this engine's deterministic/
            // stochastic split means only END_TURN is ever a chance node, so
            // there's no literal sibling-stochastic-action case to apply it
            // to the way Silver Automaton's does).
            bc.aiRng = sts::Random(base);
            bc.cardRandomRng = sts::Random(base + 1);
            bc.miscRng = sts::Random(base + 2);
            bc.shuffleRng = sts::Random(base + 3);
        }, "reseed the gameplay-relevant RNG streams from an explicit base value, for reproducible "
           "Common-Random-Numbers-style paired comparisons across search runs")
        .def("rng_counter_sum", [](const BattleContext &bc) {
            // Sum of the same 4 gameplay-relevant streams' consumption
            // counters (Random::counter, incremented on every draw --
            // see Random.h). Comparing this before/after executing ANY
            // action is how a caller detects whether that specific
            // execution actually consumed randomness, rather than
            // assuming it from the action's type -- confirmed necessary
            // the hard way: some CARD actions (shuffle-into-draw-pile
            // effects like Wild Strike/Reckless Charge) consume RNG even
            // though they aren't END_TURN, so a search that only ever
            // treated END_TURN as a chance node could execute one of
            // these "deterministic" actions on two RNG-divergent copies
            // of what's otherwise the identical state and get two
            // genuinely different results (different draw-pile order),
            // which is exactly what caused two real crashes (an
            // assertion, then a segfault) in earlier search-improvement
            // attempts that assumed action_type alone was enough.
            return bc.aiRng.counter + bc.cardRandomRng.counter + bc.miscRng.counter + bc.shuffleRng.counter;
        }, "sum of the gameplay-relevant RNG streams' consumption counters, for detecting whether a "
           "specific action execution actually consumed randomness")
        // Profiling hooks for the search's copy cost: BattleContext is copied ~2-3x per
        // simulation (tree edge expansion, chance-node sampling, and the rollout's own working
        // copy), so its copy cost is a direct multiplier on total search time.
        .def("copy_self", [](const BattleContext &bc) { return BattleContext(bc); },
             "returns a copy of this BattleContext -- exposed to measure copy cost")
        .def_static("sizeof_bytes", []() { return sizeof(BattleContext); },
             "sizeof(BattleContext) in bytes")
        .def_static("sizeof_action_queue_bytes", []() { return sizeof(ActionQueue<50>); },
             "sizeof(ActionQueue<50>) -- the std::array<std::function,50> action queue")
        .def("action_queue_size", [](const BattleContext &bc) { return bc.actionQueue.size; },
             "live entries in the action queue (vs its 50 capacity, all of which get copied)")
        .def("state_key_bundle", [](const BattleContext &bc) {
            // Batched, single-call replacement for az_search.py's own
            // _state_key(), which built the identical tuple via ~19
            // separate get_player_status_value calls (each a Python->C++
            // crossing doing an internal linear string-comparison dispatch)
            // plus 6 more such calls PER MONSTER, plus per-card attribute
            // reads for hand/draw_pile/discard_pile -- confirmed by
            // profiling to be the dominant remaining cost of the search's
            // rollouts once get_card_type caching (see _cached_card_type's
            // own comment) was already fixed. Same exact field set, same
            // order, same semantics (hand/discard sorted -- order-
            // independent; draw pile left in order -- determines future
            // draws) as the Python version it replaces, just computed with
            // direct ordinal/templated status dispatch (no string compare)
            // and packed into ONE returned tuple instead of dozens of
            // separate calls. _state_key has caused two real crashes this
            // session from incomplete-key bugs, so exactness here isn't
            // optional -- validated field-for-field against the old pure-
            // Python implementation across many random states before this
            // replaced it (see az_search.py's _state_key docstring).
            static const PlayerStatus PLAYER_STATUS_IDS[] = {
                PlayerStatus::ARTIFACT, PlayerStatus::BARRICADE, PlayerStatus::METALLICIZE,
                PlayerStatus::RITUAL, PlayerStatus::RAGE, PlayerStatus::RUPTURE,
                PlayerStatus::COMBUST, PlayerStatus::DEMON_FORM, PlayerStatus::DARK_EMBRACE,
                PlayerStatus::EVOLVE, PlayerStatus::FEEL_NO_PAIN, PlayerStatus::FIRE_BREATHING,
                PlayerStatus::JUGGERNAUT, PlayerStatus::PANACHE, PlayerStatus::ENVENOM,
                PlayerStatus::FLAME_BARRIER, PlayerStatus::BRUTALITY, PlayerStatus::REGEN,
                PlayerStatus::CORRUPTION,
            };
            constexpr int nPlayerStatuses = sizeof(PLAYER_STATUS_IDS) / sizeof(PlayerStatus);
            pybind11::tuple pStatuses(nPlayerStatuses);
            for (int i = 0; i < nPlayerStatuses; ++i) {
                pStatuses[i] = bc.player.getStatusRuntime(PLAYER_STATUS_IDS[i]);
            }

            pybind11::tuple monsters(bc.monsters.monsterCount);
            for (int i = 0; i < bc.monsters.monsterCount; ++i) {
                const Monster &mo = bc.monsters.arr[i];
                pybind11::tuple mStatuses(6);
                mStatuses[0] = mo.getStatus<MS::POISON>();
                mStatuses[1] = mo.getStatus<MS::PLATED_ARMOR>();
                mStatuses[2] = mo.getStatus<MS::ARTIFACT>();
                mStatuses[3] = mo.getStatus<MS::METALLICIZE>();
                mStatuses[4] = mo.getStatus<MS::MODE_SHIFT>();
                mStatuses[5] = mo.getStatus<MS::TIME_WARP>();
                monsters[i] = pybind11::make_tuple(
                    mo.curHp, mo.block, mo.strength, mo.vulnerable, mo.weak, mo.halfDead,
                    pybind11::make_tuple(static_cast<int>(mo.moveHistory[0]), static_cast<int>(mo.moveHistory[1])),
                    mo.miscInfo, mStatuses
                );
            }

            std::vector<int> handIds;
            handIds.reserve(bc.cards.cardsInHand);
            for (int i = 0; i < bc.cards.cardsInHand; ++i) {
                handIds.push_back(static_cast<int>(bc.cards.hand[i].id));
            }
            std::sort(handIds.begin(), handIds.end());
            pybind11::tuple hand(handIds.size());
            for (size_t i = 0; i < handIds.size(); ++i) hand[i] = handIds[i];

            pybind11::tuple draw(bc.cards.drawPile.size());
            for (size_t i = 0; i < bc.cards.drawPile.size(); ++i) {
                draw[i] = static_cast<int>(bc.cards.drawPile[i].id);
            }

            std::vector<int> discardIds;
            discardIds.reserve(bc.cards.discardPile.size());
            for (const auto &c : bc.cards.discardPile) {
                discardIds.push_back(static_cast<int>(c.id));
            }
            std::sort(discardIds.begin(), discardIds.end());
            pybind11::tuple discard(discardIds.size());
            for (size_t i = 0; i < discardIds.size(); ++i) discard[i] = discardIds[i];

            return pybind11::make_tuple(
                bc.player.curHp, bc.player.block, bc.player.energy, bc.turn,
                pStatuses, monsters, hand, draw, discard
            );
        }, "batched, single-call replacement for building az_search.py's _state_key tuple -- "
           "see that function's own docstring for the exact field list this mirrors")
        // Explicit lambda rather than binding &nativeHeuristicPlayout directly: that function
        // gained a raveTrace out-parameter, and pybind11 does not inherit C++ default arguments,
        // so exposing it raw silently turned the trace into a REQUIRED second argument and broke
        // every existing one- and two-arg caller. RAVE tracing is an internal search concern with
        // no reason to be part of this diagnostic binding, so it is simply never traced here.
        .def("heuristic_playout",
             [](const BattleContext &bc, int maxTurn) { return nativeHeuristicPlayout(bc, maxTurn, nullptr); },
             pybind11::arg("max_turn") = std::numeric_limits<int>::max(),
             "single-call, all-native replacement for az_search.py's _heuristic_playout -- "
             "plays a COPY of this BattleContext to completion using a C++ port of "
             "_heuristic_pick's scoring, returning the same W_SHAPE-scaled terminal/potential "
             "reward. See the anonymous-namespace block above this class for the full "
             "rationale and the DRIFT WARNING on keeping this in sync with the Python original")
        .def("leaf_features", [](const BattleContext &bc) {
            const auto f = nativeLeafFeatures(bc);
            return std::vector<double>(f.begin(), f.end());
        }, "The exact leaf value-function feature vector (nativeLeafFeatures) for this state: "
           "[player_hp, block, energy, strength, dexterity, metallicize, sum_monster_hp, "
           "sum_incoming_dmg, alive_count, turn]. Same features the native linear/value-net "
           "leaf estimators use -- exposed so a Python training set can be built with features "
           "that provably match inference-time (one C++ source of truth).")
        .def("action_features", [](const BattleContext &bc, const search::Action &a) {
            const HeuristicContext ctx = nativeComputeHeuristicContext(bc);
            const auto f = nativeActionFeatures(bc, a, ctx);
            return std::vector<double>(f.begin(), f.end());
        }, pybind11::arg("action"),
           "The exact per-action feature vector (nativeActionFeatures) for one legal action in "
           "this state: [is_attack, is_skill, is_power, is_other, target_hp_missing_fraction, "
           "target_block_fraction_capped, is_aoe_into_multiple_monsters, card_pick_rate_weight]. "
           "Concatenate with leaf_features() (state first, then this) to build the exact input "
           "the learned rollout-scoring net (load_policy_net) sees at inference.")
        .def("__repr__", [](const BattleContext &bc) {
            std::ostringstream oss;
            oss << bc;
            return oss.str();
        });

    m.def("new_battle", &sts::py::newBattle,
          "construct a BattleContext for an isolated fight against a specific MonsterEncounter");

    pybind11::class_<NativeMonsterSpec>(m, "NativeMonsterSpec",
        "One monster's explicit state for build_battle_context -- see that function's docstring.")
        .def(pybind11::init<>())
        .def_readwrite("monster_id_name", &NativeMonsterSpec::monsterIdName)
        .def_readwrite("cur_hp", &NativeMonsterSpec::curHp)
        .def_readwrite("max_hp", &NativeMonsterSpec::maxHp)
        .def_readwrite("block", &NativeMonsterSpec::block)
        .def_readwrite("half_dead", &NativeMonsterSpec::halfDead)
        .def_readwrite("statuses", &NativeMonsterSpec::statuses)
        .def_readwrite("move_name", &NativeMonsterSpec::moveName);

    m.def("build_battle_context", &nativeBuildBattleContext,
          pybind11::arg("player_hp"), pybind11::arg("player_max_hp"), pybind11::arg("player_block"),
          pybind11::arg("player_energy"), pybind11::arg("player_statuses"), pybind11::arg("monsters"),
          pybind11::arg("hand_cards"), pybind11::arg("draw_pile_cards"), pybind11::arg("discard_pile_cards"),
          pybind11::arg("exhaust_pile_cards"), pybind11::arg("potion_slots"), pybind11::arg("relics"),
          pybind11::arg("turn"), pybind11::arg("ascension"), pybind11::arg("rng_seed"),
          "Construct a BattleContext from EXACT, externally-reported state -- for bridging a live "
          "Slay the Spire game (via CommunicationMod/spirecomm) into this engine's own search, not "
          "for training/simulation use (which already goes through new_battle's randomized setup). "
          "player_statuses: list of (canonical PlayerStatus enum-name, amount) tuples, e.g. "
          "[('VULNERABLE', 2), ('STRENGTH', 3)]. monsters: list of NativeMonsterSpec. "
          "*_cards: list of (card_string_id, upgrade_count) tuples -- card_string_id matches the "
          "real game's own card ids exactly (e.g. 'Strike_R', 'Bash'), NOT this engine's display "
          "names; upgrade_count > 0 means upgraded (only Searing Blow uses the exact count). "
          "potion_slots: list of Potion enum values, ONE PER SLOT including empty ones "
          "(Potion.EMPTY_POTION_SLOT) -- Potion is already a bound enum, so translate spirecomm's "
          "potion_id strings to it Python-side before calling. "
          "relics: list of RelicId enum values (also already a bound enum, same translate-Python-"
          "side convention as potions). NOTE: relics with ordinal>=128 are silently skipped -- see "
          "setPlayerRelicByEnum's own comment for the pre-existing Player::relicBits0/relicBits1 "
          "capacity limit this runs into (includes VAJRA). Only relic PRESENCE is modeled, not any "
          "relic-specific accumulated counter/charge state. "
          "Unrecognized status/move names are silently skipped (logged Python-side); unrecognized "
          "monster id or card_string_id names raise, since those indicate a real bug in the caller's "
          "mapping tables rather than an expected, bounded gap. rng_seed is a synthetic value for "
          "OUR search's own future exploration only -- the real game's RNG state isn't observable "
          "through spirecomm, so this cannot and does not attempt to replay it.");

    m.def("run_mcts_search", [](const BattleContext &bc, int nSimulations, pybind11::object crnBase,
                                 pybind11::object searchSeed) {
        const bool useCrn = !crnBase.is_none();
        const std::uint64_t crnBaseValue = useCrn ? crnBase.cast<std::uint64_t>() : 0;
        const bool useSearchSeed = !searchSeed.is_none();
        const std::uint64_t searchSeedValue = useSearchSeed ? searchSeed.cast<std::uint64_t>() : 0;
        std::pair<search::Action, std::vector<std::int64_t>> result;
        {
            // GIL released ONLY around the actual search -- everything that
            // touches a Python object (crnBase above, the return tuple
            // below) stays outside this scope. Safe because nativeSimulate
            // and everything it calls touch no global mutable state (the
            // arena/transposition table are locals of this one call; the
            // only "global" data read, cardTypes, is `static constexpr` --
            // a compile-time constant, not a runtime-mutable global) --
            // confirmed by inspection before adding this, specifically to
            // let multiple Python THREADS run independent searches on the
            // same starting bc truly concurrently (root parallelization:
            // several independent trees combined by summing their root
            // visit counts, rather than one bigger sequential tree). Without
            // this, the GIL would serialize concurrent calls into this
            // function, making threads pointless for this workload.
            pybind11::gil_scoped_release release;
            result = nativeRunMctsSearch(bc, nSimulations, useCrn, crnBaseValue, useSearchSeed, searchSeedValue);
        }
        return pybind11::make_tuple(result.first, pybind11::cast(result.second));
    }, pybind11::arg("bc"), pybind11::arg("n_simulations") = 200,
       pybind11::arg("crn_base") = pybind11::none(), pybind11::arg("search_seed") = pybind11::none(),
       "All-native replacement for expectimax_search.py's choose_action -- runs the ENTIRE "
       "expectimax MCTS loop (UCB1 selection, DPW chance-node widening, the RNG-consumption-"
       "probe action classification, transposition sharing, heuristic-rollout leaf evaluation) "
       "in C++, one call per DECISION instead of one call per SIMULATION STEP. Reads "
       "g_params (see set_search_params) for its tunable weights/constants -- NEVER call "
       "set_search_params while a run_mcts_search/root-parallel-search call using the OLD "
       "params is still in flight (e.g. from another thread): g_params is process-global "
       "mutable state read concurrently by every thread of a root-parallel search, so "
       "changing it mid-flight is a real data race. Safe pattern: set params, run an "
       "evaluation to COMPLETION (sequentially, or via root_parallel_search's own internal "
       "threads for ONE fixed setting), THEN set new params -- never interleave. Across "
       "genuinely different parameter settings evaluated concurrently, use separate "
       "PROCESSES (each gets its own independent copy of g_params), not threads within one "
       "process -- see lightspeed/tune_search_cma.py, which does exactly this. Returns "
       "(action, visit_counts) matching choose_action's own return shape (visit_counts as a "
       "plain list here, not a numpy array -- wrap with np.array() if that matters to a "
       "caller). See the anonymous-namespace block above BattleContext for the full "
       "implementation and its DRIFT WARNING / non-bit-identical-with-Python caveats. Releases "
       "the GIL during the search itself, so multiple Python threads can call this "
       "concurrently for true root parallelization (see root_parallel_search in "
       "expectimax_search.py). search_seed is optional: omit it for normal stochastic search, "
       "or pass a uint64 to make the search's own sampling reproducible for paired evaluation.");

    m.def("native_playout_battle", [](BattleContext &bc, int nSimulations,
                                      pybind11::object searchSeed) {
        const bool seeded = !searchSeed.is_none();
        const std::uint64_t seed = seeded ? searchSeed.cast<std::uint64_t>() : 0;
        pybind11::gil_scoped_release release;
        nativePlayoutBattle(bc, nSimulations, seeded, seed);
    }, pybind11::arg("bc"), pybind11::arg("n_simulations") = 200,
       pybind11::arg("search_seed") = pybind11::none(),
       "Plays bc to completion using run_mcts_search's own tuned native MCTS, one decision after "
       "another, entirely in C++ -- mutates bc in place, same call shape as Agent.playout_battle. "
       "Exists so a whole-fight wall-clock speed comparison against another engine's own native "
       "playout_battle isn't skewed by Python round-trip overhead once per decision the way a "
       "Python-side run_mcts_search-in-a-loop would be. Same g_params/threading rules as "
       "run_mcts_search.");

    m.def("native_playout_current_battle", [](GameContext &gc, int nSimulations,
                                              pybind11::object searchSeed) {
        BattleContext bc;
        bc.init(gc);
        const bool seeded = !searchSeed.is_none();
        const std::uint64_t seed = seeded ? searchSeed.cast<std::uint64_t>() : 0;
        pybind11::gil_scoped_release release;
        nativePlayoutBattle(bc, nSimulations, seeded, seed);
        bc.exitBattle(gc);
    }, pybind11::arg("gc"), pybind11::arg("n_simulations") = 200,
       pybind11::arg("search_seed") = pybind11::none(),
       "Play gc's current battle with native MCTS and synchronize the result back into gc.");

    m.def("native_playout_current_battle_result", [](GameContext &gc, int nSimulations,
                                                     pybind11::object searchSeed) {
        BattleContext bc;
        bc.init(gc);
        const bool seeded = !searchSeed.is_none();
        const std::uint64_t seed = seeded ? searchSeed.cast<std::uint64_t>() : 0;
        NativePlayoutStats searchStats;
        {
            pybind11::gil_scoped_release release;
            searchStats = nativePlayoutBattle(bc, nSimulations, seeded, seed);
        }

        int monsterHp = 0;
        int monsterMaxHp = 0;
        for (int i = 0; i < bc.monsters.monsterCount; ++i) {
            const Monster &monster = bc.monsters.arr[i];
            // A half-dead Awakened One/Darkling is pending revival and should
            // count as a full remaining threat, not as zero progress.
            monsterHp += monster.halfDead
                    ? monster.maxHp : std::max(0, static_cast<int>(monster.curHp));
            monsterMaxHp += monster.maxHp;
        }
        pybind11::dict result;
        result["outcome"] = static_cast<int>(bc.outcome);
        result["player_hp"] = bc.player.curHp;
        result["player_max_hp"] = bc.player.maxHp;
        result["monster_hp"] = monsterHp;
        result["monster_max_hp"] = monsterMaxHp;
        result["monster_hp_fraction"] = monsterMaxHp > 0
                ? static_cast<double>(monsterHp) / monsterMaxHp : 0.0;
        result["turn"] = bc.turn;
        result["encounter"] = static_cast<int>(bc.encounter);
        result["is_boss"] = isBossEncounter(bc.encounter);
        result["search_decisions"] = searchStats.decisions;
        result["searched_decisions"] = searchStats.searchedDecisions;
        result["stall_fallback_decisions"] = searchStats.stallFallbackDecisions;
        result["stall_progress_override_decisions"] = searchStats.stallProgressOverrideDecisions;
        result["soft_tempo_override_decisions"] = searchStats.softTempoOverrideDecisions;
        result["stall_recovery_search_decisions"] = searchStats.stallRecoverySearchDecisions;
        result["max_consecutive_stall_fallbacks"] = searchStats.maxConsecutiveStallFallbacks;
        result["first_stall_turn"] = searchStats.firstStallTurn;
        result["last_stall_turn"] = searchStats.lastStallTurn;
        result["first_stall_player_hp"] = searchStats.firstStallPlayerHp;
        result["first_stall_monster_hp"] = searchStats.firstStallMonsterHp;
        result["first_tempo_override_turn"] = searchStats.firstTempoOverrideTurn;
        result["first_tempo_override_player_hp"] = searchStats.firstTempoOverridePlayerHp;
        result["first_tempo_override_monster_hp"] = searchStats.firstTempoOverrideMonsterHp;
        result["forced_decisions"] = (
            searchStats.decisions - searchStats.searchedDecisions
            - searchStats.stallFallbackDecisions);
        result["search_simulations_total"] = searchStats.simulations;
        result["simulations_per_decision"] = nSimulations;
        result["sequential_halving"] = g_useSeqHalving;
        result["deterministic_search"] = seeded;
        result["turn_limit_reached"] = bc.turn > 500;
        bc.exitBattle(gc);
        return result;
    }, pybind11::arg("gc"), pybind11::arg("n_simulations") = 200,
       pybind11::arg("search_seed") = pybind11::none(),
       "Play gc's current battle with native MCTS, return terminal battle details, "
       "and synchronize the result back into gc. The result preserves monster-health "
       "progress that GameContext normally discards after a loss.");

    m.def("get_search_params", []() {
        pybind11::dict d;
        d["c_ucb"] = g_params.cUcb;
        d["c_ucb_chance"] = g_params.cUcbChance;
        d["wc_chance"] = g_params.wcChance;
        d["wa_chance"] = g_params.waChance;
        d["loss_progress_credit_weight"] = g_params.lossProgressCreditWeight;
        d["win_hp_weight"] = g_params.winHpWeight;
        d["early_act_easy_pool_hp_safety_weight"] = g_params.earlyActEasyPoolHpSafetyWeight;
        d["potion_score_weight"] = g_params.potionScoreWeight;
        d["rave_bias"] = g_params.raveBias;
        d["rollout_temperature"] = g_params.rolloutTemperature;
        d["energy_waste_weight"] = g_params.energyWasteWeight;
        d["block_weight"] = g_params.blockWeight;
        d["enemy_block_weight"] = g_params.enemyBlockWeight;
        d["vulnerable_apply_bonus"] = g_params.vulnerableApplyBonus;
        d["weak_apply_bonus"] = g_params.weakApplyBonus;
        d["power_per_turn_value_weight"] = g_params.powerPerTurnValueWeight;
        d["power_immediate_value_weight"] = g_params.powerImmediateValueWeight;
        d["win_hp_fraction_weight"] = g_params.winHpFractionWeight;
        d["win_bonus_weight"] = g_params.winBonusWeight;
        d["boss_heal_credit_weight"] = g_params.bossHealCreditWeight;
        d["power_horizon_weight"] = g_params.powerHorizonWeight;
        d["power_horizon_hp"] = g_params.powerHorizonHp;
        d["boss_power_multiplier"] = g_params.bossPowerMultiplier;
        d["win_turn_penalty_weight"] = g_params.winTurnPenaltyWeight;
        d["alive_monster_penalty_weight"] = g_params.aliveMonsterPenaltyWeight;
        d["brewing_threat_estimate"] = g_params.brewingThreatEstimate;
        d["attack_base"] = g_params.attackBase;
        d["attack_finish_off_scale"] = g_params.attackFinishOffScale;
        d["attack_block_penalty_scale"] = g_params.attackBlockPenaltyScale;
        d["aoe_bonus"] = g_params.aoeBonus;
        d["skill_base"] = g_params.skillBase;
        d["skill_danger_scale"] = g_params.skillDangerScale;
        d["skill_haste_penalty"] = g_params.skillHastePenalty;
        d["attack_damage_score_weight"] = g_params.attackDamageScoreWeight;
        d["direct_block_score_weight"] = g_params.directBlockScoreWeight;
        d["self_damage_score_penalty"] = g_params.selfDamageScorePenalty;
        d["silent_poison_apply_bonus"] = g_params.silentPoisonApplyBonus;
        d["rollout_potion_base"] = g_params.rolloutPotionBase;
        d["rollout_non_card_base"] = g_params.rolloutNonCardBase;
        d["rollout_potion_danger_scale"] = g_params.rolloutPotionDangerScale;
        d["rollout_potion_finish_off_scale"] = g_params.rolloutPotionFinishOffScale;
        d["rollout_potion_discard_penalty"] = g_params.rolloutPotionDiscardPenalty;
        d["mast_weight"] = g_params.mastWeight;
        d["mast_min_visits"] = g_params.mastMinVisits;
        d["seq_halving_candidates"] = g_params.seqHalvingCandidates;
        d["backup_max_weight"] = g_params.backupMaxWeight;
        d["honest_draw_order"] = g_params.honestDrawOrder;
        d["search_max_turns"] = g_params.searchMaxTurns;
        d["power_score"] = g_params.powerScore;
        d["end_turn_time_warp_risk_score"] = g_params.endTurnTimeWarpRiskScore;
        d["skill_haste_danger_threshold"] = g_params.skillHasteDangerThreshold;
        d["per_card_weight_scale"] = g_params.perCardWeightScale;
        d["card_play_prior_weight"] = g_params.cardPlayPriorWeight;
        d["boss_card_play_prior_weight"] = g_params.bossCardPlayPriorWeight;
        d["paired_determinization"] = g_params.pairedDeterminization;
        d["merge_duplicate_actions"] = g_params.mergeDuplicateActions;
        d["escalation_sims"] = g_params.escalationSims;
        d["escalation_qgap"] = g_params.escalationQgap;
        d["escalation_danger_frac"] = g_params.escalationDangerFrac;
        d["merge_chance_outcomes"] = g_params.mergeChanceOutcomes;
        d["intangible_attack_penalty"] = g_params.intangibleAttackPenalty;
        d["artifact_aware_debuffs"] = g_params.artifactAwareDebuffs;
        d["tree_reuse"] = g_params.treeReuse;
        d["draw_first_bonus"] = g_params.drawFirstBonus;
        d["burst_debuff_timing_weight"] = g_params.burstDebuffTimingWeight;
        d["max_hp_gain_weight"] = g_params.maxHpGainWeight;
        d["gold_delta_weight"] = g_params.goldDeltaWeight;
        d["parasite_penalty_weight"] = g_params.parasitePenaltyWeight;
        d["survival_mode_threshold"] = g_params.survivalModeThreshold;
        d["survival_mode_attack_scale"] = g_params.survivalModeAttackScale;
        d["honest_wc_chance"] = g_params.honestWcChance;
        d["honest_wa_chance"] = g_params.honestWaChance;
        d["c_puct"] = g_params.cPuct;
        d["puct_temperature"] = g_params.puctTemperature;
        d["policy_net_weight"] = g_params.policyNetWeight;
        d["block_sufficiency_margin"] = g_params.blockSufficiencyMargin;
        d["defensive_card_suppression_penalty"] = g_params.defensiveCardSuppressionPenalty;
        d["vf_hp"] = g_params.vfHp;
        d["vf_monster_hp"] = g_params.vfMonsterHp;
        d["vf_incoming"] = g_params.vfIncoming;
        d["vf_block"] = g_params.vfBlock;
        d["vf_energy"] = g_params.vfEnergy;
        d["vf_strength"] = g_params.vfStrength;
        d["vf_dexterity"] = g_params.vfDexterity;
        d["vf_alive"] = g_params.vfAlive;
        d["vf_turn"] = g_params.vfTurn;
        d["vf_metallicize"] = g_params.vfMetallicize;
        return d;
    }, "Current values of every runtime-tunable search parameter (TunableParams/g_params, "
       "see its own comment above BattleContext's binding), as a dict keyed by snake_case "
       "name -- the exact keys set_search_params accepts. Defaults match this session's own "
       "hand-tuned/chosen values.");

    m.def("reset_search_tree", &nativeResetSearchTree,
          "Discard any battle-long reused search tree (tree_reuse). Call at "
          "battle boundaries; a state-key miss also clears it automatically.");

    m.def("reset_search_config", []() {
        g_params = TunableParams{};
        g_earlyActCardBias.fill(0.0);
        g_hasEarlyActCardBias = false;
        nativeMastReset();
        g_useRave = false;
        g_useSeqHalving = false;
        g_useStateMerging = false;
        g_leafEvalMode = LeafEvalMode::ROLLOUT;
        g_truncatedRolloutSteps = 3;
    }, "Restore all native search parameters and selector flags to compiled defaults. "
       "Use before applying a saved configuration so omitted keys cannot inherit stale "
       "process-global experiment settings. Never call while a search is in flight.");

    m.def("set_search_params", [](pybind11::dict d) {
        // NOT thread-safe against a concurrent in-flight search using the OLD values -- see
        // run_mcts_search's own docstring for the required calling pattern (sequential, or
        // separate processes across different settings, never threads).
        auto setIf = [&](const char *key, double &field) {
            if (d.contains(key)) {
                field = d[key].cast<double>();
            }
        };
        setIf("c_ucb", g_params.cUcb);
        setIf("c_ucb_chance", g_params.cUcbChance);
        setIf("wc_chance", g_params.wcChance);
        setIf("wa_chance", g_params.waChance);
        setIf("loss_progress_credit_weight", g_params.lossProgressCreditWeight);
        setIf("win_hp_weight", g_params.winHpWeight);
        setIf("early_act_easy_pool_hp_safety_weight", g_params.earlyActEasyPoolHpSafetyWeight);
        setIf("potion_score_weight", g_params.potionScoreWeight);
        setIf("rave_bias", g_params.raveBias);
        setIf("rollout_temperature", g_params.rolloutTemperature);
        setIf("energy_waste_weight", g_params.energyWasteWeight);
        setIf("block_weight", g_params.blockWeight);
        setIf("enemy_block_weight", g_params.enemyBlockWeight);
        setIf("vulnerable_apply_bonus", g_params.vulnerableApplyBonus);
        setIf("weak_apply_bonus", g_params.weakApplyBonus);
        setIf("power_per_turn_value_weight", g_params.powerPerTurnValueWeight);
        setIf("power_immediate_value_weight", g_params.powerImmediateValueWeight);
        setIf("win_hp_fraction_weight", g_params.winHpFractionWeight);
        setIf("win_bonus_weight", g_params.winBonusWeight);
        setIf("boss_heal_credit_weight", g_params.bossHealCreditWeight);
        setIf("power_horizon_weight", g_params.powerHorizonWeight);
        setIf("power_horizon_hp", g_params.powerHorizonHp);
        setIf("boss_power_multiplier", g_params.bossPowerMultiplier);
        setIf("win_turn_penalty_weight", g_params.winTurnPenaltyWeight);
        setIf("alive_monster_penalty_weight", g_params.aliveMonsterPenaltyWeight);
        setIf("brewing_threat_estimate", g_params.brewingThreatEstimate);
        setIf("attack_base", g_params.attackBase);
        setIf("attack_finish_off_scale", g_params.attackFinishOffScale);
        setIf("attack_block_penalty_scale", g_params.attackBlockPenaltyScale);
        setIf("aoe_bonus", g_params.aoeBonus);
        setIf("skill_base", g_params.skillBase);
        setIf("skill_danger_scale", g_params.skillDangerScale);
        setIf("skill_haste_penalty", g_params.skillHastePenalty);
        setIf("attack_damage_score_weight", g_params.attackDamageScoreWeight);
        setIf("direct_block_score_weight", g_params.directBlockScoreWeight);
        setIf("self_damage_score_penalty", g_params.selfDamageScorePenalty);
        setIf("silent_poison_apply_bonus", g_params.silentPoisonApplyBonus);
        setIf("rollout_potion_base", g_params.rolloutPotionBase);
        setIf("rollout_non_card_base", g_params.rolloutNonCardBase);
        setIf("rollout_potion_danger_scale", g_params.rolloutPotionDangerScale);
        setIf("rollout_potion_finish_off_scale", g_params.rolloutPotionFinishOffScale);
        setIf("rollout_potion_discard_penalty", g_params.rolloutPotionDiscardPenalty);
        setIf("mast_weight", g_params.mastWeight);
        setIf("mast_min_visits", g_params.mastMinVisits);
        setIf("seq_halving_candidates", g_params.seqHalvingCandidates);
        setIf("backup_max_weight", g_params.backupMaxWeight);
        setIf("honest_draw_order", g_params.honestDrawOrder);
        setIf("search_max_turns", g_params.searchMaxTurns);
        setIf("power_score", g_params.powerScore);
        setIf("end_turn_time_warp_risk_score", g_params.endTurnTimeWarpRiskScore);
        setIf("skill_haste_danger_threshold", g_params.skillHasteDangerThreshold);
        setIf("per_card_weight_scale", g_params.perCardWeightScale);
        setIf("card_play_prior_weight", g_params.cardPlayPriorWeight);
        setIf("boss_card_play_prior_weight", g_params.bossCardPlayPriorWeight);
        setIf("paired_determinization", g_params.pairedDeterminization);
        setIf("merge_duplicate_actions", g_params.mergeDuplicateActions);
        setIf("escalation_sims", g_params.escalationSims);
        setIf("escalation_qgap", g_params.escalationQgap);
        setIf("escalation_danger_frac", g_params.escalationDangerFrac);
        setIf("merge_chance_outcomes", g_params.mergeChanceOutcomes);
        setIf("intangible_attack_penalty", g_params.intangibleAttackPenalty);
        setIf("artifact_aware_debuffs", g_params.artifactAwareDebuffs);
        setIf("tree_reuse", g_params.treeReuse);
        setIf("draw_first_bonus", g_params.drawFirstBonus);
        setIf("burst_debuff_timing_weight", g_params.burstDebuffTimingWeight);
        setIf("max_hp_gain_weight", g_params.maxHpGainWeight);
        setIf("gold_delta_weight", g_params.goldDeltaWeight);
        setIf("parasite_penalty_weight", g_params.parasitePenaltyWeight);
        setIf("survival_mode_threshold", g_params.survivalModeThreshold);
        setIf("survival_mode_attack_scale", g_params.survivalModeAttackScale);
        setIf("honest_wc_chance", g_params.honestWcChance);
        setIf("honest_wa_chance", g_params.honestWaChance);
        setIf("c_puct", g_params.cPuct);
        setIf("puct_temperature", g_params.puctTemperature);
        setIf("policy_net_weight", g_params.policyNetWeight);
        setIf("block_sufficiency_margin", g_params.blockSufficiencyMargin);
        setIf("defensive_card_suppression_penalty", g_params.defensiveCardSuppressionPenalty);
        setIf("vf_hp", g_params.vfHp);
        setIf("vf_monster_hp", g_params.vfMonsterHp);
        setIf("vf_incoming", g_params.vfIncoming);
        setIf("vf_block", g_params.vfBlock);
        setIf("vf_energy", g_params.vfEnergy);
        setIf("vf_strength", g_params.vfStrength);
        setIf("vf_dexterity", g_params.vfDexterity);
        setIf("vf_alive", g_params.vfAlive);
        setIf("vf_turn", g_params.vfTurn);
        setIf("vf_metallicize", g_params.vfMetallicize);
    }, pybind11::arg("params"),
       "Overwrite any subset of the runtime-tunable search parameters (unset keys keep their "
       "current value -- this is a partial update, not a full replace) -- see "
       "get_search_params for the full key list and TunableParams' own comment (above "
       "BattleContext's binding) for what each one does and why they're runtime-mutable at "
       "all. THREAD-SAFETY: g_params is process-global mutable state with no locking -- never "
       "call this while another thread has an in-flight run_mcts_search/root_parallel_search "
       "call still running against the OLD values (see run_mcts_search's own docstring for "
       "the safe calling pattern). Intended caller: lightspeed/tune_search_cma.py.");

    m.def("set_early_act_card_biases", [](pybind11::dict biases) {
        // Replace, rather than merge, so every tuner candidate starts with
        // exactly its own sparse vector. Same process-global sequencing rule
        // as g_params: call before search, never during an in-flight search.
        g_earlyActCardBias.fill(0.0);
        for (const auto &entry : biases) {
            const int cardId = entry.first.cast<int>();
            if (cardId < 0 || cardId >= static_cast<int>(g_earlyActCardBias.size())) {
                throw pybind11::value_error("early-act card bias key is outside CardId range");
            }
            g_earlyActCardBias[cardId] = entry.second.cast<double>();
        }
        g_hasEarlyActCardBias = !biases.empty();
    }, pybind11::arg("biases"),
       "Replace sparse per-CardId rollout bonuses used only while player max HP is <= 85 "
       "(the calibrated Act 1 range). Keys are integer CardId values and values are additive "
       "heuristic-score corrections. Process-global and not safe to change during search.");

    m.def("get_early_act_card_biases", []() {
        pybind11::dict biases;
        for (int cardId = 0; cardId < static_cast<int>(g_earlyActCardBias.size()); ++cardId) {
            if (g_earlyActCardBias[cardId] != 0.0) {
                biases[pybind11::int_(cardId)] = g_earlyActCardBias[cardId];
            }
        }
        return biases;
    }, "Current nonzero sparse Act-1 per-card rollout corrections.");

    m.def("set_rave", [](bool enabled) { g_useRave = enabled; }, pybind11::arg("enabled"),
          "Enable RAVE/AMAF blending in selection (see MctsNode::amafN). Off by default. Changes "
          "what selection statistics mean, so re-tune/A-B rather than assuming a parameter set "
          "carries over. Same process-global thread-safety rule as set_search_params.");
    m.def("get_rave", []() { return g_useRave; }, "true if RAVE blending is enabled");

    m.def("set_seq_halving", [](bool enabled) { g_useSeqHalving = enabled; },
          pybind11::arg("enabled"),
          "Enable sequential-halving root allocation instead of plain UCB1 over the whole budget "
          "(see nativeRunMctsSearchSeqHalving). Off by default. Root visit counts mean something "
          "different under it (survivors get equal pulls by construction), so a parameter set "
          "tuned with it off is not automatically valid with it on -- A/B them separately. Same "
          "process-global thread-safety rule as set_search_params: set before any search runs.");
    m.def("get_seq_halving", []() { return g_useSeqHalving; },
          "true if sequential-halving root allocation is enabled");

    m.def("set_state_merging", [](bool enabled) { g_useStateMerging = enabled; },
          pybind11::arg("enabled"),
          "Enable compact-key state merging. Disabled by default because NativeStateKey is not "
          "yet a complete BattleContext identity; enabling it can trade search correctness for "
          "speed. Only use in explicitly measured experiments.");
    m.def("get_state_merging", []() { return g_useStateMerging; },
          "true if compact-key state merging is enabled");

    m.def("set_leaf_eval_mode", [](const std::string &mode, int truncatedSteps) {
        if (mode == "rollout") {
            g_leafEvalMode = LeafEvalMode::ROLLOUT;
        } else if (mode == "value") {
            g_leafEvalMode = LeafEvalMode::VALUE;
        } else if (mode == "truncated") {
            g_leafEvalMode = LeafEvalMode::TRUNCATED;
            g_truncatedRolloutSteps = truncatedSteps;
        } else if (mode == "valuenet") {
            if (!g_valueNet.loaded) {
                throw std::invalid_argument("valuenet mode requires load_value_net() first");
            }
            g_leafEvalMode = LeafEvalMode::VALUENET;
        } else {
            throw std::invalid_argument("mode must be 'rollout', 'value', 'truncated', or 'valuenet'");
        }
    }, pybind11::arg("mode"), pybind11::arg("truncated_steps") = 3,
       "Select how newly-expanded MCTS leaves are evaluated: 'rollout' (default -- full "
       "heuristic playout to terminal, most accurate but ~90% of search time), 'value' (skip "
       "the playout, return the tunable linear nativeLeafValueEstimate -- ~4x cheaper per leaf, "
       "coarser), 'truncated' (play truncated_steps actions then apply the linear estimate -- a "
       "middle ground), or 'valuenet' (trained MLP leaf estimate, needs load_value_net first). "
       "Same process-global-mutable-state thread-safety rule as set_search_params: set once "
       "before an evaluation, never mid-flight.");

    m.def("load_value_net", [](pybind11::dict d) {
        ValueNet net;
        auto mu = d["input_mu"].cast<std::vector<double>>();
        auto sd = d["input_sd"].cast<std::vector<double>>();
        if (static_cast<int>(mu.size()) != NATIVE_LEAF_FEATURE_DIM
            || static_cast<int>(sd.size()) != NATIVE_LEAF_FEATURE_DIM) {
            throw std::invalid_argument("input_mu/input_sd must have NATIVE_LEAF_FEATURE_DIM entries");
        }
        for (int i = 0; i < NATIVE_LEAF_FEATURE_DIM; ++i) {
            net.mu[i] = mu[i];
            net.sd[i] = sd[i];
        }
        for (auto layerObj : d["layers"]) {
            auto layerDict = layerObj.cast<pybind11::dict>();
            ValueNetLayer layer;
            layer.W = layerDict["W"].cast<std::vector<std::vector<double>>>();
            layer.b = layerDict["b"].cast<std::vector<double>>();
            layer.applyTanh = layerDict["activation"].cast<std::string>() == "tanh";
            net.layers.push_back(std::move(layer));
        }
        net.loaded = true;
        g_valueNet = std::move(net);
    }, pybind11::arg("weights"),
       "Load a trained leaf value-net (train_value_net.py's exported dict: input_mu, input_sd, "
       "layers=[{W,b,activation}]) for use with set_leaf_eval_mode('valuenet'). Same process-"
       "global thread-safety rule as set_search_params -- load before any concurrent search.");

    m.def("load_policy_net", [](pybind11::dict d) {
        PolicyNet net;
        net.mu = d["input_mu"].cast<std::vector<double>>();
        net.sd = d["input_sd"].cast<std::vector<double>>();
        const int expectedDim = NATIVE_LEAF_FEATURE_DIM + NATIVE_ACTION_FEATURE_DIM;
        if (static_cast<int>(net.mu.size()) != expectedDim || static_cast<int>(net.sd.size()) != expectedDim) {
            throw std::invalid_argument("input_mu/input_sd must have NATIVE_LEAF_FEATURE_DIM + "
                                         "NATIVE_ACTION_FEATURE_DIM entries");
        }
        for (auto layerObj : d["layers"]) {
            auto layerDict = layerObj.cast<pybind11::dict>();
            ValueNetLayer layer;
            layer.W = layerDict["W"].cast<std::vector<std::vector<double>>>();
            layer.b = layerDict["b"].cast<std::vector<double>>();
            layer.applyTanh = layerDict["activation"].cast<std::string>() == "tanh";
            net.layers.push_back(std::move(layer));
        }
        net.loaded = true;
        g_policyNet = std::move(net);
    }, pybind11::arg("weights"),
       "Load a trained rollout-scoring net (train_policy_net.py's exported dict: input_mu, "
       "input_sd, layers=[{W,b,activation}]) blended into nativeScoreAction via "
       "g_params.policy_net_weight (0.0 = off by default, see set_search_params). Same process-"
       "global thread-safety rule as set_search_params/load_value_net.");

    m.def("get_leaf_eval_mode", []() {
        const char *mode = g_leafEvalMode == LeafEvalMode::ROLLOUT ? "rollout"
                         : g_leafEvalMode == LeafEvalMode::VALUE ? "value"
                         : g_leafEvalMode == LeafEvalMode::VALUENET ? "valuenet" : "truncated";
        return pybind11::make_tuple(std::string(mode), g_truncatedRolloutSteps);
    }, "Returns (mode, truncated_steps) -- the current leaf-evaluation setting, see set_leaf_eval_mode.");

    pybind11::enum_<GameOutcome> gameOutcome(m, "GameOutcome");
    gameOutcome.value("UNDECIDED", GameOutcome::UNDECIDED)
        .value("PLAYER_VICTORY", GameOutcome::PLAYER_VICTORY)
        .value("PLAYER_LOSS", GameOutcome::PLAYER_LOSS);

    pybind11::enum_<ScreenState> screenState(m, "ScreenState", pybind11::metaclass(enum_metaclass));
    screenState.value("INVALID", ScreenState::INVALID)
        .value("EVENT_SCREEN", ScreenState::EVENT_SCREEN)
        .value("REWARDS", ScreenState::REWARDS)
        .value("BOSS_RELIC_REWARDS", ScreenState::BOSS_RELIC_REWARDS)
        .value("CARD_SELECT", ScreenState::CARD_SELECT)
        .value("MAP_SCREEN", ScreenState::MAP_SCREEN)
        .value("TREASURE_ROOM", ScreenState::TREASURE_ROOM)
        .value("REST_ROOM", ScreenState::REST_ROOM)
        .value("SHOP_ROOM", ScreenState::SHOP_ROOM)
        .value("BATTLE", ScreenState::BATTLE);

    pybind11::enum_<CharacterClass> characterClass(m, "CharacterClass");
    characterClass.value("IRONCLAD", CharacterClass::IRONCLAD)
            .value("SILENT", CharacterClass::SILENT)
            .value("DEFECT", CharacterClass::DEFECT)
            .value("WATCHER", CharacterClass::WATCHER)
            .value("INVALID", CharacterClass::INVALID);

    pybind11::enum_<Room> roomEnum(m, "Room", pybind11::metaclass(enum_metaclass));
    roomEnum.value("SHOP", Room::SHOP)
        .value("REST", Room::REST)
        .value("EVENT", Room::EVENT)
        .value("ELITE", Room::ELITE)
        .value("MONSTER", Room::MONSTER)
        .value("TREASURE", Room::TREASURE)
        .value("BOSS", Room::BOSS)
        .value("BOSS_TREASURE", Room::BOSS_TREASURE)
        .value("NONE", Room::NONE)
        .value("INVALID", Room::INVALID);

    pybind11::enum_<CardRarity>(m, "CardRarity")
        .value("COMMON", CardRarity::COMMON)
        .value("UNCOMMON", CardRarity::UNCOMMON)
        .value("RARE", CardRarity::RARE)
        .value("BASIC", CardRarity::BASIC)
        .value("SPECIAL", CardRarity::SPECIAL)
        .value("CURSE", CardRarity::CURSE)
        .value("INVALID", CardRarity::INVALID);

    pybind11::enum_<CardColor>(m, "CardColor")
        .value("RED", CardColor::RED)
        .value("GREEN", CardColor::GREEN)
        .value("BLUE", CardColor::BLUE)
        .value("PURPLE", CardColor::PURPLE)
        .value("COLORLESS", CardColor::COLORLESS)
        .value("CURSE", CardColor::CURSE)
        .value("INVALID", CardColor::INVALID);

    pybind11::enum_<CardType>(m, "CardType")
        .value("ATTACK", CardType::ATTACK)
        .value("SKILL", CardType::SKILL)
        .value("POWER", CardType::POWER)
        .value("CURSE", CardType::CURSE)
        .value("STATUS", CardType::STATUS)
        .value("INVALID", CardType::INVALID);

    pybind11::enum_<CardId>(m, "CardId", pybind11::metaclass(enum_metaclass))
        .value("INVALID", CardId::INVALID)
        .value("ACCURACY", CardId::ACCURACY)
        .value("ACROBATICS", CardId::ACROBATICS)
        .value("ADRENALINE", CardId::ADRENALINE)
        .value("AFTER_IMAGE", CardId::AFTER_IMAGE)
        .value("AGGREGATE", CardId::AGGREGATE)
        .value("ALCHEMIZE", CardId::ALCHEMIZE)
        .value("ALL_FOR_ONE", CardId::ALL_FOR_ONE)
        .value("ALL_OUT_ATTACK", CardId::ALL_OUT_ATTACK)
        .value("ALPHA", CardId::ALPHA)
        .value("AMPLIFY", CardId::AMPLIFY)
        .value("ANGER", CardId::ANGER)
        .value("APOTHEOSIS", CardId::APOTHEOSIS)
        .value("APPARITION", CardId::APPARITION)
        .value("ARMAMENTS", CardId::ARMAMENTS)
        .value("ASCENDERS_BANE", CardId::ASCENDERS_BANE)
        .value("AUTO_SHIELDS", CardId::AUTO_SHIELDS)
        .value("A_THOUSAND_CUTS", CardId::A_THOUSAND_CUTS)
        .value("BACKFLIP", CardId::BACKFLIP)
        .value("BACKSTAB", CardId::BACKSTAB)
        .value("BALL_LIGHTNING", CardId::BALL_LIGHTNING)
        .value("BANDAGE_UP", CardId::BANDAGE_UP)
        .value("BANE", CardId::BANE)
        .value("BARRAGE", CardId::BARRAGE)
        .value("BARRICADE", CardId::BARRICADE)
        .value("BASH", CardId::BASH)
        .value("BATTLE_HYMN", CardId::BATTLE_HYMN)
        .value("BATTLE_TRANCE", CardId::BATTLE_TRANCE)
        .value("BEAM_CELL", CardId::BEAM_CELL)
        .value("BECOME_ALMIGHTY", CardId::BECOME_ALMIGHTY)
        .value("BERSERK", CardId::BERSERK)
        .value("BETA", CardId::BETA)
        .value("BIASED_COGNITION", CardId::BIASED_COGNITION)
        .value("BITE", CardId::BITE)
        .value("BLADE_DANCE", CardId::BLADE_DANCE)
        .value("BLASPHEMY", CardId::BLASPHEMY)
        .value("BLIND", CardId::BLIND)
        .value("BLIZZARD", CardId::BLIZZARD)
        .value("BLOODLETTING", CardId::BLOODLETTING)
        .value("BLOOD_FOR_BLOOD", CardId::BLOOD_FOR_BLOOD)
        .value("BLUDGEON", CardId::BLUDGEON)
        .value("BLUR", CardId::BLUR)
        .value("BODY_SLAM", CardId::BODY_SLAM)
        .value("BOOT_SEQUENCE", CardId::BOOT_SEQUENCE)
        .value("BOUNCING_FLASK", CardId::BOUNCING_FLASK)
        .value("BOWLING_BASH", CardId::BOWLING_BASH)
        .value("BRILLIANCE", CardId::BRILLIANCE)
        .value("BRUTALITY", CardId::BRUTALITY)
        .value("BUFFER", CardId::BUFFER)
        .value("BULLET_TIME", CardId::BULLET_TIME)
        .value("BULLSEYE", CardId::BULLSEYE)
        .value("BURN", CardId::BURN)
        .value("BURNING_PACT", CardId::BURNING_PACT)
        .value("BURST", CardId::BURST)
        .value("CALCULATED_GAMBLE", CardId::CALCULATED_GAMBLE)
        .value("CALTROPS", CardId::CALTROPS)
        .value("CAPACITOR", CardId::CAPACITOR)
        .value("CARNAGE", CardId::CARNAGE)
        .value("CARVE_REALITY", CardId::CARVE_REALITY)
        .value("CATALYST", CardId::CATALYST)
        .value("CHAOS", CardId::CHAOS)
        .value("CHARGE_BATTERY", CardId::CHARGE_BATTERY)
        .value("CHILL", CardId::CHILL)
        .value("CHOKE", CardId::CHOKE)
        .value("CHRYSALIS", CardId::CHRYSALIS)
        .value("CLASH", CardId::CLASH)
        .value("CLAW", CardId::CLAW)
        .value("CLEAVE", CardId::CLEAVE)
        .value("CLOAK_AND_DAGGER", CardId::CLOAK_AND_DAGGER)
        .value("CLOTHESLINE", CardId::CLOTHESLINE)
        .value("CLUMSY", CardId::CLUMSY)
        .value("COLD_SNAP", CardId::COLD_SNAP)
        .value("COLLECT", CardId::COLLECT)
        .value("COMBUST", CardId::COMBUST)
        .value("COMPILE_DRIVER", CardId::COMPILE_DRIVER)
        .value("CONCENTRATE", CardId::CONCENTRATE)
        .value("CONCLUDE", CardId::CONCLUDE)
        .value("CONJURE_BLADE", CardId::CONJURE_BLADE)
        .value("CONSECRATE", CardId::CONSECRATE)
        .value("CONSUME", CardId::CONSUME)
        .value("COOLHEADED", CardId::COOLHEADED)
        .value("CORE_SURGE", CardId::CORE_SURGE)
        .value("CORPSE_EXPLOSION", CardId::CORPSE_EXPLOSION)
        .value("CORRUPTION", CardId::CORRUPTION)
        .value("CREATIVE_AI", CardId::CREATIVE_AI)
        .value("CRESCENDO", CardId::CRESCENDO)
        .value("CRIPPLING_CLOUD", CardId::CRIPPLING_CLOUD)
        .value("CRUSH_JOINTS", CardId::CRUSH_JOINTS)
        .value("CURSE_OF_THE_BELL", CardId::CURSE_OF_THE_BELL)
        .value("CUT_THROUGH_FATE", CardId::CUT_THROUGH_FATE)
        .value("DAGGER_SPRAY", CardId::DAGGER_SPRAY)
        .value("DAGGER_THROW", CardId::DAGGER_THROW)
        .value("DARKNESS", CardId::DARKNESS)
        .value("DARK_EMBRACE", CardId::DARK_EMBRACE)
        .value("DARK_SHACKLES", CardId::DARK_SHACKLES)
        .value("DASH", CardId::DASH)
        .value("DAZED", CardId::DAZED)
        .value("DEADLY_POISON", CardId::DEADLY_POISON)
        .value("DECAY", CardId::DECAY)
        .value("DECEIVE_REALITY", CardId::DECEIVE_REALITY)
        .value("DEEP_BREATH", CardId::DEEP_BREATH)
        .value("DEFEND_BLUE", CardId::DEFEND_BLUE)
        .value("DEFEND_GREEN", CardId::DEFEND_GREEN)
        .value("DEFEND_PURPLE", CardId::DEFEND_PURPLE)
        .value("DEFEND_RED", CardId::DEFEND_RED)
        .value("DEFLECT", CardId::DEFLECT)
        .value("DEFRAGMENT", CardId::DEFRAGMENT)
        .value("DEMON_FORM", CardId::DEMON_FORM)
        .value("DEUS_EX_MACHINA", CardId::DEUS_EX_MACHINA)
        .value("DEVA_FORM", CardId::DEVA_FORM)
        .value("DEVOTION", CardId::DEVOTION)
        .value("DIE_DIE_DIE", CardId::DIE_DIE_DIE)
        .value("DISARM", CardId::DISARM)
        .value("DISCOVERY", CardId::DISCOVERY)
        .value("DISTRACTION", CardId::DISTRACTION)
        .value("DODGE_AND_ROLL", CardId::DODGE_AND_ROLL)
        .value("DOOM_AND_GLOOM", CardId::DOOM_AND_GLOOM)
        .value("DOPPELGANGER", CardId::DOPPELGANGER)
        .value("DOUBLE_ENERGY", CardId::DOUBLE_ENERGY)
        .value("DOUBLE_TAP", CardId::DOUBLE_TAP)
        .value("DOUBT", CardId::DOUBT)
        .value("DRAMATIC_ENTRANCE", CardId::DRAMATIC_ENTRANCE)
        .value("DROPKICK", CardId::DROPKICK)
        .value("DUALCAST", CardId::DUALCAST)
        .value("DUAL_WIELD", CardId::DUAL_WIELD)
        .value("ECHO_FORM", CardId::ECHO_FORM)
        .value("ELECTRODYNAMICS", CardId::ELECTRODYNAMICS)
        .value("EMPTY_BODY", CardId::EMPTY_BODY)
        .value("EMPTY_FIST", CardId::EMPTY_FIST)
        .value("EMPTY_MIND", CardId::EMPTY_MIND)
        .value("ENDLESS_AGONY", CardId::ENDLESS_AGONY)
        .value("ENLIGHTENMENT", CardId::ENLIGHTENMENT)
        .value("ENTRENCH", CardId::ENTRENCH)
        .value("ENVENOM", CardId::ENVENOM)
        .value("EQUILIBRIUM", CardId::EQUILIBRIUM)
        .value("ERUPTION", CardId::ERUPTION)
        .value("ESCAPE_PLAN", CardId::ESCAPE_PLAN)
        .value("ESTABLISHMENT", CardId::ESTABLISHMENT)
        .value("EVALUATE", CardId::EVALUATE)
        .value("EVISCERATE", CardId::EVISCERATE)
        .value("EVOLVE", CardId::EVOLVE)
        .value("EXHUME", CardId::EXHUME)
        .value("EXPERTISE", CardId::EXPERTISE)
        .value("EXPUNGER", CardId::EXPUNGER)
        .value("FAME_AND_FORTUNE", CardId::FAME_AND_FORTUNE)
        .value("FASTING", CardId::FASTING)
        .value("FEAR_NO_EVIL", CardId::FEAR_NO_EVIL)
        .value("FEED", CardId::FEED)
        .value("FEEL_NO_PAIN", CardId::FEEL_NO_PAIN)
        .value("FIEND_FIRE", CardId::FIEND_FIRE)
        .value("FINESSE", CardId::FINESSE)
        .value("FINISHER", CardId::FINISHER)
        .value("FIRE_BREATHING", CardId::FIRE_BREATHING)
        .value("FISSION", CardId::FISSION)
        .value("FLAME_BARRIER", CardId::FLAME_BARRIER)
        .value("FLASH_OF_STEEL", CardId::FLASH_OF_STEEL)
        .value("FLECHETTES", CardId::FLECHETTES)
        .value("FLEX", CardId::FLEX)
        .value("FLURRY_OF_BLOWS", CardId::FLURRY_OF_BLOWS)
        .value("FLYING_KNEE", CardId::FLYING_KNEE)
        .value("FLYING_SLEEVES", CardId::FLYING_SLEEVES)
        .value("FOLLOW_UP", CardId::FOLLOW_UP)
        .value("FOOTWORK", CardId::FOOTWORK)
        .value("FORCE_FIELD", CardId::FORCE_FIELD)
        .value("FOREIGN_INFLUENCE", CardId::FOREIGN_INFLUENCE)
        .value("FORESIGHT", CardId::FORESIGHT)
        .value("FORETHOUGHT", CardId::FORETHOUGHT)
        .value("FTL", CardId::FTL)
        .value("FUSION", CardId::FUSION)
        .value("GENETIC_ALGORITHM", CardId::GENETIC_ALGORITHM)
        .value("GHOSTLY_ARMOR", CardId::GHOSTLY_ARMOR)
        .value("GLACIER", CardId::GLACIER)
        .value("GLASS_KNIFE", CardId::GLASS_KNIFE)
        .value("GOOD_INSTINCTS", CardId::GOOD_INSTINCTS)
        .value("GO_FOR_THE_EYES", CardId::GO_FOR_THE_EYES)
        .value("GRAND_FINALE", CardId::GRAND_FINALE)
        .value("HALT", CardId::HALT)
        .value("HAND_OF_GREED", CardId::HAND_OF_GREED)
        .value("HAVOC", CardId::HAVOC)
        .value("HEADBUTT", CardId::HEADBUTT)
        .value("HEATSINKS", CardId::HEATSINKS)
        .value("HEAVY_BLADE", CardId::HEAVY_BLADE)
        .value("HEEL_HOOK", CardId::HEEL_HOOK)
        .value("HELLO_WORLD", CardId::HELLO_WORLD)
        .value("HEMOKINESIS", CardId::HEMOKINESIS)
        .value("HOLOGRAM", CardId::HOLOGRAM)
        .value("HYPERBEAM", CardId::HYPERBEAM)
        .value("IMMOLATE", CardId::IMMOLATE)
        .value("IMPATIENCE", CardId::IMPATIENCE)
        .value("IMPERVIOUS", CardId::IMPERVIOUS)
        .value("INDIGNATION", CardId::INDIGNATION)
        .value("INFERNAL_BLADE", CardId::INFERNAL_BLADE)
        .value("INFINITE_BLADES", CardId::INFINITE_BLADES)
        .value("INFLAME", CardId::INFLAME)
        .value("INJURY", CardId::INJURY)
        .value("INNER_PEACE", CardId::INNER_PEACE)
        .value("INSIGHT", CardId::INSIGHT)
        .value("INTIMIDATE", CardId::INTIMIDATE)
        .value("IRON_WAVE", CardId::IRON_WAVE)
        .value("JAX", CardId::JAX)
        .value("JACK_OF_ALL_TRADES", CardId::JACK_OF_ALL_TRADES)
        .value("JUDGMENT", CardId::JUDGMENT)
        .value("JUGGERNAUT", CardId::JUGGERNAUT)
        .value("JUST_LUCKY", CardId::JUST_LUCKY)
        .value("LEAP", CardId::LEAP)
        .value("LEG_SWEEP", CardId::LEG_SWEEP)
        .value("LESSON_LEARNED", CardId::LESSON_LEARNED)
        .value("LIKE_WATER", CardId::LIKE_WATER)
        .value("LIMIT_BREAK", CardId::LIMIT_BREAK)
        .value("LIVE_FOREVER", CardId::LIVE_FOREVER)
        .value("LOOP", CardId::LOOP)
        .value("MACHINE_LEARNING", CardId::MACHINE_LEARNING)
        .value("MADNESS", CardId::MADNESS)
        .value("MAGNETISM", CardId::MAGNETISM)
        .value("MALAISE", CardId::MALAISE)
        .value("MASTERFUL_STAB", CardId::MASTERFUL_STAB)
        .value("MASTER_OF_STRATEGY", CardId::MASTER_OF_STRATEGY)
        .value("MASTER_REALITY", CardId::MASTER_REALITY)
        .value("MAYHEM", CardId::MAYHEM)
        .value("MEDITATE", CardId::MEDITATE)
        .value("MELTER", CardId::MELTER)
        .value("MENTAL_FORTRESS", CardId::MENTAL_FORTRESS)
        .value("METALLICIZE", CardId::METALLICIZE)
        .value("METAMORPHOSIS", CardId::METAMORPHOSIS)
        .value("METEOR_STRIKE", CardId::METEOR_STRIKE)
        .value("MIND_BLAST", CardId::MIND_BLAST)
        .value("MIRACLE", CardId::MIRACLE)
        .value("MULTI_CAST", CardId::MULTI_CAST)
        .value("NECRONOMICURSE", CardId::NECRONOMICURSE)
        .value("NEUTRALIZE", CardId::NEUTRALIZE)
        .value("NIGHTMARE", CardId::NIGHTMARE)
        .value("NIRVANA", CardId::NIRVANA)
        .value("NORMALITY", CardId::NORMALITY)
        .value("NOXIOUS_FUMES", CardId::NOXIOUS_FUMES)
        .value("OFFERING", CardId::OFFERING)
        .value("OMEGA", CardId::OMEGA)
        .value("OMNISCIENCE", CardId::OMNISCIENCE)
        .value("OUTMANEUVER", CardId::OUTMANEUVER)
        .value("OVERCLOCK", CardId::OVERCLOCK)
        .value("PAIN", CardId::PAIN)
        .value("PANACEA", CardId::PANACEA)
        .value("PANACHE", CardId::PANACHE)
        .value("PANIC_BUTTON", CardId::PANIC_BUTTON)
        .value("PARASITE", CardId::PARASITE)
        .value("PERFECTED_STRIKE", CardId::PERFECTED_STRIKE)
        .value("PERSEVERANCE", CardId::PERSEVERANCE)
        .value("PHANTASMAL_KILLER", CardId::PHANTASMAL_KILLER)
        .value("PIERCING_WAIL", CardId::PIERCING_WAIL)
        .value("POISONED_STAB", CardId::POISONED_STAB)
        .value("POMMEL_STRIKE", CardId::POMMEL_STRIKE)
        .value("POWER_THROUGH", CardId::POWER_THROUGH)
        .value("PRAY", CardId::PRAY)
        .value("PREDATOR", CardId::PREDATOR)
        .value("PREPARED", CardId::PREPARED)
        .value("PRESSURE_POINTS", CardId::PRESSURE_POINTS)
        .value("PRIDE", CardId::PRIDE)
        .value("PROSTRATE", CardId::PROSTRATE)
        .value("PROTECT", CardId::PROTECT)
        .value("PUMMEL", CardId::PUMMEL)
        .value("PURITY", CardId::PURITY)
        .value("QUICK_SLASH", CardId::QUICK_SLASH)
        .value("RAGE", CardId::RAGE)
        .value("RAGNAROK", CardId::RAGNAROK)
        .value("RAINBOW", CardId::RAINBOW)
        .value("RAMPAGE", CardId::RAMPAGE)
        .value("REACH_HEAVEN", CardId::REACH_HEAVEN)
        .value("REAPER", CardId::REAPER)
        .value("REBOOT", CardId::REBOOT)
        .value("REBOUND", CardId::REBOUND)
        .value("RECKLESS_CHARGE", CardId::RECKLESS_CHARGE)
        .value("RECURSION", CardId::RECURSION)
        .value("RECYCLE", CardId::RECYCLE)
        .value("REFLEX", CardId::REFLEX)
        .value("REGRET", CardId::REGRET)
        .value("REINFORCED_BODY", CardId::REINFORCED_BODY)
        .value("REPROGRAM", CardId::REPROGRAM)
        .value("RIDDLE_WITH_HOLES", CardId::RIDDLE_WITH_HOLES)
        .value("RIP_AND_TEAR", CardId::RIP_AND_TEAR)
        .value("RITUAL_DAGGER", CardId::RITUAL_DAGGER)
        .value("RUPTURE", CardId::RUPTURE)
        .value("RUSHDOWN", CardId::RUSHDOWN)
        .value("SADISTIC_NATURE", CardId::SADISTIC_NATURE)
        .value("SAFETY", CardId::SAFETY)
        .value("SANCTITY", CardId::SANCTITY)
        .value("SANDS_OF_TIME", CardId::SANDS_OF_TIME)
        .value("SASH_WHIP", CardId::SASH_WHIP)
        .value("SCRAPE", CardId::SCRAPE)
        .value("SCRAWL", CardId::SCRAWL)
        .value("SEARING_BLOW", CardId::SEARING_BLOW)
        .value("SECOND_WIND", CardId::SECOND_WIND)
        .value("SECRET_TECHNIQUE", CardId::SECRET_TECHNIQUE)
        .value("SECRET_WEAPON", CardId::SECRET_WEAPON)
        .value("SEEING_RED", CardId::SEEING_RED)
        .value("SEEK", CardId::SEEK)
        .value("SELF_REPAIR", CardId::SELF_REPAIR)
        .value("SENTINEL", CardId::SENTINEL)
        .value("SETUP", CardId::SETUP)
        .value("SEVER_SOUL", CardId::SEVER_SOUL)
        .value("SHAME", CardId::SHAME)
        .value("SHIV", CardId::SHIV)
        .value("SHOCKWAVE", CardId::SHOCKWAVE)
        .value("SHRUG_IT_OFF", CardId::SHRUG_IT_OFF)
        .value("SIGNATURE_MOVE", CardId::SIGNATURE_MOVE)
        .value("SIMMERING_FURY", CardId::SIMMERING_FURY)
        .value("SKEWER", CardId::SKEWER)
        .value("SKIM", CardId::SKIM)
        .value("SLICE", CardId::SLICE)
        .value("SLIMED", CardId::SLIMED)
        .value("SMITE", CardId::SMITE)
        .value("SNEAKY_STRIKE", CardId::SNEAKY_STRIKE)
        .value("SPIRIT_SHIELD", CardId::SPIRIT_SHIELD)
        .value("SPOT_WEAKNESS", CardId::SPOT_WEAKNESS)
        .value("STACK", CardId::STACK)
        .value("STATIC_DISCHARGE", CardId::STATIC_DISCHARGE)
        .value("STEAM_BARRIER", CardId::STEAM_BARRIER)
        .value("STORM", CardId::STORM)
        .value("STORM_OF_STEEL", CardId::STORM_OF_STEEL)
        .value("STREAMLINE", CardId::STREAMLINE)
        .value("STRIKE_BLUE", CardId::STRIKE_BLUE)
        .value("STRIKE_GREEN", CardId::STRIKE_GREEN)
        .value("STRIKE_PURPLE", CardId::STRIKE_PURPLE)
        .value("STRIKE_RED", CardId::STRIKE_RED)
        .value("STUDY", CardId::STUDY)
        .value("SUCKER_PUNCH", CardId::SUCKER_PUNCH)
        .value("SUNDER", CardId::SUNDER)
        .value("SURVIVOR", CardId::SURVIVOR)
        .value("SWEEPING_BEAM", CardId::SWEEPING_BEAM)
        .value("SWIFT_STRIKE", CardId::SWIFT_STRIKE)
        .value("SWIVEL", CardId::SWIVEL)
        .value("SWORD_BOOMERANG", CardId::SWORD_BOOMERANG)
        .value("TACTICIAN", CardId::TACTICIAN)
        .value("TALK_TO_THE_HAND", CardId::TALK_TO_THE_HAND)
        .value("TANTRUM", CardId::TANTRUM)
        .value("TEMPEST", CardId::TEMPEST)
        .value("TERROR", CardId::TERROR)
        .value("THE_BOMB", CardId::THE_BOMB)
        .value("THINKING_AHEAD", CardId::THINKING_AHEAD)
        .value("THIRD_EYE", CardId::THIRD_EYE)
        .value("THROUGH_VIOLENCE", CardId::THROUGH_VIOLENCE)
        .value("THUNDERCLAP", CardId::THUNDERCLAP)
        .value("THUNDER_STRIKE", CardId::THUNDER_STRIKE)
        .value("TOOLS_OF_THE_TRADE", CardId::TOOLS_OF_THE_TRADE)
        .value("TRANQUILITY", CardId::TRANQUILITY)
        .value("TRANSMUTATION", CardId::TRANSMUTATION)
        .value("TRIP", CardId::TRIP)
        .value("TRUE_GRIT", CardId::TRUE_GRIT)
        .value("TURBO", CardId::TURBO)
        .value("TWIN_STRIKE", CardId::TWIN_STRIKE)
        .value("UNLOAD", CardId::UNLOAD)
        .value("UPPERCUT", CardId::UPPERCUT)
        .value("VAULT", CardId::VAULT)
        .value("VIGILANCE", CardId::VIGILANCE)
        .value("VIOLENCE", CardId::VIOLENCE)
        .value("VOID", CardId::VOID)
        .value("WALLOP", CardId::WALLOP)
        .value("WARCRY", CardId::WARCRY)
        .value("WAVE_OF_THE_HAND", CardId::WAVE_OF_THE_HAND)
        .value("WEAVE", CardId::WEAVE)
        .value("WELL_LAID_PLANS", CardId::WELL_LAID_PLANS)
        .value("WHEEL_KICK", CardId::WHEEL_KICK)
        .value("WHIRLWIND", CardId::WHIRLWIND)
        .value("WHITE_NOISE", CardId::WHITE_NOISE)
        .value("WILD_STRIKE", CardId::WILD_STRIKE)
        .value("WINDMILL_STRIKE", CardId::WINDMILL_STRIKE)
        .value("WISH", CardId::WISH)
        .value("WORSHIP", CardId::WORSHIP)
        .value("WOUND", CardId::WOUND)
        .value("WRAITH_FORM", CardId::WRAITH_FORM)
        .value("WREATH_OF_FLAME", CardId::WREATH_OF_FLAME)
        .value("WRITHE", CardId::WRITHE)
        .value("ZAP", CardId::ZAP);

    pybind11::enum_<MonsterEncounter> meEnum(m, "MonsterEncounter", pybind11::metaclass(enum_metaclass));
    meEnum.value("INVALID", ME::INVALID)
        .value("CULTIST", ME::CULTIST)
        .value("JAW_WORM", ME::JAW_WORM)
        .value("TWO_LOUSE", ME::TWO_LOUSE)
        .value("SMALL_SLIMES", ME::SMALL_SLIMES)
        .value("BLUE_SLAVER", ME::BLUE_SLAVER)
        .value("GREMLIN_GANG", ME::GREMLIN_GANG)
        .value("LOOTER", ME::LOOTER)
        .value("LARGE_SLIME", ME::LARGE_SLIME)
        .value("LOTS_OF_SLIMES", ME::LOTS_OF_SLIMES)
        .value("EXORDIUM_THUGS", ME::EXORDIUM_THUGS)
        .value("EXORDIUM_WILDLIFE", ME::EXORDIUM_WILDLIFE)
        .value("RED_SLAVER", ME::RED_SLAVER)
        .value("THREE_LOUSE", ME::THREE_LOUSE)
        .value("TWO_FUNGI_BEASTS", ME::TWO_FUNGI_BEASTS)
        .value("GREMLIN_NOB", ME::GREMLIN_NOB)
        .value("LAGAVULIN", ME::LAGAVULIN)
        .value("THREE_SENTRIES", ME::THREE_SENTRIES)
        .value("SLIME_BOSS", ME::SLIME_BOSS)
        .value("THE_GUARDIAN", ME::THE_GUARDIAN)
        .value("HEXAGHOST", ME::HEXAGHOST)
        .value("SPHERIC_GUARDIAN", ME::SPHERIC_GUARDIAN)
        .value("CHOSEN", ME::CHOSEN)
        .value("SHELL_PARASITE", ME::SHELL_PARASITE)
        .value("THREE_BYRDS", ME::THREE_BYRDS)
        .value("TWO_THIEVES", ME::TWO_THIEVES)
        .value("CHOSEN_AND_BYRDS", ME::CHOSEN_AND_BYRDS)
        .value("SENTRY_AND_SPHERE", ME::SENTRY_AND_SPHERE)
        .value("SNAKE_PLANT", ME::SNAKE_PLANT)
        .value("SNECKO", ME::SNECKO)
        .value("CENTURION_AND_HEALER", ME::CENTURION_AND_HEALER)
        .value("CULTIST_AND_CHOSEN", ME::CULTIST_AND_CHOSEN)
        .value("THREE_CULTIST", ME::THREE_CULTIST)
        .value("SHELLED_PARASITE_AND_FUNGI", ME::SHELLED_PARASITE_AND_FUNGI)
        .value("GREMLIN_LEADER", ME::GREMLIN_LEADER)
        .value("SLAVERS", ME::SLAVERS)
        .value("BOOK_OF_STABBING", ME::BOOK_OF_STABBING)
        .value("AUTOMATON", ME::AUTOMATON)
        .value("COLLECTOR", ME::COLLECTOR)
        .value("CHAMP", ME::CHAMP)
        .value("THREE_DARKLINGS", ME::THREE_DARKLINGS)
        .value("ORB_WALKER", ME::ORB_WALKER)
        .value("THREE_SHAPES", ME::THREE_SHAPES)
        .value("SPIRE_GROWTH", ME::SPIRE_GROWTH)
        .value("TRANSIENT", ME::TRANSIENT)
        .value("FOUR_SHAPES", ME::FOUR_SHAPES)
        .value("MAW", ME::MAW)
        .value("SPHERE_AND_TWO_SHAPES", ME::SPHERE_AND_TWO_SHAPES)
        .value("JAW_WORM_HORDE", ME::JAW_WORM_HORDE)
        .value("WRITHING_MASS", ME::WRITHING_MASS)
        .value("GIANT_HEAD", ME::GIANT_HEAD)
        .value("NEMESIS", ME::NEMESIS)
        .value("REPTOMANCER", ME::REPTOMANCER)
        .value("AWAKENED_ONE", ME::AWAKENED_ONE)
        .value("TIME_EATER", ME::TIME_EATER)
        .value("DONU_AND_DECA", ME::DONU_AND_DECA)
        .value("SHIELD_AND_SPEAR", ME::SHIELD_AND_SPEAR)
        .value("THE_HEART", ME::THE_HEART)
        .value("LAGAVULIN_EVENT", ME::LAGAVULIN_EVENT)
        .value("COLOSSEUM_EVENT_SLAVERS", ME::COLOSSEUM_EVENT_SLAVERS)
        .value("COLOSSEUM_EVENT_NOBS", ME::COLOSSEUM_EVENT_NOBS)
        .value("MASKED_BANDITS_EVENT", ME::MASKED_BANDITS_EVENT)
        .value("MUSHROOMS_EVENT", ME::MUSHROOMS_EVENT)
        .value("MYSTERIOUS_SPHERE_EVENT", ME::MYSTERIOUS_SPHERE_EVENT);

    pybind11::enum_<RelicId> relicEnum(m, "RelicId", pybind11::metaclass(enum_metaclass));
    relicEnum.value("AKABEKO", RelicId::AKABEKO)
        .value("ART_OF_WAR", RelicId::ART_OF_WAR)
        .value("BIRD_FACED_URN", RelicId::BIRD_FACED_URN)
        .value("BLOODY_IDOL", RelicId::BLOODY_IDOL)
        .value("BLUE_CANDLE", RelicId::BLUE_CANDLE)
        .value("BRIMSTONE", RelicId::BRIMSTONE)
        .value("CALIPERS", RelicId::CALIPERS)
        .value("CAPTAINS_WHEEL", RelicId::CAPTAINS_WHEEL)
        .value("CENTENNIAL_PUZZLE", RelicId::CENTENNIAL_PUZZLE)
        .value("CERAMIC_FISH", RelicId::CERAMIC_FISH)
        .value("CHAMPION_BELT", RelicId::CHAMPION_BELT)
        .value("CHARONS_ASHES", RelicId::CHARONS_ASHES)
        .value("CHEMICAL_X", RelicId::CHEMICAL_X)
        .value("CLOAK_CLASP", RelicId::CLOAK_CLASP)
        .value("DARKSTONE_PERIAPT", RelicId::DARKSTONE_PERIAPT)
        .value("DEAD_BRANCH", RelicId::DEAD_BRANCH)
        .value("DUALITY", RelicId::DUALITY)
        .value("ECTOPLASM", RelicId::ECTOPLASM)
        .value("EMOTION_CHIP", RelicId::EMOTION_CHIP)
        .value("FROZEN_CORE", RelicId::FROZEN_CORE)
        .value("FROZEN_EYE", RelicId::FROZEN_EYE)
        .value("GAMBLING_CHIP", RelicId::GAMBLING_CHIP)
        .value("GINGER", RelicId::GINGER)
        .value("GOLDEN_EYE", RelicId::GOLDEN_EYE)
        .value("GREMLIN_HORN", RelicId::GREMLIN_HORN)
        .value("HAND_DRILL", RelicId::HAND_DRILL)
        .value("HAPPY_FLOWER", RelicId::HAPPY_FLOWER)
        .value("HORN_CLEAT", RelicId::HORN_CLEAT)
        .value("HOVERING_KITE", RelicId::HOVERING_KITE)
        .value("ICE_CREAM", RelicId::ICE_CREAM)
        .value("INCENSE_BURNER", RelicId::INCENSE_BURNER)
        .value("INK_BOTTLE", RelicId::INK_BOTTLE)
        .value("INSERTER", RelicId::INSERTER)
        .value("KUNAI", RelicId::KUNAI)
        .value("LETTER_OPENER", RelicId::LETTER_OPENER)
        .value("LIZARD_TAIL", RelicId::LIZARD_TAIL)
        .value("MAGIC_FLOWER", RelicId::MAGIC_FLOWER)
        .value("MARK_OF_THE_BLOOM", RelicId::MARK_OF_THE_BLOOM)
        .value("MEDICAL_KIT", RelicId::MEDICAL_KIT)
        .value("MELANGE", RelicId::MELANGE)
        .value("MERCURY_HOURGLASS", RelicId::MERCURY_HOURGLASS)
        .value("MUMMIFIED_HAND", RelicId::MUMMIFIED_HAND)
        .value("NECRONOMICON", RelicId::NECRONOMICON)
        .value("NILRYS_CODEX", RelicId::NILRYS_CODEX)
        .value("NUNCHAKU", RelicId::NUNCHAKU)
        .value("ODD_MUSHROOM", RelicId::ODD_MUSHROOM)
        .value("OMAMORI", RelicId::OMAMORI)
        .value("ORANGE_PELLETS", RelicId::ORANGE_PELLETS)
        .value("ORICHALCUM", RelicId::ORICHALCUM)
        .value("ORNAMENTAL_FAN", RelicId::ORNAMENTAL_FAN)
        .value("PAPER_KRANE", RelicId::PAPER_KRANE)
        .value("PAPER_PHROG", RelicId::PAPER_PHROG)
        .value("PEN_NIB", RelicId::PEN_NIB)
        .value("PHILOSOPHERS_STONE", RelicId::PHILOSOPHERS_STONE)
        .value("POCKETWATCH", RelicId::POCKETWATCH)
        .value("RED_SKULL", RelicId::RED_SKULL)
        .value("RUNIC_CUBE", RelicId::RUNIC_CUBE)
        .value("RUNIC_DOME", RelicId::RUNIC_DOME)
        .value("RUNIC_PYRAMID", RelicId::RUNIC_PYRAMID)
        .value("SACRED_BARK", RelicId::SACRED_BARK)
        .value("SELF_FORMING_CLAY", RelicId::SELF_FORMING_CLAY)
        .value("SHURIKEN", RelicId::SHURIKEN)
        .value("SNECKO_EYE", RelicId::SNECKO_EYE)
        .value("SNECKO_SKULL", RelicId::SNECKO_SKULL)
        .value("SOZU", RelicId::SOZU)
        .value("STONE_CALENDAR", RelicId::STONE_CALENDAR)
        .value("STRANGE_SPOON", RelicId::STRANGE_SPOON)
        .value("STRIKE_DUMMY", RelicId::STRIKE_DUMMY)
        .value("SUNDIAL", RelicId::SUNDIAL)
        .value("THE_ABACUS", RelicId::THE_ABACUS)
        .value("THE_BOOT", RelicId::THE_BOOT)
        .value("THE_SPECIMEN", RelicId::THE_SPECIMEN)
        .value("TINGSHA", RelicId::TINGSHA)
        .value("TOOLBOX", RelicId::TOOLBOX)
        .value("TORII", RelicId::TORII)
        .value("TOUGH_BANDAGES", RelicId::TOUGH_BANDAGES)
        .value("TOY_ORNITHOPTER", RelicId::TOY_ORNITHOPTER)
        .value("TUNGSTEN_ROD", RelicId::TUNGSTEN_ROD)
        .value("TURNIP", RelicId::TURNIP)
        .value("TWISTED_FUNNEL", RelicId::TWISTED_FUNNEL)
        .value("UNCEASING_TOP", RelicId::UNCEASING_TOP)
        .value("VELVET_CHOKER", RelicId::VELVET_CHOKER)
        .value("VIOLET_LOTUS", RelicId::VIOLET_LOTUS)
        .value("WARPED_TONGS", RelicId::WARPED_TONGS)
        .value("WRIST_BLADE", RelicId::WRIST_BLADE)
        .value("BLACK_BLOOD", RelicId::BLACK_BLOOD)
        .value("BURNING_BLOOD", RelicId::BURNING_BLOOD)
        .value("MEAT_ON_THE_BONE", RelicId::MEAT_ON_THE_BONE)
        .value("FACE_OF_CLERIC", RelicId::FACE_OF_CLERIC)
        .value("ANCHOR", RelicId::ANCHOR)
        .value("ANCIENT_TEA_SET", RelicId::ANCIENT_TEA_SET)
        .value("BAG_OF_MARBLES", RelicId::BAG_OF_MARBLES)
        .value("BAG_OF_PREPARATION", RelicId::BAG_OF_PREPARATION)
        .value("BLOOD_VIAL", RelicId::BLOOD_VIAL)
        .value("BOTTLED_FLAME", RelicId::BOTTLED_FLAME)
        .value("BOTTLED_LIGHTNING", RelicId::BOTTLED_LIGHTNING)
        .value("BOTTLED_TORNADO", RelicId::BOTTLED_TORNADO)
        .value("BRONZE_SCALES", RelicId::BRONZE_SCALES)
        .value("BUSTED_CROWN", RelicId::BUSTED_CROWN)
        .value("CLOCKWORK_SOUVENIR", RelicId::CLOCKWORK_SOUVENIR)
        .value("COFFEE_DRIPPER", RelicId::COFFEE_DRIPPER)
        .value("CRACKED_CORE", RelicId::CRACKED_CORE)
        .value("CURSED_KEY", RelicId::CURSED_KEY)
        .value("DAMARU", RelicId::DAMARU)
        .value("DATA_DISK", RelicId::DATA_DISK)
        .value("DU_VU_DOLL", RelicId::DU_VU_DOLL)
        .value("ENCHIRIDION", RelicId::ENCHIRIDION)
        .value("FOSSILIZED_HELIX", RelicId::FOSSILIZED_HELIX)
        .value("FUSION_HAMMER", RelicId::FUSION_HAMMER)
        .value("GIRYA", RelicId::GIRYA)
        .value("GOLD_PLATED_CABLES", RelicId::GOLD_PLATED_CABLES)
        .value("GREMLIN_VISAGE", RelicId::GREMLIN_VISAGE)
        .value("HOLY_WATER", RelicId::HOLY_WATER)
        .value("LANTERN", RelicId::LANTERN)
        .value("MARK_OF_PAIN", RelicId::MARK_OF_PAIN)
        .value("MUTAGENIC_STRENGTH", RelicId::MUTAGENIC_STRENGTH)
        .value("NEOWS_LAMENT", RelicId::NEOWS_LAMENT)
        .value("NINJA_SCROLL", RelicId::NINJA_SCROLL)
        .value("NUCLEAR_BATTERY", RelicId::NUCLEAR_BATTERY)
        .value("ODDLY_SMOOTH_STONE", RelicId::ODDLY_SMOOTH_STONE)
        .value("PANTOGRAPH", RelicId::PANTOGRAPH)
        .value("PRESERVED_INSECT", RelicId::PRESERVED_INSECT)
        .value("PURE_WATER", RelicId::PURE_WATER)
        .value("RED_MASK", RelicId::RED_MASK)
        .value("RING_OF_THE_SERPENT", RelicId::RING_OF_THE_SERPENT)
        .value("RING_OF_THE_SNAKE", RelicId::RING_OF_THE_SNAKE)
        .value("RUNIC_CAPACITOR", RelicId::RUNIC_CAPACITOR)
        .value("SLAVERS_COLLAR", RelicId::SLAVERS_COLLAR)
        .value("SLING_OF_COURAGE", RelicId::SLING_OF_COURAGE)
        .value("SYMBIOTIC_VIRUS", RelicId::SYMBIOTIC_VIRUS)
        .value("TEARDROP_LOCKET", RelicId::TEARDROP_LOCKET)
        .value("THREAD_AND_NEEDLE", RelicId::THREAD_AND_NEEDLE)
        .value("VAJRA", RelicId::VAJRA)
        .value("ASTROLABE", RelicId::ASTROLABE)
        .value("BLACK_STAR", RelicId::BLACK_STAR)
        .value("CALLING_BELL", RelicId::CALLING_BELL)
        .value("CAULDRON", RelicId::CAULDRON)
        .value("CULTIST_HEADPIECE", RelicId::CULTIST_HEADPIECE)
        .value("DOLLYS_MIRROR", RelicId::DOLLYS_MIRROR)
        .value("DREAM_CATCHER", RelicId::DREAM_CATCHER)
        .value("EMPTY_CAGE", RelicId::EMPTY_CAGE)
        .value("ETERNAL_FEATHER", RelicId::ETERNAL_FEATHER)
        .value("FROZEN_EGG", RelicId::FROZEN_EGG)
        .value("GOLDEN_IDOL", RelicId::GOLDEN_IDOL)
        .value("JUZU_BRACELET", RelicId::JUZU_BRACELET)
        .value("LEES_WAFFLE", RelicId::LEES_WAFFLE)
        .value("MANGO", RelicId::MANGO)
        .value("MATRYOSHKA", RelicId::MATRYOSHKA)
        .value("MAW_BANK", RelicId::MAW_BANK)
        .value("MEAL_TICKET", RelicId::MEAL_TICKET)
        .value("MEMBERSHIP_CARD", RelicId::MEMBERSHIP_CARD)
        .value("MOLTEN_EGG", RelicId::MOLTEN_EGG)
        .value("NLOTHS_GIFT", RelicId::NLOTHS_GIFT)
        .value("NLOTHS_HUNGRY_FACE", RelicId::NLOTHS_HUNGRY_FACE)
        .value("OLD_COIN", RelicId::OLD_COIN)
        .value("ORRERY", RelicId::ORRERY)
        .value("PANDORAS_BOX", RelicId::PANDORAS_BOX)
        .value("PEACE_PIPE", RelicId::PEACE_PIPE)
        .value("PEAR", RelicId::PEAR)
        .value("POTION_BELT", RelicId::POTION_BELT)
        .value("PRAYER_WHEEL", RelicId::PRAYER_WHEEL)
        .value("PRISMATIC_SHARD", RelicId::PRISMATIC_SHARD)
        .value("QUESTION_CARD", RelicId::QUESTION_CARD)
        .value("REGAL_PILLOW", RelicId::REGAL_PILLOW)
        .value("SSSERPENT_HEAD", RelicId::SSSERPENT_HEAD)
        .value("SHOVEL", RelicId::SHOVEL)
        .value("SINGING_BOWL", RelicId::SINGING_BOWL)
        .value("SMILING_MASK", RelicId::SMILING_MASK)
        .value("SPIRIT_POOP", RelicId::SPIRIT_POOP)
        .value("STRAWBERRY", RelicId::STRAWBERRY)
        .value("THE_COURIER", RelicId::THE_COURIER)
        .value("TINY_CHEST", RelicId::TINY_CHEST)
        .value("TINY_HOUSE", RelicId::TINY_HOUSE)
        .value("TOXIC_EGG", RelicId::TOXIC_EGG)
        .value("WAR_PAINT", RelicId::WAR_PAINT)
        .value("WHETSTONE", RelicId::WHETSTONE)
        .value("WHITE_BEAST_STATUE", RelicId::WHITE_BEAST_STATUE)
        .value("WING_BOOTS", RelicId::WING_BOOTS)
        .value("CIRCLET", RelicId::CIRCLET)
        .value("RED_CIRCLET", RelicId::RED_CIRCLET)
        .value("INVALID", RelicId::INVALID);

    pybind11::enum_<Potion> potionEnum(m, "Potion", pybind11::metaclass(enum_metaclass));
    potionEnum.value("INVALID", Potion::INVALID)
        .value("EMPTY_POTION_SLOT", Potion::EMPTY_POTION_SLOT)
        .value("AMBROSIA", Potion::AMBROSIA)
        .value("ANCIENT_POTION", Potion::ANCIENT_POTION)
        .value("ATTACK_POTION", Potion::ATTACK_POTION)
        .value("BLESSING_OF_THE_FORGE", Potion::BLESSING_OF_THE_FORGE)
        .value("BLOCK_POTION", Potion::BLOCK_POTION)
        .value("BLOOD_POTION", Potion::BLOOD_POTION)
        .value("BOTTLED_MIRACLE", Potion::BOTTLED_MIRACLE)
        .value("COLORLESS_POTION", Potion::COLORLESS_POTION)
        .value("CULTIST_POTION", Potion::CULTIST_POTION)
        .value("CUNNING_POTION", Potion::CUNNING_POTION)
        .value("DEXTERITY_POTION", Potion::DEXTERITY_POTION)
        .value("DISTILLED_CHAOS", Potion::DISTILLED_CHAOS)
        .value("DUPLICATION_POTION", Potion::DUPLICATION_POTION)
        .value("ELIXIR_POTION", Potion::ELIXIR_POTION)
        .value("ENERGY_POTION", Potion::ENERGY_POTION)
        .value("ENTROPIC_BREW", Potion::ENTROPIC_BREW)
        .value("ESSENCE_OF_DARKNESS", Potion::ESSENCE_OF_DARKNESS)
        .value("ESSENCE_OF_STEEL", Potion::ESSENCE_OF_STEEL)
        .value("EXPLOSIVE_POTION", Potion::EXPLOSIVE_POTION)
        .value("FAIRY_POTION", Potion::FAIRY_POTION)
        .value("FEAR_POTION", Potion::FEAR_POTION)
        .value("FIRE_POTION", Potion::FIRE_POTION)
        .value("FLEX_POTION", Potion::FLEX_POTION)
        .value("FOCUS_POTION", Potion::FOCUS_POTION)
        .value("FRUIT_JUICE", Potion::FRUIT_JUICE)
        .value("GAMBLERS_BREW", Potion::GAMBLERS_BREW)
        .value("GHOST_IN_A_JAR", Potion::GHOST_IN_A_JAR)
        .value("HEART_OF_IRON", Potion::HEART_OF_IRON)
        .value("LIQUID_BRONZE", Potion::LIQUID_BRONZE)
        .value("LIQUID_MEMORIES", Potion::LIQUID_MEMORIES)
        .value("POISON_POTION", Potion::POISON_POTION)
        .value("POTION_OF_CAPACITY", Potion::POTION_OF_CAPACITY)
        .value("POWER_POTION", Potion::POWER_POTION)
        .value("REGEN_POTION", Potion::REGEN_POTION)
        .value("SKILL_POTION", Potion::SKILL_POTION)
        .value("SMOKE_BOMB", Potion::SMOKE_BOMB)
        .value("SNECKO_OIL", Potion::SNECKO_OIL)
        .value("SPEED_POTION", Potion::SPEED_POTION)
        .value("STANCE_POTION", Potion::STANCE_POTION)
        .value("STRENGTH_POTION", Potion::STRENGTH_POTION)
        .value("SWIFT_POTION", Potion::SWIFT_POTION)
        .value("WEAK_POTION", Potion::WEAK_POTION);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}

// os.add_dll_directory("C:\\Program Files\\mingw-w64\\x86_64-8.1.0-posix-seh-rt_v6-rev0\\mingw64\\bin")


