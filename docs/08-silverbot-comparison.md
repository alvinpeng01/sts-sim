# Comparison with Silver Automaton

`silverbot-reference/` is Daniel Ziegler's separate STS AI, vendored read-only.
It is the only external system this project has measured itself against, and its
own `EXPERIMENT_LOG.md` is unusually well instrumented. Numbers below were read
from that log and from its headers, not from memory.

## Where each system stands

| | Silver Automaton | This project |
|---|---|---|
| A0 win rate | **0.794 ± 0.013** (honest1) | 13/100 with v31 (pre-fix) |
| A20 | **18.6% ± 2.4% heart kill**, n=1000 @ 10k sims; avg floor 39.9 | 0 victories; mean floor 23.57 @ 300 sims |
| Combat | heuristic MCTS, no network | heuristic MCTS, no network |
| Overworld | online PPO + GAE, pipelined collection | offline supervised on counterfactual-rollout targets |

Both systems reached the same structural conclusion independently: combat is
played by search, the network handles the overworld.

## Combat search

| Aspect | Silverbot | Ours |
|---|---|---|
| Algorithm | UCT/MCTS with graph dedup + DPW | UCB1 + additive PUCT term, DPW on chance nodes, transposition table; **sequential halving at the root** under the shipped config |
| Rollout policy | hand-tuned 133-card priority list, `randomize` mode | type-based scoring (ATTACK/SKILL/POWER) + per-card priors, **sampled rollouts since 2026-07-31** (`rolloutTemperature = 2.199`) |
| Leaf evaluation | instant Optuna-tuned 13-weight formula | full heuristic playout to a terminal state |
| Win eval | `winBonus 53 + postBattleHealedHp + potions·11 − turn·0.4` | `200 + curHp − 0.5·turn` plus tuned HP/potion/turn adjustments |
| Loss eval | `(1 − monsterHpRatio)·37 + alive·(−3.4) + energyWasted·(−1.75) + …` | `min(−1, −400 + turn) + 566.8·(1 − monsterHpRatio) + potions/2 − 7.57·alive` |
| Runic Dome | materializes hidden intents in rollouts | no Dome model at all |
| Card selects | context-aware (e.g. Exhume picks the best card in the exhaust pile) | the same `nativeScoreAction` for everything |
| DPW | C = 3.7, α = 0.52 | C = 3.66, α = 0.67 |
| Boss budget | 3× sims on boss encounters | none |
| Tree reuse | yes | no — measured worth ~1.34x effective budget, below noise |

The two loss branches are shaped the same way and for the same reason: this
project's `lossProgressCreditWeight` was added after reading Silverbot's
`monsterDamageWeight`, during the Time Eater investigation where raising sims
from 200 to 1600 moved nothing.

### Head to head on identical fights (2026-07-31)

Until now this document compared the two engines by reading their code. They can
now be compared by **running both on the same reconstructed fights** — a top
human's actual decks, relics, potions and HP from 100 A20 Heart runs. Harness
`lightspeed/_silverbot_human_deck.py`; the benchmark itself is described in
[03-combat-search.md](03-combat-search.md).

It needs no translation layer: their `CardId`, `RelicId`, `Potion` and
`MonsterEncounter` enums are **integer-identical to ours** across all
371/181/44/64 members, both being forks of the same upstream. It does need its own
process, since both engines' Python modules are named `slaythespire`.

Scoring `mean(human_damage − hp_paid)`, where 0 is human parity and a death pays
all remaining HP, on the 528 test fights silverbot can play, at 100 sims:

| | objective | ratio on wins | deaths |
|---|---|---|---|
| ours, pre-tuning config | −11.83 | 1.45 | 88/528 |
| ours, after the first config apply | −6.84 | 1.14 | 75/528 |
| **ours, after the 42-param run** | **−5.78** | **1.11** | **78/528** |
| silverbot | −6.58 | 1.01 | 84/528 |

**They spend less HP on the fights they win; we lose fewer fights.** Decomposed
across all 528: the total gap is made of **-542 HP from differing deaths** (they
lose 27 fights we win, we lose 18 they win) and **+679 HP from damage on shared
wins**. Those nearly cancel. Combat tuning on 2026-07-31 moved the net from 5.2
HP against us to 0.80 in our favour.

The decomposition matters for how much weight to put on the remaining gap: the
objective prices a death at exactly the HP you were holding, which is the most
generous possible treatment of dying. A death in a real run forfeits every
remaining floor. The crossover is low -- at only 25 HP of extra death penalty we
were already ahead even before the final tuning round.

Two caveats. "100 sims" is not guaranteed to be equal work in both engines —
theirs is `simulation_count_base` with `boss_simulation_multiplier = 2` — though
our side is insulated because our play is flat from 43 to 1500 sims. And
silverbot **cannot play 101 of the 2841 fights**: 11 of the 100 runs carry
Prismatic Shard, which legitimately offers Silent/Defect/Watcher cards to an
Ironclad, and silverbot aborts the process (assert, not exception) on an
unimplemented off-class card. Our engine implements all four characters and plays
them. The benchmark carries a precomputed `off_class_cards` field to filter them,
and our numbers above are scored on the identical filtered subset.

This is the first measurement that puts a *reachable* target on our combat.
Silverbot is not a ceiling, but unlike the human it is a bot on the same engine
family, so the remaining 0.26 HP — and their 1.15 vs our 1.42 on elites — is
concrete headroom rather than an aspiration.

## Training

| Aspect | Silverbot | Ours |
|---|---|---|
| Method | online PPO + GAE | offline supervised on soft rollout targets |
| Model | dim 256, 4 layers, 8 heads, pre-norm | dim 96, 2 layers, 4 heads |
| Critic | separate value network | shared value head on the transformer |
| Training sims | 1,000 (3,000 on bosses) | 300 |
| Eval sims | 10,000 (30,000 on bosses) | 300 (900 tested) |
| Hardware | 30-core boxes, A100s (~9% utilized — CPU-MCTS-bound) | one 6-core desktop, CPU only |

## The data-volume gap

This is the largest measured difference between the two systems, and it is not
about search at all.

| | Silverbot | Ours through v30 | Ours v31 |
|---|---|---|---|
| Offline SL rows | 338k train / 201k val | **1,308 / 238** | 4,008 / 778 |

Silverbot's offline SL dataset is **258×** our pre-v31 training set, collected for
about $5 in 3.2 h on one spot instance. And that dataset is not even its main
agent — its main agent is online PPO, where every decision in every episode is a
sample used once. "Overfitting the dataset" is not structurally available to it.

Two things this corrects:

- **Silverbot does overfit offline, and knows it.** Its log records a
  value-function SL ceiling that is "irreducible variance, not capacity (train
  EV→1.0 overfit check); **smaller nets win offline**." Our v28 (1.6M) beating
  v30 (6.6M) replicates their finding rather than contradicting it.
- **Silverbot is past the data problem.** Its own note puts the remaining
  bottleneck at "value/advantage-signal quality (irreducible return variance,
  EV ~0.38), **not data volume**." We were still in the data problem and had not
  noticed.

Raising `--max-labels-per-episode` from 2 to 12 — same sim budget, same rollout
policy — was worth +1.95 paired floors. See [06-experiment-log.md](06-experiment-log.md).

## What not to copy

- The 133-card priority list is hand-tuned and brittle; our tunable heuristic can
  be fitted by CMA-ES instead.
- Optuna-tuned terminal weights at 5,000 sims may not transfer to a 300-sim
  budget.
- Silverbot's tree reuse. Measured on our search it could recycle only ~34% of the tree (1.34x effective budget), which is far below what a 3x budget buys us. See [07-known-issues.md](07-known-issues.md).
- The full PPO pipeline assumes infrastructure this project does not have.

## What is worth taking, ordered by evidence

1. **More label volume.** The only lever that has paid. Tested at 10×; 30× was
   started and abandoned.
2. **Context-aware card selects in rollouts.** Exhume, Secret Weapon and similar
   currently score through the generic path.
3. **Never use potions in rollouts.** Silverbot's default. Ours may burn them in
   simulated lines that never happen.
4. **Boss sim multiplier.** Cheap to try; note that the 300→900 budget test says
   general budget increases buy little, so this should be evaluated as a targeted
   change rather than assumed.

Explicitly demoted by measurement:

- ~~Silverbot's `dest_room` auxiliary loss.~~ Their head predicts each path
  option's destination room type, and their log credits it with grounding routing
  ("collapsed to ~0.003 by iteration 6", routing probe 0.918 → 0.968). It does not
  transfer: `whole_run_transformer_v27.py:143` already embeds
  `action_target_rooms` straight into each action's representation, so such a head
  would predict its own input. Our routing failure is therefore **not** a
  grounding problem — the net knows an option leads to an elite and has learned
  that elites are bad (ELITE −2.55). That is a valuation problem inherited from
  labels whose rollouts play elites badly.
- ~~Runic Dome intent materialization.~~ Our search is *clairvoyant* under Dome
  rather than seeing phantom moves, so materializing intents removes information
  and makes the agent slightly worse. Dome is held in 3/100 runs — far inside the
  ±0.55 noise of a 200-seed paired eval. See [07-known-issues.md](07-known-issues.md).

- ~~Raise the sim budget from 300 to 800+.~~ 900 sims buys +0.21 to +0.96 floors
  on matched checkpoints and seeds — less than half what a checkpoint change buys
  at a fixed budget.
- ~~Soften the loss evaluation.~~ Already done, and tuned; see
  [03-combat-search.md](03-combat-search.md). Do not "fix" the −400 base again.

## Another external yardstick

Silverbot is a bot's learned policy. For what *human* winning runs do, see the
1,008,636-run developer-dataset analysis recorded in
[07-known-issues.md](07-known-issues.md) — it independently lands on elites and
campfires, the same two rooms Silverbot's routing coefficients flag and the same
two this policy handles worst.

A survey of the wider field (2026-07-31) found no other project publishing an
Ironclad Ascension 20 win rate. `benmuth/AutoClad` targets the identical goal on
this same engine and reports none; `xaved88/bottled_ai` reports 20% Ironclad but
at an unstated (probably low) ascension, and never reaches Act 4. Silver
Automaton remains the only meaningful benchmark.

## Shared problems

Both systems document **deck-blindness**: card picks driven largely by card
identity rather than by what the deck already holds. Silverbot's log describes it
as an equilibrium problem. Neither has solved it.
