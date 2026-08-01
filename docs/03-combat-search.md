# Combat search

All line references are to `sts_lightspeed/bindings/slaythespire.cpp` unless
stated otherwise. Parameter values were read live from the compiled
`build/slaythespire.cp313-win_amd64.pyd` after applying
`lightspeed/tuned_search_params.json`.

## Where combat happens

`native_playout_current_battle_result(gc, sims[, seed])` plays an **entire
fight** and returns a stats dict. Inside, `nativePlayoutBattle` (`:2434`) loops
until the battle resolves, calling `nativeRunMctsSearch` (`:2346`) once per
player decision.

Each call allocates a fresh `MctsArena` and a fresh transposition table, so
**there is no tree reuse between decisions**. Two prior attempts at rerooting
crashed — both in the *Python* `az_search.py`, not here — and a 2026-07-30
measurement puts the ceiling on what reuse could buy at a 1.34× effective budget,
below what this project can resolve. See
[07-known-issues.md](07-known-issues.md).

Search horizon: `NATIVE_MAX_TURNS_PER_SEARCH = 20` (`:1472`) turns past the
current one.

### Sequential halving is on in production

`nativeRunMctsSearch` delegates to `nativeRunMctsSearchSeqHalving` (`:2245`)
whenever `g_useSeqHalving` is set — and the shipped config sets it
(`tuned_search_params.json` → `fitness_config.seq_halving = true`, applied by
`search_config.apply_search_config`). Verified live: `sts.get_seq_halving()`
returns `True` after `ensure_search_config()`.

This changes the root action choice. The plain UCB1 driver picks the **most
visited** root edge; the sequential-halving driver picks by **mean value** among
survivors, because its visit counts are an artifact of its own phase schedule
and carry no preference information.

### Runtime selector state

| Selector | Value under the shipped config | Notes |
|---|---|---|
| `seq_halving` | **on** | root budget allocation by halving phases |
| `leaf_eval_mode` | `rollout` | verified by `search_config.py`, which raises otherwise |
| `state_merging` | off | `search_config.py` treats it as unsafe and flags it |
| `RAVE` | off | the tuned config was calibrated without it |

## Node selection

`nativeSelectIdx` (`:1968`):

```
score_i = exploit_i  +  c · sqrt( ln(N+1) / (n_i + 1) )  +  c_puct · prior_i · sqrt(N+1) / (1 + n_i)
```

- Unvisited edges are taken first, in **heuristic-score order** (`visitOrder`),
  not enumeration order.
- `c` is `cUcbChance` when the destination is a chance node (an `END_TURN`, or
  any action already known to have sampled children), otherwise `cUcb`.
- The PUCT term is additive on top of UCB1, not a replacement. Priors come from
  the same heuristic scores. `c_puct` is tuned to **6.01**, so it is live.
- With RAVE on, `exploit` would blend in the AMAF mean via the Gelly/Silver
  schedule `β = amafN / (n + amafN + 4·n·amafN·b²)`. RAVE is off.

## Chance nodes and DPW

`nativeDpwChanceChild` (`:2049`) caps the number of sampled outcomes at
`ceil(wcChance · (n+1)^waChance)` — tuned to `3.66 · (n+1)^0.667`. Below the cap
a new outcome is sampled; at the cap an existing child is chosen with
probability proportional to `visitCount + 1`.

Sampling uses common random numbers keyed by a hash of
`(crnBase, NativeStateKey, localSampleIndex)`, so sibling branches see the same
stochastic outcomes — this is what makes paired counterfactual comparison in
label generation meaningful.

`NativeStateKey` (`:1788`) covers player HP/block/energy/turn, 19 player
statuses, and per monster: HP, block, Strength, Vulnerable, Weak, halfDead,
**both `moveHistory` entries, `miscInfo`**, and 6 statuses. Cards are keyed as
**sorted** hand, **sorted** discard, and **order-preserving** draw pile — draw
order is deliberately kept, since it determines future draws.

Two things it does *not* cover: card **upgrade state** (piles store `c.id` only,
so Strike and Strike+ key identically) and the **exhaust pile**.

**The whole key is inert in production.** Every use of it — the transposition
table for deterministic children and the merging of chance children — is gated
behind `g_useStateMerging`, which is off and which `search_config.py` actively
flags as unsafe. So the key's gaps cost nothing today, but they also mean it has
never been exercised: anything that starts depending on it (state merging, or
tree reuse) must close those gaps first.

## Rollout policy

`nativeHeuristicPickFast` (`:1159`) picks by `nativeScoreAction` (`:1003`).
**Which terms are actually live depends on the tuned config, and most of the
recently added ones are not.**

Base scoring:

| Branch | Formula | Live? |
|---|---|---|
| non-CARD (potions, END_TURN) | flat `5.0` | yes, not tunable |
| ATTACK | `attackBase + (1 − targetHp/targetMaxHp)·attackFinishOffScale − min(block,20)·attackBlockPenaltyScale [+ aoeBonus if ≥2 living and AoE]` | yes |
| SKILL | `skillBase + dangerFraction·skillDangerScale [− skillHastePenalty if Haste-wasted and danger < threshold] [− defensiveCardSuppressionPenalty if block sufficient and defensive]` | yes |
| POWER | `powerScore + powerPerTurnValueWeight·perTurnValue(card,deck)·monsterHpRatio + powerImmediateValueWeight·immediateValue(card)` | yes, **since 2026-07-31** |
| STATUS / CURSE | `1.0` | yes, not tunable |
| every CARD branch | `+ perCardWeightScale·cardPickRateWeight[card] + silverPriorWeight·silverPrior` | yes |

`dangerFraction = ctx.unblocked / max(1, player.curHp)` — continuous, not a
binary in-danger gate. `silverPrior` is derived from Silver Automaton's
133-card play ranking: `(134 − rank)/133`. Boss encounters use
`bossSilverCardPlayPriorWeight` (tuned to 1.0) instead of the general 5.0.

**Turned on 2026-07-31** by `tune_search_human.py`, having been 0.0 since they
were written: `vulnerableApplyBonus`, `weakApplyBonus`, `powerPerTurnValueWeight`,
`powerImmediateValueWeight`, `energyWasteWeight`, `enemyBlockWeight`,
`directBlockScoreWeight` and `rolloutTemperature` (now **2.199**).

**Still off at 0.0**: `silentPoisonApplyBonus` (Silent-only), `policyNetWeight`
(needs a net loaded), and all ten `vf*` weights (live only in `leaf_eval_mode`
value/truncated).

Corrected 2026-07-31: `attackDamageScoreWeight`, `selfDamageScorePenalty`,
`blockWeight` and `winHpFractionWeight` were listed here as off, and are not.
`tune_search_human.py`'s 42-parameter run put all four in the shipped artifact
(0.0258 / 3.572 / 4.139 / 9.157, `tuned_search_params.json` at 16:31), and
`rolloutTemperature` is 2.489 there rather than the 2.199 quoted above. The
staleness was caught by `tests/test_whole_run_mcts_config.py`, which asserted
`block_weight == 0.0` and started failing the moment the artifact shipped; that
assertion now reads the compiled default from the live parameter set instead of
a literal, so it cannot rot the same way again. **Read the artifact, not this
list.**

Two consequences of `rolloutTemperature` leaving zero, both worth knowing before
touching this code. It was previously true that "every rollout from a node
replays essentially the same line, differing only at chance nodes" — that is what
made extra simulations mostly re-measure one line, and it is the mechanism behind
the flat sim curve below. It is no longer true: the rollout now samples.

And sampling reached a code path nothing had ever exercised.
`nativeGumbelNoise` held `static thread_local std::mt19937_64
gumbelRng(std::random_device{}())` — the rollout's sampling RNG seeded outside
every reproducibility guarantee in the file, dormant for as long as the
temperature was 0 because argmax never calls it. Turning the parameter on made
identical `bc` + identical `search_seed` return different actions, silently
destroying common random numbers in every paired comparison. Now seeded from the
call's search seed (SplitMix64-mixed, since `play`-style callers derive highly
correlated seeds). Verified: eight searches at one seed are identical and a
528-fight evaluation reproduces exactly.

Powers were also scored a **flat 13.90 regardless of which Power** until the same
date, with 8 of Ironclad's 14 — Barricade, Corruption, Dark Embrace, Evolve, Feel
No Pain, Fire Breathing, Juggernaut, Rupture — returning 0 from both value tables
by deliberate choice, on the reasoning that guessing a per-turn number for a
conditional Power is worse than not trying. They are now scored with the
condition **measured from the deck** (exhaust / status / skill / block counts,
hoisted into `HeuristicContext` alongside `monsterHpRatio`) rather than guessed.
Measured effect: **+0.31 ± 0.49 HP (t = 0.64)** on 526 validation fights, with 314
of them playing identically — i.e. no detectable benefit. Kept because it is
neutral and the two weights are tunable, but do not expect anything from it.

### Anti-stall machinery

`nativePlayoutBattle` carries three escape hatches, all reported in the audit
counters that appear in every eval row:

- **Hard stall** — 20 turns with no reduction in monster durability (HP + block)
  switches to `safeProgressAction`: a legal, non-`END_TURN` action that strictly
  reduces monster durability and leaves no unblocked incoming damage.
  `stall_fallback_decisions`, capped at 3 consecutive.
- **Soft tempo** — 12 turns on a specific set of easy multi-monster encounters
  (Two Louse, Small Slimes, Gremlin Gang, Large Slime, Lots of Slimes, Exordium
  Wildlife, Three Louse). `soft_tempo_override_decisions`.
- **Turn limit** — recorded as `turn_limit_battles`.

They fire, but rarely. Across the 600 runs in
`runs/postfix_v28_v31_v33_a20_200seeds.jsonl` — 122,872 combat decisions, of
which 103,667 were searched and 19,194 forced — there were 11 stall fallbacks,
65 soft-tempo overrides, 8 stall progress overrides, and 0 turn-limit battles.
The overworld safety filter (which drops provably immediately-losing overworld
actions, `whole_run_env.partition_legal_actions`) fired 12 times.

## Terminal evaluation

Two different functions, and conflating them has caused a documented wrong
conclusion:

**`nativeTerminalReward` (`:386`)** — shared with `env.py` / PPO:

```
win  :  200 + curHp − 0.5·turn
loss :  min(−1, −400 + 1.0·turn)
```

**`nativeExpectimaxTerminalReward` (`:492`)** — what the **search** evaluates:

```
win  : base + (winHpWeight + act1EasyPoolSafety)·effectiveHp − curHp
              + potionScore + winBonusAdjust
              + winHpFractionWeight·100·hpFraction
              − turnPenaltyAdjust − energyPenalty
loss : base + lossProgressCreditWeight·(1 − monsterHpRatio)
              + potionScore/2 − aliveMonsterPenaltyWeight·aliveCount − energyPenalty
```

with `lossProgressCreditWeight` tuned to **566.81** (3.78× its 150.0 default).
The loss branch therefore has a large gradient, not a flat one: a turn-3 wipe
with monsters untouched scores near −397, while dying on turn 15 with the boss
at 5% HP scores near +153. That ~550-point spread exceeds the win branch's own
HP spread (`winHpWeight` 5.41 × ~70 HP ≈ 380).

The term exists for exactly this reason. Its own comment (`:393-407`) cites the
Time Eater investigation where raising sims from 200 to 1600 moved the win rate
not at all, because the flat loss constant in `nativeTerminalReward` gave UCB1
nothing to climb. **Do not "fix" the −400 base again** — a constant offset
cannot make losses indistinguishable to UCB1, only spread matters, and the
spread is already there. The subtractive base does still stand on the
`env.py`/PPO path, which is a separate question.

Fixed constants (`:94-100`): `NATIVE_W_HP 1.5`, `NATIVE_BETA 3.0`,
`NATIVE_W_WIN 200`, `NATIVE_W_DEATH 400`, `NATIVE_W_SHAPE 0.1`,
turn penalty on win 0.5, turn-survived bonus on loss 1.0.

One residual wrinkle, recorded not fixed: as `monsterHpRatio → 0` a loss
approaches +197 while a marginal win (1 HP, turn 30) sits near +190. It needs a
ratio under ~1% to bite, which is close to unreachable, since killing every
monster *is* a victory.

### Incoming-damage prediction

`nativePredictedIncomingDamage` (`:559`) does **not** use the raw
`Monster::getMoveBaseDamage` table lookup. It applies the engine's real damage
resolution: monster Strength added, monster Weak ×0.75, player Vulnerable ×1.5,
floored and clamped. The raw lookup was silently wrong by up to ±16 on Time
Eater. (Earlier documentation claiming the search uses the raw lookup was wrong.)

There is **no Runic Dome handling** anywhere in the search — `RUNIC_DOME`
appears only as a relic flag and an enum binding. Under Dome the search sees
whatever the engine's queued move says, with no bluffing model.

## Tunable parameters

`TunableParams` (`:120-332`) holds **55 doubles**, all exposed to Python via
`sts.get_search_params()` / `set_search_params()` in snake_case.
`tuned_search_params.json` overrides **29** of them; the other 26 sit at their
compiled defaults.

The 29 tuned values, with their defaults:

| Parameter | Default | Tuned |
|---|---:|---:|
| `c_ucb` | 1.5 | 8.3496 |
| `c_ucb_chance` | 1.5 | 0.5966 |
| `wc_chance` | 1.0 | 3.6607 |
| `wa_chance` | 0.5 | 0.6667 |
| `loss_progress_credit_weight` | 150.0 | 566.8092 |
| `brewing_threat_estimate` | 8.0 | 21.6791 |
| `attack_base` | 10.0 | 3.1892 |
| `attack_finish_off_scale` | 5.0 | 20.4019 |
| `attack_block_penalty_scale` | 0.15 | 0.2819 |
| `aoe_bonus` | 6.0 | 9.5120 |
| `skill_base` | 4.0 | 2.3743 |
| `skill_danger_scale` | 30.0 | 6.5402 |
| `skill_haste_penalty` | 5.0 | 7.9293 |
| `skill_haste_danger_threshold` | 0.1 | 0.2575 |
| `power_score` | 6.0 | 13.8993 |
| `end_turn_time_warp_risk_score` | 11.0 | 0.4457 |
| `per_card_weight_scale` | 0.0 | 21.8266 |
| `silver_card_play_prior_weight` | 0.0 | 5.0000 |
| `boss_silver_card_play_prior_weight` | −1.0 | 1.0000 |
| `c_puct` | 0.0 | 6.0105 |
| `puct_temperature` | 10.0 | 3.2108 |
| `win_hp_weight` | 1.0 | 5.4115 |
| `early_act_easy_pool_hp_safety_weight` | 0.0 | 1.0000 |
| `block_sufficiency_margin` | 4.0 | 5.1387 |
| `defensive_card_suppression_penalty` | 8.0 | 6.1622 |
| `potion_score_weight` | 0.0 | 29.7085 |
| `boss_heal_credit_weight` | 0.0 | 0.1533 |
| `win_turn_penalty_weight` | 1.0 | 1.0942 |
| `alive_monster_penalty_weight` | 0.0 | 7.5725 |

Note how far the tuner moved the type bases: `attack_base` 10 → 3.19 while
`attack_finish_off_scale` went 5 → 20.4. The rollout policy it converged on is
much more about finishing a wounded target than about attacking in general.

`end_turn_time_warp_risk_score` collapsing from 11.0 to 0.45 effectively removes
the Time Warp END_TURN deterrent.

## Applying and verifying a configuration

`search_config.py` is the only supported way to configure the search:

1. `load_search_config` — refuses a JSON without a `params` block.
2. `apply_search_config` — calls `sts.reset_search_config()` **first**, because
   the artifact contains only overrides and checking just those keys cannot
   detect stale values left in unspecified parameters by an earlier experiment.
3. Applies `params`, then sets `seq_halving` from `fitness_config`.
4. `active_search_config_mismatches` re-reads the live values and raises on any
   difference, on state merging being on, on RAVE being on, or on
   `leaf_eval_mode != "rollout"`.

`WholeRunEnv` calls `ensure_search_config()` in its constructor, so whole-run
combat cannot silently inherit native defaults or another experiment's globals.

## CMA-ES tuning

`tune_search_cma.py` optimizes the native parameters as **multiplicative factors**
around their current defaults (`x_i = raw_i / default_i`, search starts at
all-ones), because the raw constants span wildly different scales and one
isotropic step size cannot serve all of them.

Fitness, from the shipped artifact's `fitness_config`: 20 episodes per encounter
across 15 named encounters at Ascension 20 and 150 sims, scoring
`hp_fitness_weight = 2.0` on the HP fraction, with per-encounter synthetic deck
resources for act1/act2/act3 × basic/elite/boss. The recorded best score is
2.1181.

Every candidate is evaluated in its own **process**, because `g_params` is
unlocked global mutable state and two candidates must never be in flight at once.

### The fitness set omits where runs actually die

Added 2026-07-30. Independent of the power-level problem below, and cheaper to
fix.

The tuner scores 15 of the engine's 64 encounters. Surveying the **shipped
config across all 42 act/tier encounters** (30 episodes each, 150 sims, A20,
zero relics — so absolute numbers carry the power-level caveat below, but the
*relative* picture is what matters here):

| Tier | Win rate |  | Act | Win rate |
|---|---|---|---|---|
| basic | 0.98 |  | act1 | 0.96 |
| elite | 0.92 |  | act2 | 0.94 |
| **boss** | **0.67** |  | act3 | 0.78 |

Weakest encounters: `AWAKENED_ONE` 0.27, `DONU_AND_DECA` 0.40, `REPTOMANCER`
0.47, `TIME_EATER` 0.57, `COLLECTOR` 0.73, `AUTOMATON` 0.77, `HEXAGHOST` and
`SLIME_BOSS` 0.80. **Every weak encounter is a boss or a late elite.**

Two structural problems with the fitness set:

- **It contains 3 of the 9 bosses** (`THE_GUARDIAN`, `AUTOMATON`, `TIME_EATER`)
  and **no Act 2 elite at all** — no `GREMLIN_LEADER`, `BOOK_OF_STABBING` or
  `SLAVERS` — while runs die at a mean floor of ~26, which is Act 2. It also
  omits `HEXAGHOST`, `LAGAVULIN`, `CHAMP`, `COLLECTOR`, `NEMESIS`,
  `REPTOMANCER`, `GIANT_HEAD`, `AWAKENED_ONE` and `DONU_AND_DECA`.
- **87% of its compute buys almost no gradient.** In a 3-arm ablation, 13 of the
  15 encounters sat at or near 1.00 for every arm and together produced a
  fitness spread of 0.023; `AUTOMATON` and `TIME_EATER` alone produced 0.258 —
  **11x the signal from 13% of the episodes.** Per episode the two hard fights
  are ~70x more informative.

The saturated encounters are not worthless (they still carry HP-margin signal
and guard against regression), but they should be cheap regression checks with
few episodes, not the bulk of the budget. Reweighting toward bosses and Act 2
elites raises effective sample size at identical cost — the fitness set is
`ENCOUNTERS` in `tune_search_cma.py`, and `env.ALL_ENCOUNTERS` already carries
correct per-act/tier resources for all 42.

Note the interaction with the next section: `TIME_EATER` supplies roughly half
the fitness set's discriminating signal, but the agent reaches Act 3 in about
8.5% of runs. The objective is weighted toward content it almost never sees.

### Dead ends, measured 2026-07-30

Recorded so they are not re-run. All on the same harness, paired seeds.

| Hypothesis | Test | Result |
|---|---|---|
| More search budget helps | 6 hardest encounters, 150 → 600 sims, 25 paired episodes | **z = 0.00** (7 gained, 8 lost); win 0.607 → 0.600 |
| The per-card **draft** prior is the wrong signal; Silverbot's play-priority ordering would beat it | 600 paired fights, 3 arms | **Refuted**, z = 2.96 *against*. Removing it costs 3.2pp |
| Raising the play-priority weight recovers that loss | same, silver weight 5.0 → 20.0 | **Inert.** 0.898 vs 0.900 — quadrupling it changes nothing |
| `powerScore` = 13.9 feeds Awakened One its Strength | powerScore sweep 13.9/6/0/−10, `AUTOMATON` control | **Refuted.** 0.17 → 0.17 → 0.15 → 0.07. The engine already models Curiosity, so the *tree* can see the Strength gain; suppressing Powers in the *rollout* only degrades leaf values |

**Added 2026-07-31**, on the human benchmark (526 validation fights, 100 sims,
deterministic after the Gumbel seeding fix, so these are exact):

| Hypothesis | Test | Result |
|---|---|---|
| A cheaper leaf lets the saved time buy enough extra simulations to come out ahead | `leaf_eval_mode` value / truncated(3) vs rollout | **Refuted, decisively.** value is 6.8x faster per fight and **-19.14 +/- 1.16 HP** worse at matched sims. value at 400 sims still runs 3x faster than rollout at 100 and is **-14.96 +/- 1.01** worse |
| Filling the conditional-Power value tables, with the condition read from the deck rather than guessed | 8 previously-zeroed Ironclad Powers scored | **Inert.** +0.31 +/- 0.49 HP (t = 0.64); 314 of 526 fights play identically |

| Distilling the search into the rollout policy would raise the playout's quality | 42,495 decisions / 271,907 scored actions collected from the train split at 100 sims; nets at hidden [4]/[8]/[16] | **Refuted.** Worse at matched wall clock (**-2.12 +/- 0.84 HP** at the best weight) AND worse at matched simulations (-0.57 to -1.22 across weights), so it is not a cost problem |

| Widening `nativeActionFeatures` with action identity is what unblocks distillation | 45,770 decisions / 294,998 actions, four arms x three seeds, held out by fight (`_probe_card_identity.py`) | **Refuted as an engine change.** Identity is worth +2.3pp top-1 (0.327 -> 0.350), of which the embedding contributes +0.7pp over a free per-card scalar, against a net that costs 4.97x search speed. Full table and the two cheap findings it produced below |

| The rollout throws potions away, since a discard scored the same flat 5.0 as a drink | `rollout_potion_discard_penalty` = 50 (enough to rank every discard below every drink) vs shipped, 500 paired train fights, 100 sims | **Null.** -0.58 +/- 0.56 HP (t = -1.04), deaths 76 -> 82, with 226 of 500 fights playing differently. The rollout does discard constantly; it does not cost anything measurable. The same arm read +1.0 HP on a 120-fight val spot-check first -- a textbook selection high, caught only by re-measuring on train |

### Four search changes, all refuted on the same day (2026-08-01)

Three algorithmic changes with literature behind them, plus the rest of the potion
branch, measured the same way: single parameter against the shipped config, 500
paired train fights at 100 sims, common random numbers. Harness
`lightspeed/_param_ab.py`, which exists because nothing measured ONE parameter
with a paired standard error before. Baseline **-5.580**, 76 deaths.

| change | setting | delta HP | t | deaths |
|---|---|---:|---:|---:|
| **Max-Monte-Carlo backup** (MaxUCT) | 0.25 | +0.65 +/- 0.50 | +1.29 | 76 |
| | 0.5 | -0.42 +/- 0.58 | -0.72 | 81 |
| | **1.0** (the published algorithm) | **-1.76 +/- 0.60** | **-2.93** | 83 |
| **Gumbel-Top-k root candidates** | m = 4 | **-2.08 +/- 0.55** | **-3.78** | 82 |
| | m = 6 | -0.30 +/- 0.50 | -0.61 | 76 |
| | m = 8 | **-1.44 +/- 0.48** | **-3.01** | 85 |
| **MAST** (online per-card table) | 0.5 / 1 / 2 | -0.23 / +0.05 / +0.21 | -0.43 / +0.10 / +0.38 | 76 / 74 / 73 |
| potion danger scale | 5 / 15 | -1.04 / -0.41 | -1.98 / -0.85 | 80 / 80 |
| potion base | 12 / 25 | -0.93 / -0.93 | -1.75 / -1.82 | 78 / 82 |

**Nothing produced a gain clearing its own standard error.** Two produced
significant harm. The strongest result in the table is the one nobody predicted:
restricting the root candidate set *hurts*, decisively at m = 4. Sequential
halving's first phase, which the `visitOrder` comment upstream calls a real cost
("burns 10+ simulations trying every one once"), is apparently buying something —
searching every root action matters more than resolving a few of them well. Note
the non-monotonicity (4 bad, 6 null, 8 bad) does not fit a clean story and is not
explained; m changes the halving phase count too, so the arms are not a clean
ladder.

MaxUCT is worth recording carefully because the theory was sound and the result
is not. Keller & Helmert's argument -- that Monte-Carlo backup averages over
actions UCB1 explored to be bad, and a single-agent MDP's correct value is the max
-- predicts a gain, and the measurement says otherwise at every weight above 0.25.
Deaths rise monotonically with the weight (76 -> 81 -> 83), which is what an
optimistic value estimate looks like: the maximization bias flagged when the
parameter was written appears to dominate the bias it was meant to remove, at 100
simulations across ~10 actions.

**The caveat that applies to all four, stated so it is not mistaken for an
excuse.** These are perturbations of a converged optimum, not clean tests of the
ideas. `tuned_search_params.json` is the output of a 42-parameter CMA-ES run
against this exact objective, so every other weight is already fitted to the
Monte-Carlo backup, the full root candidate set and a potion-blind rollout. A
structural change that needs the rest of the config to move cannot show a gain
this way. The honest test is a re-tune with the parameter in the space -- which
costs ~7 hours, and which `03`'s own layer-swap section says buys benchmark HP
that does not convert to floors. That is the reason to leave these alone, not the
t-statistics.

All four remain in the engine as verified no-ops (250 val fights byte-identical
across the rebuild) and are deliberately **not** in `tune_search_human.py`'s
search space, following the precedent `power_horizon_weight` and
`boss_power_multiplier` set.

| Powers are undervalued in LONG fights, where they compound | three variants on the human benchmark: global 2x scale, an absolute remaining-HP horizon (`power_horizon_weight`), and a boss-only multiplier (`boss_power_multiplier`) | **Refuted, all three.** Boss-only is +0.01 +/- 0.18 HP on 1730 train fights -- a tight null, not an underpowered one. The horizon looked like +2.66 HP on bosses when picked by a 7-config sweep on the test split, then measured -0.84 +/- 0.40 on val and -0.13 +/- 0.27 on train |

The Power rows carry a methodological warning worth more than the result. Those
sweeps were run **on the test split**, and with ~100 boss fights per split the
per-boss standard error is ~1.4 HP, so taking the best of seven configurations
manufactures a ~2 HP "improvement" reliably and for free. Two claims that came
out of that pass -- "2x Powers helps bosses and hurts non-boss fights" and
"turning Power scoring off costs 8 boss deaths" -- were never confirmed on clean
data and should be treated as unverified. Sweep on val or train; keep test for a
single pre-registered setting.

Both parameters remain in the engine, defaulting to verified no-ops
(`power_horizon_weight = 0.0`, `boss_power_multiplier = 1.0`). They are
deliberately NOT in `tune_search_human.py`'s search space: adding parameters that
measure null only widens the surface CMA-ES can overfit.

The distillation row deserves its own note, because the premise was sound and the
measurement is the useful part. Search really is worth **+30.55 +/- 1.60 HP** over
the policy it rolls out with (1 simulation scores -31.83 against 100 simulations'
-1.28), so there was a large gap to close. Capacity was not the limit either:
top-1 accuracy against the search's picks was 0.334 / 0.340 / 0.340 at hidden
4 / 8 / 16 against a 0.202 random baseline, saturating immediately.

The limit is the **feature vector**. `nativeActionFeatures` is
`[is_attack, is_skill, is_power, is_other, target_hp_missing, target_block,
is_aoe_multi, card_pick_rate_weight]` -- it never identifies WHICH card is being
played. Everything expressible in those 8 numbers is already hardcoded in
`nativeScoreAction`, so a net over them has almost no marginal information, and
its errors cost as much as its agreements gain. Distillation through this socket
cannot work without widening the features, which is an engine change.

### Refuted 2026-07-31: widening the feature vector is not worth the engine change

The paragraph above names the feature vector as the limit. Measured directly, it
is the limit by about a twentieth of what would be needed. Harness:
`lightspeed/_probe_card_identity.py`, which needs **no engine change** --
`Action.source_idx`, `bc.hand[i].id/.upgraded/.cost_for_turn` and `bc.potions`
are all already bound, so action identity can be recovered in Python at
collection time and the accuracy question answered offline for the price of one
re-collection.

45,770 decisions / 294,998 scored actions from all 1,730 train-split human
benchmark fights at 100 sims, on the post-rebuild engine. Four arms, three seeds
each, held out **by fight** -- the existing trainer splits contiguously by
decision, which leaks, since consecutive decisions inside a fight share a deck, a
monster and most of a state. Val top-1 against a 0.202 random baseline:

| arm | hidden 8 / embed 4 | hidden 16 / embed 8 |
|---|---:|---:|
| baseline -- the engine's 18 features | 0.326 | 0.327 |
| + action-type flags and energy cost | 0.330 | 0.339 |
| + one learned scalar per identity | **0.342** | 0.343 |
| + learned embedding (the proposed change) | 0.340 | **0.350** |

The baseline reproduces the 0.334 on record, which is what validates the harness.
Both configurations give the same ordering, so this is not a best-of-N pick.

Decomposing the hidden-16 column: **+1.2pp** is just distinguishing action
*types*, **+0.4pp** is a per-card scalar, and **+0.7pp** is everything the
embedding's card x state interaction buys over that free scalar. Against this,
the h4 net at 0.334 already lost **-2.12 HP** at matched wall clock, and hidden
16 costs 4.97x search speed -- 100 sims -> ~20, worth about -4.1 HP. There is no
version of that trade that pays. **Do not widen the vector for a net.**

Two findings from the probe are worth more than the result, and both are cheap:

- **The largest single component is not card identity at all.** Every non-CARD
  action returns the identical vector `{0,0,0,1,0,0,0,0}` -- END_TURN, all 33
  playable potions and every card-select option are one symbol. That is +1.2pp of
  the +2.3pp, and it points at a hole in the hand-tuned heuristic itself (below),
  not just in the net's inputs.
- **The per-card scalar arm is already implemented.** `g_earlyActCardBias`
  (`:370`, applied `:1144`, `set_early_act_card_biases`) is exactly a per-CardId
  additive bonus in the hot path, gated to `maxHp <= 85`. Ungating it and fitting
  it to the search's own picks is the +0.4pp arm at **zero inference cost**, so
  unlike every net in this table it does not have to recover any HP to break
  even. Fitting it properly means learning a correction in the heuristic's own
  units on top of the frozen heuristic, which needs `nativeHeuristicScores`
  (`:1459`) exposed -- currently internal. Note the probe indexed by
  `(CardId, upgraded)` while the table is per-`CardId`; upgrade state looks worth
  keeping.

1.2% of val actions carry an identity never seen in training, and card-select
options are one "unknown" symbol, since `getSourceIdx` indexes a task-dependent
pile and `bc.cardSelectInfo` is not bound. Top-1 against the search's own picks
is a gate, not a deliverable.

Two cost facts worth carrying forward regardless, both contradicting the earlier
ablation that rejected `policyNetWeight`: the **7x figure in that comment is
stale** (the per-action recomputation of state features was fixed), and the real
cost is strongly width-dependent -- 1.93x at hidden 4, 3.52x at 8, 4.97x at 16,
6.75x at 32x32. Since accuracy saturates at 4 units, anything wider is pure
waste. Harnesses: `collect_rollout_policy_data.py`,
`train_rollout_policy_net.py`, `_eval_rollout_policy_net.py` -- the last compares
at matched WALL CLOCK, measuring the net's cost empirically, which is the only
comparison that decides whether to ship one.

The leaf-mode row is worth reading before anyone runs `tune_value_leaf.py`: its
ten `vf*` weights would have to recover 15 HP, which is roughly three times the
entire remaining gap between this engine and a top human. The linear estimate is
not a slightly coarser rollout, it is a different quality of play.

The common thread: all four are "make the search decide better", and all four
did nothing. At the fights that decide runs, search decision quality was not the
binding constraint — the power level it was being measured at was.

**Caveat on the sim-budget row.** It ran in the zero-relic regime on fights that
were near-unwinnable as configured, and nothing moves a fight you cannot win.
That z = 0.00 deserves a re-run with relics before being treated as settled. The
independent full-run 300 → 900 comparison (see `06-experiment-log.md`) does not
have this problem, since real runs carry real relics, and that is what the
"budget is not the lever" conclusion actually rests on.

### The fitness fights at the wrong power level

Added 2026-07-30. Largest measured effect on combat win rate found so far, and
it is a property of the objective rather than of the search.

`ACT_TIER_RESOURCES` specifies a relic count per tier — act3/boss is
`(130 hp, 40 cards, 0.5 upgrades, 2 removals, 11 relics, 2 boss relics)`.
`IroncladFightEnv` honours HP, deck size and upgrades, but grants relics **only
when a `relic_generator` is passed**, and `_worker_init` never passes one. Every
fight the tuner optimizes is fought with Burning Blood and nothing else, against
bosses a real run reaches holding 8–11 relics.

Six weakest encounters, 30 paired episodes each, 150 sims, A20, identical decks
and seeds — the relics are the only difference:

| Encounter | Burning Blood only | Tier relics | Δ |
|---|---|---|---|
| AWAKENED_ONE | 0.20 | 0.77 | +0.57 |
| DONU_AND_DECA | 0.43 | 0.93 | +0.50 |
| REPTOMANCER | 0.33 | 0.93 | +0.60 |
| TIME_EATER | 0.50 | 0.97 | +0.47 |
| COLLECTOR | 0.80 | 0.93 | +0.13 |
| AUTOMATON | 0.80 | 0.97 | +0.17 |
| **overall** | **0.511** | **0.917** | **+0.406** |

Paired: 76 relics-only wins vs 3, **z = 8.10**. HP on wins 0.486 → 0.700.

For scale, measured the same day on the same harness: removing the per-card
draft prior costs 3.2pp (z = 2.96), and quadrupling sims from 150 to 600 on
these same six encounters is z = 0.00. This effect is an order of magnitude
larger than either.

The consequence is not just the win rate. Every tuned parameter was selected to
survive Act 3 bosses at zero relics, a regime the agent never occupies. Play
patterns correct there are not obviously correct with Runic Pyramid retaining
the hand or Art of War rewarding attack-free turns, neither of which the tuner
can currently see.

**Measured twice, two ways.** A first pass used a deterministic top-N by
`RELIC_WEIGHTS`, because the sampling path segfaulted at the time; it gave
0.889 / z = 7.69, and was documented with a caveat that a top-N loadout should
be *stronger* than a typical sampled one, so +0.378 was an upper bound. That
caveat was wrong in direction. With the segfault fixed and the relic pool
cleaned of 13 non-Ironclad relics, proper sampling gives **0.917 / z = 8.10** —
higher, not lower. Both runs are 30 paired episodes per encounter on identical
decks and seeds; the numbers in the table are the sampled ones.

**Fix**: pass a `relic_generator` in `_worker_init`, then re-tune from scratch,
since every current parameter was selected against the zero-relic regime.
No longer blocked — the SCRY segfault that forced the deterministic-loadout
workaround is fixed (see `07-known-issues.md`), and the sampled path has since
run 360 fights twice with no crash, so `weighted_ironclad_relics` can be passed
directly.

## The human benchmark

Added 2026-07-31. Harness `lightspeed/_human_deck_combat.py`, validator
`lightspeed/_human_deck_eval.py`, silverbot arm `lightspeed/_silverbot_human_deck.py`,
cached fights `runs/human_fight_benchmark_100.json`.

Every combat measurement before this was either a synthetic-deck win rate (wrong
power level, per the section above) or a full-run floor count (±0.5 floors of
noise, hours per verdict). Neither could say what *good* looks like.

This one replays 100 real A20 Heart runs floor by floor with the importer's own
reconstruction and rebuilds each fight exactly as the human entered it — his
deck, his relics, his potions, his HP, his encounter — then plays it with our
search. His damage on that floor is in the archive, so **every fight carries its
own paired human result**. 2841 fights resolve; 5 event-fight names do not.

Objective is `mean(human_damage - our_hp_paid)`: 0 is parity, a death pays all
remaining HP. Use the **difference, not the ratio** — the ratio has mean 5.25 /
sd 9.80 / p90 13.0 because it explodes on fights he took 1–2 damage on.

### Reconstruct the potions, or the number is wrong by a third

The first version omitted potions. He drinks one in **23.7% of fights and 38.6%
of elites**, and the archive records `potions_obtained` / `potions_used` /
`potions_discarded` per floor, so the inventory is a replay of those three.
Adding them moved the shipped config from **−13.515 to −8.784 on the same
fights** — 4.7 HP, about a third of what had been reported as our deficit.

An earlier revision of this file claimed we pay **1.96x** a top human's HP. That
was measured potionless on 20 runs and is wrong. With potions and all 100 runs
the shipped config of the time paid **1.29x**. Note the direction of his potion
use, which is not the obvious one: his damage is *higher* with a potion (22.3 vs
11.9) because he banks them for hard fights.

### The bracket: silverbot on the same fights

The human is an upper reference, not an achievable one, so 1.29x alone could not
say whether it was near a ceiling. Silver Automaton runs on this benchmark with
no translation — its CardId, RelicId, Potion and MonsterEncounter enums are
integer-identical to ours across all 371/181/44/64 members — which turns one
number into a bracket. It must run in its own process, since both engines' Python
modules are named `slaythespire`.

It **cannot play 101 of the 2841 fights**: 11 of the 100 runs carry Prismatic
Shard, which legitimately offers Silent/Defect/Watcher cards to an Ironclad, and
silverbot aborts the process (assert, not exception) on an unimplemented
off-class card. Our engine implements all four characters and plays them. The
benchmark carries a precomputed `off_class_cards` field to filter them.

Measured on the 528 playable test fights, 100 sims:

| | objective | ratio on wins | deaths |
|---|---|---|---|
| ours, before the 2026-07-31 tuning | −11.83 | 1.45 | 88/528 |
| ours, after the first config apply | −6.84 | 1.14 | 75/528 |
| **ours, after the 42-param run** | **−5.78** | **1.11** | **78/528** |
| silverbot | −6.58 | 1.01 | 84/528 |

Silverbot started **1.8x better** and finished the day **0.80 HP behind** us, on the metric most generous to it (a death priced at zero cost beyond the HP held). We also die less (78 vs 84) and hold more HP per fight (48.67 vs 47.87), so under any nonzero death penalty the margin widens. Both of our rows are measured on the post-rebuild engine, so the config alone accounts for **+4.99 −/+ 0.69 HP (t = 7.27)** of that move. Per room, ours vs
theirs: bosses **1.28 vs 1.44** (we are now better), elites 1.42 vs 1.15 (they
are), normal fights 0.86 vs 0.78. We die less than they do; they spend less HP on
the fights they win.

Caveat on fairness: "100 sims" is not guaranteed to mean equal work in both
engines — theirs is `simulation_count_base` with `boss_simulation_multiplier=2`.
Our side is largely insulated because our play is flat from 43 to 1500 sims, but
treat the magnitude as approximate and the direction as solid.

### Search budget is settled, in the regime the dead-ends table asked for

The "more search budget" row above carries a caveat: it ran zero-relic on
near-unwinnable fights and "deserves a re-run with relics before being treated as
settled." That re-run now exists, on real decks and real relics. Paired, scoring
a death as the HP it actually cost:

| | n | HP delta | t | deaths |
|---|---|---|---|---|
| 43 → 100 sims | 169 | −0.68 ± 0.99 | −0.69 | 44 → 47 |
| 100 → 300 | 169 | −1.49 ± 0.97 | −1.54 | 47 → 43 |
| 43 → 300 | 169 | −2.17 ± 1.10 | −1.98 | 44 → 43 |
| 300 → 1500 | 40 | −0.97 ± 1.23 | −0.79 | 9 → 9 |

**Flat across 35x compute.** At 5x budget, 22 of 40 fights took *identical*
damage. The whole budget dimension is worth ~3 HP against a gap of ~11, so the
rollout policy — not the search — is the ceiling. It also reprices the policy
net: its ablation rejected it for costing ~7x search speed, but 7x from 300 sims
lands at ~43, which is only −2.17 HP.

### Why this is a better tuning objective

`tune_search_human.py` tunes against it. The docs above diagnose the old fitness
set as 87% dead gradient because 13 of 15 encounters sit at ~1.00 win rate; that
reproduces here — **430 of 560 fights (77%) are wins that score a flat 1.0 under
a win-rate objective.** The fix taken here is not the reweighting proposed above
(which discards the easy fights) but changing what is measured: with a human
damage number per fight, a fight we always win still says whether we won it at 8
HP or 16, and every fight stays live.

**Split three ways by run_id** — 60 train / 20 validation / 20 test. Validation
decides which point on the trajectory to keep and when to stop; test is read
once. Both are needed: 42 parameters will fit whatever they are shown, and the
training score cannot detect it. Two failure modes are already on record.

Scoring every candidate on the *same* fights makes the confirmation round a test
of search-RNG luck only, blind to fight-set overfitting — three generations of
that measured **−3.13 ± 1.14 HP worse than shipped** on held-out while looking
better on train. Each evaluation now draws a fresh 35% sample.

And a run that improves train by 11.4 HP can deliver 4.12 on held-out. The first
7-hour run did exactly that: it made all its progress by generation 200, then
random-walked 500 more at a 48.4% acceptance rate — exactly the coin-flip rate of
a converged optimizer — leaving a final artifact that was an arbitrary sample of
a plateau. Warm-start from the shipped config rather than compiled defaults, or
roughly 2.4 HP is spent rediscovering ground already held.


## The layer swap: combat is not the binding constraint (2026-07-31)

Harness: `lightspeed/eval_heart1_hybrid.py` (pre-existing, previously unused) —
Silverbot's `heart1.pt` overworld policy driving OUR engine and OUR combat
(`native_playout_current_battle`), so a comparison against v31 holds combat
byte-identical and isolates the run layer. Their policy is pure inference
(`choose_overworld_action`: one forward pass, softmax, pick — no search), the
same architecture class as v31.

All A20, seeds 21.4M+, combat at 100 sims unless stated:

| configuration | mean floor | wins |
|---|---|---|
| v31 + our combat (n=500) | 22.36 | 0 |
| **heart1 + our combat (n=24, paired vs v31)** | **37.00** | **3/24** |
| heart1 + their combat, their engine, 2k sims (n=24) | 39.29 | 5/24 |

Paired on the 24 shared seeds: **+15.71 ± 3.13 floors (t = +5.02)**, 8 runs
reaching Act 4, 3 victories — the first A20 wins this stack has produced in any
configuration. Their full stack at 2k sims matches their own documented 10k-sim
benchmark (18.6% ± 2.4%, n=1000, avg floor 39.9), so 2k→10k is flat for them
too; the 50,000 in their `SearchAgent.h` is a code default, not what their
evals use.

Simulation budget, measured everywhere it could matter:

| experiment | result |
|---|---|
| v31, 100 → 1500 sims, 200 paired seeds | +0.115 ± 0.507 floors, 0 wins both arms |
| hybrid, 100 → 1000 sims, 12 paired seeds | −7.25 ± 5.23 floors, wins unchanged 2/12 |
| benchmark HP, ours, 100 → 1600 sims | **+3.31 HP** (earlier "flat 43→1500" claim was wrong — measured on 40 Act-1 fights, retracted) |
| config tuning worth +6.05 benchmark HP | **−0.23 ± 0.29 floors** |

The pattern: sims and config both buy benchmark HP; neither buys floors above
~100 sims. Below it combat matters enormously (5 sims costs −4.63 floors), so
the system sits just past a sharp knee. Benchmark HP is a valid human-relative
combat measure and an invalid floor proxy — the 2026-07-31 tuning optimized it
faithfully and moved nothing that matters.

Two caveats keep this from being the final word on combat. Our search is
**draw-order clairvoyant** ([07-known-issues.md](07-known-issues.md)) — the
layer-swap attribution survives (both arms share it) but every ours-vs-theirs
combat comparison flatters us. And the n=24/n=12 samples bound win-rate claims
loosely; the floor deltas are the load-bearing numbers.

Where this leaves the project: the run policy is worth ~15 floors and the
difference between 0% and ~20% win rate; heart1 proves the target reachable
with our own combat underneath; the hybrid harness measures any candidate
directly, with no proxy. heart1 is PPO-trained over full episodes where v31 is
outcome-supervised on MCTS labels — that training difference is now the
highest-value open question.

## Character coverage

The heuristic classifiers (`isAoeCard`, `isVulnerableApplier`, `isDefensiveCard`,
`isSilentPoisonApplier`, `nativeImmediateBlockBase`) are character-agnostic and
carry entries for all four classes. Only Ironclad is tuned and validated;
`cardPickRateWeight` was learned from ~5,500 Ironclad decisions and Silent falls
back to a 0.05 smoothing floor. Defect and Watcher classifier data is populated
but untested.

Their **card logic is not** unimplemented, which earlier revisions of this file
and of [07-known-issues.md](07-known-issues.md) both claimed. The engine
implements every playable card of all four characters; what was missing was
card-select *enumeration* in the search bindings, fixed 2026-07-31. Details and
the measurement are in [07-known-issues.md](07-known-issues.md); the harness is
`lightspeed/_class_card_audit.py`.
