# Validating the engine against the real game

The engine's job is to be Slay the Spire. Until 2026-07-31 nothing checked that
directly — correctness was argued from code review and from `silverbot-reference`,
which is a fork of the same upstream engine and therefore shares any inherited
bug. This documents the oracle that does check it directly, what it found, and
what it still cannot see.

Harness: `lightspeed/_game_jar_audit.py`.

## The oracle

Slay the Spire is installed on the development machine, and a JDK is present, so
the game's own `desktop-1.0.jar` can be disassembled with `javap` and read for
its actual constants. That is ground truth: complete, offline, and first-hand.

Two properties make the comparison tractable.

**Only A0 and A20 matter here, and that removes the hard part.** Every ascension
gate in the game sits at 2, 3, 4, 9, 17, 18 or 19. A0 is below all of them and
A20 is above all of them, so at those two points the gate *thresholds* are
irrelevant — only the base value and the top-tier value matter. This is why
TheCollector's `A_2_BLOCK_AMT`, which is actually applied on an A9 gate, is
harmless at A0 and A20 even though the constant name misleads.

**Ascension does not scale cards.** A card is just `(base, upgraded)`, read from
`<init>`'s `putfield baseDamage` and `upgrade()`'s `upgradeDamage(delta)`.

## Oracles, ranked

| Oracle | Coverage | Catches |
|---|---|---|
| **`desktop-1.0.jar`** | complete | absolute correctness |
| `silverbot-reference` | 121 of our 345 cards; full monster roster | drift only — shared upstream bugs invisible |
| `sts_raw_states.log` | 28 distinct cards, one session | absolute, but data fields only, not effects |
| `slay-sim/sts/` (Python engine) | Ironclad + colorless | independent, but has documented approximations |

Use the jar. The others were how this started, and silverbot in particular is
still a good *detector* — its Era-2 source audit means a disagreement with it is
a strong prior that something is wrong — but it is not the authority.

## What was verified clean

- **Card damage**: all 120 attack cards, base and upgraded, identical to the
  game. Zero differences.
- **Card block and magic numbers**: 282 cards screened, 271 clean; the 11 flagged
  were all screen artifacts, values living in helper actions
  (`FeedAction`, `c.specialData` growth, the Nightmare start-of-turn handler)
  rather than as literals. Heavy Blade resolved *correct* here — the game's
  magic number is the (3, 5) strength multiplier and ours computes 3x/5x.
- **Per-card data tables and predicates**: `cardTypes`, `cardRarities`,
  `cardTargets`, `cardSortedIdx` identical to silverbot; `cardColors` differs
  only in the 8 entries we deliberately fixed. `isCardEthereal`,
  `isCardStrikeCard`, `doesCardSelfRetain`, `isStarterStrikeOrDefend`,
  `cardTargetsEnemy`, `isXCost` identical; `isCardInnate` and `doesCardExhaust`
  each carry one entry we have and silverbot lacks (PRIDE, TERROR), both correct
  in ours.
- **Monsters**: `initHp` across 66 monsters and `preBattleAction` across 33,
  identical. Collector, Giant Head, Corrupt Heart, Nemesis and Repulsor all
  confirmed correct in ours — in several of those silverbot's implementation is
  the thinner one.

The engine's bulk data is in good shape. The failures cluster in **state
transitions** and **ascension tier tables**, not in the numbers.

## What it found

All four are recorded in [07-known-issues.md](07-known-issues.md) with fixes.

| Bug | A0 | A20 | Where |
|---|---|---|---|
| Lagavulin retains Metallicize 8 on its scheduled wake | wrong | wrong | Act 1 elite, ~every run |
| Champ Gloat strength `{3,4,5}` vs the game's `{2,3,4}` | 3 vs 2 | 5 vs 4 | Act 2 boss |
| Darkling Chomp one hit vs `CHOMP_AMT = 2` | 8 vs 16 | 9 vs 18 | Act 3 |
| Writhing Mass flail block 16/18 vs 15/16 | 16 vs 15 | 18 vs 16 | Act 3 |

Two notes on reading the jar, both learned the hard way:

- **Field names are not authoritative.** TheCollector's block constant is named
  `A_2_BLOCK_AMT` but the constructor gates it at A9, and the `+5` at A19 is
  applied in `takeTurn` rather than stored at all. The bytecode is the answer;
  the constant table is a hint.
- **A value can be correct without being a literal.** Writhing Mass's flail block
  is not a constant anywhere in the class — the `GainBlockAction` is constructed
  from the same `DamageInfo` as the attack, which is how the game makes block
  equal damage. Our invented `18` corresponds to nothing in the game.

## Behaviour, not just numbers

Everything above compares constants. Constants being right says nothing about
what a card *does* — the Armaments leak had every number correct and still cost
3.2 floors. `lightspeed/_card_effect_audit.py` closes part of that gap by
comparing **which effects a card queues**.

A card's `use()` queues actions; our case bodies queue a parallel sequence of
`Actions::` helpers and `Buff/Debuff<MS::X>` calls. Canonicalising both into one
vocabulary makes them comparable, and that catches a missing effect, an extra
effect, the wrong power, or the wrong target scope.

**Ironclad reconciles completely.** 75/75 cards, with the residual being naming
(our `XAction` against the game's `X`) and two deliberate implementation
choices: Berserk, where we do `++player.energyPerTurn` instead of modelling a
`BerserkPower` — equivalent, and it stacks the same — and Armaments, where we
split the upgraded branch out rather than handling it inside one action. **No
behavioural defect was found in the Ironclad card set.**

Three bytecode traps, each found by falling into it:

- javap names a class **twice** per site (the `new` opcode and the matching
  `invokespecial`), so counting both doubles every effect.
- Many cards open `use()` with `if (Settings.isDebug) { ... }` — **Bash's debug
  branch calls `DamageAllEnemiesAction`**. That is dead code in a real game and
  has to be excluded by following the `ifeq` target, or the signature is wrong
  for every card carrying one.
- VFX/SFX/Wait actions are cosmetic noise.

**All five colours are mapped**: red 66/75, green 62/72, blue 54/73,
purple 62/72, colorless 29/39 — and **every one of the 58 residuals was then
read individually against both implementations, not sampled.** No behavioural
defect was found in any of them. Each resolved as naming, a deliberate
implementation choice, or the tool's ceiling.

The ceiling accounts for most of them, and the worked examples are worth keeping
because they all read as "ours is missing an effect" and none is: Melter zeroes
`monsters.arr[t].block` in a lambda, Steam Barrier decays through `specialData`,
Streamline calls `setCostForTurn`, Thunder Strike loops over
`lightningChanneledThisCombat`, Vault sets `skipMonsterTurn`, Rip and Tear and
Ragnarok pick random targets inside lambdas, and Foreign Influence, Omniscience,
Wish, Nightmare, Setup, Seek and Recycle all open card-select screens rather
than queueing an `Actions::` helper.

That ceiling is the interesting part. Our engine expresses many effects as
inline lambdas or direct field writes, and a regex over `Actions::` calls cannot
see inside them — Chaos channels its orbs in a lambda, All for One filters the
discard pile in one, Genetic Algorithm writes `specialData` directly. All three
read as "ours is missing an effect" and none of them is. The residual counts
therefore do not go to zero and **should not be chased with more aliases**.
Going further means executing both engines and comparing state — real
differential testing — not more parsing.

Two aliases were caught masking real problems while this was built, which is the
standing argument for adding them one at a time with the card open.
`SECONDWIND -> EXHAUSTMANY` made Second Wind look like it opened a card-select
when it does not. `WATCHER -> PWR:MANTRA` hid an extractor bug: the game nests
`powers/watcher/`, and a greedy path pattern was capturing the *subpackage*
instead of the class, collapsing every Watcher power to a single symbol. Fixing
that one regex moved purple from 31/72 to 48/72 before any alias was added.

## Relics

Checked 2026-07-31 (`lightspeed/_relic_audit.py`), and they had never been
checked before — which mattered, because the CMA-ES work measured relics as
worth +0.406 win rate at the encounter level.

**Constants**: 64 of 76 comparable relics clean. All 12 flags were read; none is
a defect. Five are one systematic thing worth knowing — our `bc.turn` is
0-indexed where the game's constants are 1-based, so Captain's Wheel is
`turn == 2` against `TURN_ACTIVATION=3` and both are correct. Do not "fix" those
to match the constant.

**Inert relics** — the more useful pass, and one the jar is not even needed for.
Eight relics sit in a live pool but are read by no behaviour code at all:
obtainable, displayed, saved, and doing nothing. Two are Ironclad-reachable and
are real gaps — **Toy Ornithopter** (common; heal 5 HP per potion, and the
search does drink potions) and **Dolly's Mirror** (shop; `onEquip` calls
`makeStatEquivalentCopy`, duplicating a chosen card into the deck with its
upgrade state and accumulated stats intact).
Frozen Eye is also inert but arguably correctly so, since its effect is to show
a human the draw order the simulator already knows. The other five are Silent
and Defect relics. Details in [07-known-issues.md](07-known-issues.md).

**Hooks** — the third pass, and the one that gets at *when* a relic acts rather
than what it is worth. Each game relic overrides named hooks (`atTurnStart`,
`onUseCard`, `onManualDiscard`, `onVictory`, ...); ours checks relics inside
particular C++ functions. Comparing the counts finds a relic we fire in one
place that the game fires in three. 148 relics have gameplay hooks; 19 came out
thinner on our side; **all 19 were read and none is a defect.**

They share one cause worth recording, because it will flag again every time:
the game gives each relic its own counter and therefore needs explicit
`atTurnStart`/`onVictory` hooks to reset it, while we read shared per-turn
counters (`attacksPlayedThisTurn`, `skillsPlayedThisTurn`, `cardsPlayedThisTurn`)
that reset centrally. Kunai, Shuriken, Ornamental Fan and Letter Opener all flag
for that reason and all four are right. Art of War (`attacksPlayedThisTurn == 0`),
Pocketwatch (`cardsPlayedThisTurn <= 3`) and Unceasing Top are the same story.

So every relic has now been checked on three axes — is it read at all, are its
constants right, does it act in as many places as the game acts. **No relic
defect remains on the Ironclad path.**

Two lessons generalise past relics. "Is this value right" is a weaker question
than "is this ever read at all" — the second needs no oracle and found more. And
a structural comparison between two differently-built engines produces flags by
construction, so the tool is only as good as the discipline of reading every
one of them.

## The documentation was the reason nobody looked

`sts_lightspeed/README.md` carries an "Implementation Progress" list asserting
**"All enemies / All relics / All Ironclad cards / All colorless cards"**. The
relic line is false — eight relics sat in live pools reading no code at all —
and the claim is inherited from gamerpuppy's upstream text, so it predates this
fork.

That is worth recording as a pattern rather than a one-off, because it is the
second instance in two days. `docs/` asserted Defect and Watcher were "largely
unimplemented" when the engine implements all four characters; the engine's own
README asserted complete relic coverage when seven relics did nothing. **The
written claims and the code had drifted in opposite directions, and neither had
a test.** Every claim of the form "all X are implemented" in this repo should be
assumed unverified until a harness backs it.

## What it still cannot see

- Values inside helper actions: `FeedAction`'s max-HP gain, Windmill Strike's
  per-retain growth, Nightmare's copy count, Omniscience's play-twice. Each
  needs its own check.
- Card **costs**, **exhaust/ethereal/innate** and monster **move-selection AI**
  have been compared against silverbot but not yet against the jar.
- Ordering and interaction effects. Every constant can be right while the
  modifier pipeline applies them in the wrong order — which is precisely what
  silverbot's Heavy Blade and Perfected Strike fixes were about, and what
  `_engine_invariants.py` also cannot see.
- 63 of our card cases found no name match in the jar, mostly naming mismatches
  (`STRIKE_RED` vs `Strike_Red`). Those are unverified.

## Silverbot's own audit, as a checklist

`silverbot-reference/README.md` documents an "Era 2 — source audit + live
bridge" programme: the engine validated against decompiled game source *and*
against the live game, with any divergence in predicted damage, HP, block,
intent or outcome "treated as a crash and root-caused." Our fork never had that
pass, which is why the four bugs above exist.

Their named fixes are a ready-made checklist, and the jar can adjudicate each:
ascension tier gates (Champ Gloat, Collector block, Gremlin Leader Encourage),
Awakened One rebirth, the Darkling two-phase revive, a shared player
damage-modifier pipeline plus per-card fixes for Heavy Blade / Mind Blast /
Rampage / Searing Blow / Perfected Strike, Necronomicon's duplication gate,
Centennial Puzzle's latch, Ritual applying a turn late, and Time Eater / Time
Warp end-of-turn sequencing. `silverbot/bridge/REMAINING_DIVERGENCES.md`
catalogues what they still know to be wrong.
