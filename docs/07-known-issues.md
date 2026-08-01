# Known issues

Verified against the tree on 2026-07-30. Each entry says how it was checked.

## Fixed: combat upgrades leaked into the master deck

**Status: fixed 2026-07-30, engine rebuilt at 20:44.**

`cardOnExit` (`sts_lightspeed/src/combat/BattleContext.cpp:608`) used to contain:

```cpp
if (c.isUpgraded() && !deckCard.isUpgraded()) {
    deck.upgrade(deckIdx);
}
```

Cards carry a `uniqueId` indexing the master deck, so any card that *ended a
combat upgraded* had that upgrade written through permanently. Armaments is the
main source: it upgrades a card "for the rest of combat", `chooseArmamentsCard`
correctly keeps that local to the hand, and then this clause made it permanent.
Roughly one free permanent upgrade per Armaments play. The clause is local to
this fork — `silverbot-reference`'s `cardOnExit` has only the Ritual Dagger line.

The current code keeps the Ritual Dagger `misc` write and a comment recording
what was removed.

**Effect, measured on one checkpoint and 12 seeds with the engine as the only
variable:** upgrades per run 7.6 → 1.5, mean floor 28.4 → 25.2.

**Now pinned by an invariant.** `lightspeed/_engine_invariants.py` asserts that a
battle cannot change the master deck, which is this bug stated as a property —
it would have failed on the first fight containing an Armaments rather than
waiting for a run-economy audit to notice 7.6 upgrades against 137 REST
campfires. Measured 2026-07-31 over 103 battles across all four characters:
**0 upgrades gained during a battle, 5 gained outside one**. The second number
is what makes the first meaningful — upgrades do occur in these runs, so the
zero is the check being exercised rather than nothing ever changing.

Two design notes worth keeping. The check must go through
`native_playout_current_battle_result`; `new_battle()` builds a `BattleContext`
decoupled from `GameContext.deck`, and a write-through is invisible there
(verified: 0/4 fights mutated `gc.deck` that way), which is why the leak
surfaced in run economy and not in any combat test. And the harness self-tests
by corrupting a deck deliberately and confirming the comparison fires — a
harness that never reports anything is indistinguishable from one that cannot.

**Why it still matters:**

- Every checkpoint through v33 was trained and evaluated under the bug. Their
  observations encode inflated `deck_upgrades` and their value heads learned
  returns under easier dynamics.
- Every label dataset on disk is pre-fix, including v31's 4,008 rows and the
  ~6,450 rows from the abandoned v34 run.
- The CMA-ES search parameters were tuned against the buggy dynamics.
- **Any evaluation run before 20:44 on 2026-07-30 is not comparable to one run
  after.** The 200-seed rebaseline shows the ranking survives and every absolute
  number moves: v28 23.70 → 21.73, v31 26.30 → 23.57.

## Fixed: played cards were never removed from hand (live bridge)

**Found and fixed 2026-07-30.** Any
`BattleContext` built by `sts.build_battle_context()` — the constructor the live
bridge uses, `sts/bridge/native_recommend.py:372` — let every card in hand be
played an unlimited number of times.

Reproduced directly: a single unupgraded Bash, played four times from one hand,
killed a 44 HP Jaw Worm. The card was copied into the discard pile on each play
but stayed in hand.

```
play 0: monster_hp=36  hand=['BASH']  discard=1
play 1: monster_hp=24  hand=['BASH']  discard=2
play 2: monster_hp=12  hand=['BASH']  discard=3
play 3: monster_hp=0   hand=['BASH']  discard=4   -> PLAYER_VICTORY
```

**Root cause — a signedness mismatch.** `buildCardInstance`
(`bindings/slaythespire.cpp:3265`) never assigns a `uniqueId`, so every card
keeps `CardInstance`'s default of **−1** (`CardInstance.h:28`). On play,
`BattleContext::useCard` calls `cards.removeFromHandById(c.uniqueId)`
(`BattleContext.cpp:947`). That parameter is declared `std::uint16_t`
(`CardManager.h:67`) while `getUniqueId()` returns `std::int16_t`
(`CardInstance.h:45`), so −1 arrives as 65535, the comparison
`hand[i].getUniqueId() == uniqueId` promotes to `-1 == 65535`, and **nothing ever
matches**.

**Scope.** Bridge only. Cards built through the normal
`GameContext` → `new_battle` path carry real sequential uniqueIds (verified: a
starting Ironclad hand shows ids 0,1,3,9,10), so training and evaluation are
unaffected — confirmed by re-running the same probe there, where Armaments and
Warcry conserve card count exactly and only Dual Wield adds a card, which is what
Dual Wield does.

**Impact while it was live.** The search backing every live recommendation, and
therefore autobattle, planned against a state where the hand was an infinite
resource, and would happily recommend lines that could not be executed. Any
recommendation logged before this fix should be treated as unreliable.
`tests/test_native_recommend.py` did not catch it because it checks id mapping and
recommendation shape, not card conservation.

**Fix applied.** Two changes:

1. `nativeBuildBattleContext` now assigns sequential uniqueIds across all four
   piles as it builds them, and sets `nextUniqueCardId` from that counter rather
   than recomputing it from pile sizes.
2. `removeFromHandById` now takes `std::int16_t`, matching `getUniqueId`, so an
   unset −1 can never again be silently converted into an unmatchable 65535.

**Verified after rebuild:** the Bash repro plays exactly once and leaves the
hand; card totals are conserved across Strike / Bash / Armaments / Warcry plays
(delta `{}` in every case); hand cards now carry real ids. The normal
`new_battle` path is unchanged (11 → 11 cards), the 157-test suite passes (including six new regression tests, `tests/test_engine_card_identity.py`), and
the 200-seed paired baseline is materially unmoved: v28 21.73 → 21.76,
v31 23.57 → 23.57, paired delta +1.83 → **+1.80 ± 0.55**
(`runs/postfit_colorfix_v28_v31_200seeds.jsonl`).

## Fixed: `cardColors[]` had 8 wrong entries

**Fixed 2026-07-30.** `cardColors[]` at `include/constants/Cards.h:424` is a flat
table indexed directly by `getCardColor()` (`:450`). Eight entries were wrong, in
two adjacent index ranges (46–49 and 72–75), each block rotated by one position:

| Idx | Card | Was | Now | Character |
|---|---|---|---|---|
| 46 | Brilliance | BLUE | PURPLE | Watcher |
| 47 | Brutality | GREEN | RED | Ironclad |
| 48 | Buffer | PURPLE | BLUE | Defect |
| 49 | Bullet Time | RED | GREEN | Silent |
| 72 | Collect | BLUE | PURPLE | Watcher |
| 73 | Combust | GREEN | RED | Ironclad |
| 74 | Compile Driver | PURPLE | BLUE | Defect |
| 75 | Concentrate | RED | GREEN | Silent |

Neighbouring entries (Bowling Bash 45, Bullseye 50, Cold Snap 71, Conclude 76)
were correct and were left alone; the array is still 371 entries, matching
`cardNames[]`.

**Verified after rebuild** by querying `sts.get_card_color()`: all eight now
correct, controls (Bash, Neutralize, Zap, Eruption, Burn) unchanged, and **all 72
cards in the Ironclad transform pool now report RED** — which is what closes the
transform bug below.

A third bug surfaced during that check and was fixed too: the pybind `CardColor`
enum binding was missing `.value("BLUE", …)` entirely, so `sts.CardColor.BLUE`
did not exist and every blue card rendered as `CardColor.???`.

**Card rewards, pools and shops are not affected.** `CardPools.h` uses explicit
hard-coded per-character lists and never reads `cardColors`.

Only three consumers exist, one of which is a standalone binary:

| Site | Consequence |
|---|---|
| `bindings/bindings-util.cpp:91` — `NNInterface::createOneHotCardEncodingMap()` filters on `getCardColor(cid) == RED` | Brutality and Combust were absent from that vocabulary and collapsed onto fill index 0; two slots were wasted on Silent cards. Nothing consumes this map. |
| `src/game/GameContext.cpp:1633` and `:1670` — `returnTrulyRandomCardFromAvailable` / `getTransformedCard` | For an Ironclad transforming Brutality or Combust the exclusion test failed, so **a transform could return the same card it was given**, p = 1/72. This was the one user-visible consequence. |
| `apps/small-test.cpp:66,122` | validation binary only |

**Why the old deferral rationale did not hold.** The stated reason for not
fixing this was that changing the vocabulary invalidates trained nets. That is
true of `NNInterface`'s one-hot map — but the whole-run transformers embed cards
by **raw `CardId`** and never touch `NNInterface`; nothing in `lightspeed/`
references it, and its only C++ consumer is its own constructor. Correcting the
table shifted no embedding row in v13–v36.

Confirmed empirically: the 200-seed paired baseline is materially unchanged after
the fix — v28 21.73 → 21.76, v31 23.57 → 23.57, paired delta +1.83 → +1.80. The
small v28 drift is expected: transform events now return a different card, at the
same RNG cost (one `rng.random` call either way).

**The transform consequence, traced out** — for an Ironclad transforming
Brutality, before the fix:

1. `getCardColor(BRUTALITY)` returned `GREEN`.
2. `cc` is `IRONCLAD = 0`, so `static_cast<CardColor>(cc)` is `RED = 0`.
3. `excludeInPool = (rarity != BASIC) && (RED == GREEN)` was therefore **false**,
   so the `else` branch ran: `return pool[rng.random(poolSize - 1)]`.
4. `Random::random(range)` is `nextInt(range + 1)` (`Random.h:154`) — uniform over
   `[0, poolSize-1]`, i.e. the **entire** 72-entry Ironclad transform pool.
5. `BRUTALITY` is entry 68 of that pool and `COMBUST` entry 38 (`CardPools.h`), so
   each could transform into itself with probability 1/72.

With the colours corrected, `excludeInPool` is true for every Ironclad pool card,
so the exclusion branch (`rng.random(poolSize - 2)` plus a shift on collision)
runs and a transform can no longer return the card it was given.

**Still open:** `createOneHotCardEncodingMap()` remains coupled to
`getCardColor()`. Giving it an explicit Ironclad card list — the approach
`CardPools.h` already takes — would decouple the (unused) NN vocabulary from the
colour table for good. Not done, since nothing consumes that map today.

## Fixed: the search could not enumerate the other three characters' selects

**Found and fixed 2026-07-31.** Three separate defects, none reachable from the
Ironclad training path, which is why all three survived. Harness:
`lightspeed/_class_card_audit.py`; pinned by `tests/test_class_card_selects.py`.

**1. Five card-select tasks returned an empty action vector.**
`Action::enumerateCardSelectActions` (`src/sim/search/Action.cpp`) covered 20 of
the 26 `CardSelectTask`s. `DISCARD`, `HOLOGRAM`, `NIGHTMARE`, `SETUP` and
`RETAIN` fell to `default:` and returned an **empty** vector — the same failure
`InputState::SCRY` had: `nativeHeuristicPick` indexes `legal[0]`
(`slaythespire.cpp:1286`) and `nativeExpandLeaf` gives the searched node zero
edges. Every one belongs to another character:

| Task | Triggered by | Character |
|---|---|---|
| `DISCARD` | Acrobatics, Prepared, Concentrate, Dagger Throw, Survivor, Tools of the Trade | Silent |
| `HOLOGRAM` | Hologram | Defect |
| `NIGHTMARE` | Nightmare | Watcher |
| `SETUP` | Setup | Watcher |
| `RETAIN` | Well-Laid Plans | Watcher |

Measured before the fix: eight cards reached `CARD_SELECT` with `legal=0`, and
`native_playout_battle` on a Silent deck **segfaulted** (exit 139) at the first
fight. The execution half was never the problem — `Action::execute` and
`isValidMultiCardSelectAction` already handled all five; only the enumeration
was missing, so the fix is additive.

**2. `SEEK` and `MEDITATE` enumerated actions the validator rejects.** `SEEK`
emitted every *pair* unconditionally, so an unupgraded Seek (`pickCount` 1)
offered `C(n,2)` invalid actions; `MEDITATE` sat in the "just dont deal with
this" group whose mask-0 action has size 0, which fails its exact-count check.
This is *worse* than an empty list: an invalid action still reaches
`executeMultiCardSelectActionHelper` and acts, after `Action::execute` writes a
rejection dump to `std::cerr` that the inert assert never stops.

All four tasks now share one enumeration that keeps every subset
`isValidMultiCardSelectAction` accepts, so enumeration and validation cannot
drift apart again.

**3. Four cards were dispatched to a switch that did not implement them.**
`BattleContext::useCard` switches on `CardInstance::getType()`, i.e.
`cardTypes[]`, and hands off to `useAttackCard` / `useSkillCard` /
`usePowerCard`. A `case` in the wrong one of the three means the card falls to
that switch's `default:` and **does nothing at all**:

| Card | Character | `cardTypes` | case was in | should be |
|---|---|---|---|---|
| Blizzard | Defect | ATTACK | `useSkillCard` | `useAttackCard` |
| Scrape | Defect | ATTACK | `useSkillCard` | `useAttackCard` |
| Self Repair | Defect | POWER | `useSkillCard` | `usePowerCard` |
| Pressure Points | Watcher | SKILL | `useAttackCard` | `useSkillCard` |

In all four the **placement** was wrong and the table was right. That is worth
stating explicitly because the failure looks symmetrical from inside the file —
a mismatch gives you no indication which side moved — and Blizzard in
particular reads like a table error, since a card whose case sits among the
skills and whose damage is computed rather than dealt from a base value looks
like a Skill. It is not: `cardTypes` here agrees entry-for-entry with
`silverbot-reference`'s independent copy of the same upstream table on all four
cards. **Check the reference copy before editing `cardTypes`** — the table was
also checked for the transposition pattern `cardColors` had and is clean, every
neighbour of all four indices being correct.

This is a recurring defect class, not a one-off: `BattleContext.cpp:2229`
already records `BURST` being "placed in `usePowerCard()` by mistake, which
asserted/aborted on the very first real play." It aborted then because that was
a debug build. Under `-DNDEBUG` it would have silently done nothing instead,
which is exactly how these four went unnoticed — see the build note below.

**Ironclad training and evaluation are unaffected.** All nine cards are Silent,
Defect or Watcher; the Ironclad fuzz over 57,908 decision points found no empty
enumeration, and no floor number moves.

## Fixed 2026-07-31: four monster behaviours disagreed with the real game

**Found and fixed 2026-07-31.** All four confirmed against the game's own
`desktop-1.0.jar`, not inferred — see [11-engine-validation.md](11-engine-validation.md)
for the method and `lightspeed/_game_jar_audit.py` for the harness. Ordered by
impact at A0 and A20, the only ascensions this project runs.

### 1. Lagavulin keeps Metallicize on its scheduled wake

Our engine has two wake paths and only one cleans up. The damage-triggered wake
(`Monster.cpp:401`, `:465`) clears `ASLEEP` and does
`decrementStatus<MS::METALLICIZE>(8)`. The scheduled turn-3 wake
(`MonsterSpecific.cpp:890`) does **neither**, so a Lagavulin that wakes on
schedule keeps Metallicize 8 and regains 8 block every turn for the rest of the
fight. The game's `ARMOR_AMT = 8` has no ascension tier, so this is identical at
A0 and A20.

Measured directly — ending turns without attacking:

```
turn  metal  block          turn  metal  block   (attacking instead)
   0      8      8             0      8      8
   1      8      8             1      8      8
   2      8      8             2      0      0
   7      8      8             7      0      0
```

**The bug is conditional**, which is what makes it interesting: punch through
the block before turn 3 and our engine behaves correctly. It only fires on the
"don't wake the sleeping elite" line — a legitimate real-game line — and then
charges roughly 8 damage per turn for the rest of the fight, ~64 over a typical
post-wake fight against its 111–115 HP.

Lagavulin is an Act 1 elite, so this is in play in essentially every run at both
ascensions. **Hypothesis worth testing, not established**: this may feed the
elite-avoidance weakness documented below (22% Act 1 elite capture, ELITE
routing logit −2.55, unmoved by 10x the labels). If our Lagavulin is genuinely
tankier than the real one, the labels have been correctly learning to avoid an
elite that is harder than the real game's.

### 2. Champ Gloat strength

The game: `STR_AMT = 2`, `A_4_STR_AMT = 3`, `A_19_STR_AMT = 4`, confirmed in the
constructor. Ours uses `{3, 4, 5}` — wrong at every ascension, including **5 vs
4 at A20 and 3 vs 2 at A0**. Act 2 boss, which is where runs die (mean floor
~26). One-line fix in `MonsterSpecific.cpp`'s `THE_CHAMP_GLOAT`.

### 3. Darkling Chomp hit count — FIXED

The game has `CHOMP_AMT = 2`; ours called `attackPlayerHelper(bc, asc2 ? 9 : 8)`
with one hit, so our Darkling Chomp dealt half its real damage — 8 vs 16 at A0,
9 vs 18 at A20. Now passes `2` as the hit count. Act 3, so it bites mainly at
A0, where the agent reaches floor ~39 and wins 13/100.

### 4. Writhing Mass flail block — FIXED

Ours gave `asc2 ? 18 : 16` block. The game constructs the flail's
`GainBlockAction` from the *same* `DamageInfo` as its attack, so block equals
damage: **15 base, 16 at A2+**. There is no `18` anywhere in the class — ours
was invented. Now `asc2 ? 16 : 15`, matching the flail's own damage. Act 3.

Both were independently confirmed twice: against `desktop-1.0.jar` directly, and
against `silverbot-reference`, which carries the same corrections with source
citations (`// Java CHOMP_AMT=2 hits`, and a note that the flail's block comes
from `DamageInfo.base`).

### 5. Two card energy costs were wrong — FIXED 2026-07-31

`getEnergyCost` ends in `default: return 1`, so a card nobody added to the
switch silently costs 1, and a *wrong* entry looks exactly like an absent one.
That has bitten before — the switch carries a comment about Silent cards that
"fell through to this switch's `default: return 1`". Checked all 370 costs
against the game: **351 identical, 2 wrong**, both now corrected.

| Card | Game | Was | Now |
|---|---|---|---|
| Beta | (2, **1**) | (2, 2) | (2, 1) |
| Safety | (**1**, 1) | (0, 0) | (1, 1) |

Both live in the game's `tempCards/` package rather than a character folder,
which is why a sweep organised by colour missed them. **Fixing them cannot move
a training or evaluation number**: Safety is generated only by Deceive Reality
and Beta only by Alpha, both Watcher (PURPLE), and neither appears anywhere in
`CardPools.h`, so an Ironclad can never hold either. **All Ironclad and
colorless costs are correct.**

Not counted as differences, but worth knowing:

- **Curse cost sentinel.** The game marks unplayable cards `-2`; we use `-3`
  (`Cards.h:1119`, the only such return). Both are negative so every
  `energy >= cost` test fails identically, and nothing in our engine compares
  against the literal. Only worth revisiting if Blue Candle or Medical Kit —
  which make curses and status cards playable — are ever wired up.
- **Blood for Blood** looked wrong at first and is not. Its `upgrade()` has two
  `upgradeBaseCost` branches (its cost falls as you lose HP); ours matches the
  correct one.
- The three Wish options (Become Almighty, Fame and Fortune, Live Forever) are
  marked unplayable in the game and default to 1 in ours. They are applied by
  `chooseWishCard` and never enter a deck, so this is inert.

### Not bugs, checked and cleared

Collector's block tiers are gated at A9 in the game and A4 in ours, but both
resolve to 23 at A20 and 15 at A0, so it is **A0/A20-neutral** and not worth
touching. Giant Head, Corrupt Heart, Nemesis and Repulsor are all correct in
ours — in several of those `silverbot-reference`'s implementation is the thinner
one, so it is not a uniformly better reference.

## Fixed 2026-07-31: seven obtainable relics did nothing

Relics were the last unexamined surface, and the least safe one to leave — the
CMA-ES work measured relics as worth **+0.406 win rate** at the encounter level,
an order of magnitude more than any other combat lever tested
([03-combat-search.md](03-combat-search.md)). Harness:
`lightspeed/_relic_audit.py`.

**Eight relics sat in a live pool and were never read by any behaviour code.**
They were obtainable, displayed and saved, and had no effect. **Seven are now
wired; the eighth is inert on purpose.** The audit's inert count reads 8 -> 1.

Two are Ironclad-reachable and so are *not* training-neutral — they belong with
the monster fixes behind one rebuild and one fresh baseline:

| Relic | Pool | Real effect | Ironclad? |
|---|---|---|---|
| **Toy Ornithopter** | common | heal 5 HP whenever you drink a potion | **yes — FIXED** |
| **Dolly's Mirror** | shop | on pickup, add a **stat-equivalent copy** of a chosen card to the deck | **yes — FIXED** |
| Frozen Eye | shop | view the draw pile in order | yes, but see below |

The other five are single-character and therefore training-neutral. All are now
wired, and in every case the hook already existed — only the relic check was
missing:

| Relic | Character | Effect | Where it went |
|---|---|---|---|
| Tingsha | Silent | 3 damage to a random enemy on discard | `BattleContext::onManualDiscard` |
| Tough Bandages | Silent | 3 Block on discard | same |
| Hovering Kite | Silent | 1 energy on the **first** discard each turn | same, keyed on `cardsDiscardedThisTurn == 1` |
| Snecko Skull | Silent | poison the player applies gets +1 | the `debuffEnemy` template — resolves its own `todo poison and snake skull` |
| Gold-Plated Cables | Defect | rightmost orb triggers its passive twice | `Player::triggerOrbPassives` |

Two details worth keeping. Hovering Kite needs no per-turn reset hook of its own
because every one of the five `onManualDiscard` call sites increments
`cardsDiscardedThisTurn` immediately beforehand, so "first discard" is exactly
`== 1`. And Snecko Skull is guarded on `!isSourceMonster`, so it boosts only
poison the *player* applies — every card path plus Envenom — and not a monster
poisoning the player.

Toy Ornithopter is drawn most often: it is a **common** relic in the shared
pool, and the search actively drinks potions (`potionScoreWeight` is tuned to
29.7), so every potion drunk while holding it should return 5 HP and returns
none.

Dolly's Mirror is rarer but hits harder when it lands. `DollysMirror.onEquip`
calls `makeStatEquivalentCopy` — it **duplicates** a chosen card into the deck,
carrying its upgrade state and accumulated stats across, so it can produce a
second Ritual Dagger at its current damage or a second Searing Blow at its
current upgrade count. A duplicated card is a larger deck change than an
upgrade, which makes this the more consequential of the two when it occurs.

Frozen Eye is arguably *correctly* inert — its effect is to let a human see draw
order, and the simulator already knows it. Left alone deliberately rather than
by accident.

**The fixes.** Toy Ornithopter queues `Actions::HealPlayer(5)` at the top of
`BattleContext::drinkPotion`, a flat 5 not doubled by Sacred Bark (which scales
potion potency, not relic effects); discarding a potion routes to
`discardPotion()` and correctly does not trigger it. Dolly's Mirror calls
`openCardSelectScreen(CardSelectScreenType::DUPLICATE, 1)` on pickup — that
screen already existed for the Duplicator event and resolves as
`deck.obtain(*this, c.card, 1)`, copying the Card itself and so carrying upgrade
state and accumulated stats across, which is what `makeStatEquivalentCopy` does.

**Unlike the Beta/Safety cost fixes, these are NOT training-neutral.** Both
relics are Ironclad-reachable, so both change Ironclad behaviour and every
number measured before them is on a different game. They belong in the same
batch as the four monster fixes, behind one rebuild and one fresh baseline.

**Relic constants otherwise check out**: 64 of 76 comparable relics have every
game constant present in our code. All 12 flags were read and none is a defect —
five are the same systematic thing, **our `bc.turn` is 0-indexed while the
game's constants are 1-based**:

| Relic | Game | Ours |
|---|---|---|
| Captain's Wheel | `TURN_ACTIVATION=3` | `turn == 2` |
| Horn Cleat | `TURN_ACTIVATION=2` | `turn == 1` |
| Stone Calendar | `TURNS=7` | `turn == 6` |
| Pen Nib | `COUNT=10` | `data == 9`, then reset |
| Strange Spoon | `DISCARD_CHANCE=50` | `randomBoolean()` |

That offset is worth knowing before anyone "corrects" one of them to match the
constant. The remaining seven flags are overworld/event relics whose values live
in `GameContext` outside the search window.

## Open: Entropic Brew hangs the live bridge

**Found 2026-07-31, not fixed. Live-play path only — training and evaluation are
unaffected.**

Drinking Entropic Brew in a `build_battle_context()`-built state never returns.
Root cause is a two-part gap:

- `nativeSeedRng` (`bindings/slaythespire.cpp:1769`) seeds four RNGs — `aiRng`,
  `cardRandomRng`, `miscRng`, `shuffleRng` — and leaves **`potionRng`** (and
  `monsterHpRng`) default-constructed.
- Entropic Brew is the only potion that draws from `potionRng`
  (`BattleContext.cpp:3733` → `returnRandomPotion`), and
  `returnRandomPotionOfRarity` (`src/game/Game.cpp:309`) loops **unbounded**
  until it draws a potion of the required rarity. Its own comment reads
  "this is dumb." With a degenerate RNG the loop has no exit.

**Scope.** Real `GameContext` battles are fine — verified, 39/39 Entropic Brew
drinks completed — because the game seeds `potionRng` properly. The bridge
constructor is the one that doesn't, and `sts/bridge/native_recommend.py` uses
exactly that constructor. The search does drink potions (`potionScoreWeight` is
tuned to 29.7), so a live player holding an Entropic Brew can hang the bridge
subprocess.

**It is intermittent**, which makes it worse to diagnose: the same potion
completed normally in a probe with a different draw pile, because pile contents
change how much RNG the constructor consumes before the draw. Presentation is a
frozen bridge with no error — the same shape as the SCRY segfault, which
"presents as a hung or vanished job, not an error".

**Fix, not applied**: seed `potionRng` and `monsterHpRng` in `nativeSeedRng`,
which is the actual gap. Bounding the loop in `returnRandomPotionOfRarity` is
worth doing alongside it — but as a throw, not a silent cap, per this project's
own rule about not swallowing unexpected states. Note that function is on the
real-game path too, so any change there must not perturb the RNG sequence in
cases that currently terminate.

`lightspeed/_class_card_audit.py` **skips** this potion rather than reporting it,
because a hang would take the harness down with it.

## Open: Stance Potion deadlocks the battle (Watcher only)

**Found 2026-07-31, not fixed.** The last surviving instance of the SCRY bug
class, and the worst-behaved of them.

`Potion::STANCE_POTION` does `addToBot(Actions::SetState(InputState::CHOOSE_STANCE_ACTION))`
(`BattleContext.cpp:3849`) — and **nothing anywhere resolves that state**. There
is no chooser function in `src/`, `include/` or `bindings/`; grep finds exactly
two references to `CHOOSE_STANCE_ACTION` in the whole tree, the `SetState` above
and a stale comment. `sts::py::getLegalActions` has no case for it either, so it
returns an empty vector while `outcome` is still `UNDECIDED`: the battle is
stuck with no legal action.

Measured by drinking all 42 potions in a built battle
(`lightspeed/_class_card_audit.py`, potion pass). Every other potion resolves to
`PLAYER_NORMAL` or a `CARD_SELECT` with a non-empty enumeration. Stance Potion
alone gives `CHOOSE_STANCE_ACTION` with `legal=0`.

This is a harder fix than the card-select enumerations were. Those already had
`Action::execute` and `isValidMultiCardSelectAction` support and only lacked
enumeration; here the mechanic was never implemented at all, so it needs a
chooser, a validation case, an execute case, and a decision about how to encode
a stance choice in the packed 32-bit `Action` (there is no natural `ActionType`
for it, and adding one touches the `FOREACH_ACTIONTYPE` codegen).

**Not reachable on the Ironclad path.** `PotionPool::getPotionForClass` is
correctly class-partitioned — verified, 33 potions each, with Stance Potion
exclusive to the Watcher alongside Ambrosia and Bottled Miracle. So this blocks
Watcher work and nothing else.

## Verified 2026-08-01: v31's auxiliary heads are trained and two of them work

They are trained: `run_label_quality_v31.py` passes
`trajectory_auxiliary_targets` and `--scope all-v27`, which is the only scope
putting `auxiliary_heads.` in the trainable prefix list. That the gradient
reached them says nothing about whether they learned anything, so they were
scored against the SAME target definitions
`generate_whole_run_rollouts.attach_episode_auxiliary_targets` produces, on 60
fresh A20 episodes / 4,617 on-policy decisions:

| head | metric | value | baseline | note |
|---|---|---:|---:|---|
| `next_combat_survival` | AUC | **0.817** | 0.50 | base rate 0.912 |
| `next_rest_reach` | AUC | **0.834** | 0.50 | easiest — largely readable off the map |
| `next_combat_hp` | R² | **0.302** | 0.0 | pearson +0.560 |
| `terminal_floor` | R² | **0.025** | 0.0 | pearson +0.383 |

**The combat-outcome pair works.** `next_combat_survival` and `next_combat_hp`
are exactly what a run-level planner needs to price a route without simulating
the fights: whether the next combat is survived, and at what HP fraction. Those
are the expensive part of an overworld rollout (0.54 s/episode is almost all
combat), so a planner backed by these heads costs milliseconds instead.

**`terminal_floor` is not usable as a leaf value.** Its correlation is real
(+0.383) but its R² is ~0, because it is badly under-dispersed — predicted sd
0.045 against actual 0.125. It emits something close to the training marginal
everywhere, which is the same marginal-collapse pathology recorded for the
campfire policy head below. Ranking information survives; calibration does not.
Use `run_critic.py`'s value head instead (val R² +0.3208) for anything needing an
absolute estimate.

Every head is under-dispersed to some degree (`next_combat_hp` predicts sd 0.185
against an actual 0.313), so a planner multiplying survival probabilities along a
path should expect compressed differences between routes and may need
recalibration. The survival head is the exception worth noting: its spread is
large in logit space (sd 1.84), so it does discriminate rather than emitting a
constant.

Measured on-policy under v31 itself, which is the right distribution for planning
on behalf of v31 — unlike the human archive, whose distribution mismatch is what
refuted imitation.

## Verified 2026-07-31: relic and potion pools are per-character correct

Potions: the four pools are 33 entries each and the class-exclusive sets match
the real game — Ironclad gets Blood Potion / Elixir / Heart of Iron, Silent gets
Cunning Potion / Ghost in a Jar / Poison Potion, Defect gets Essence of Darkness
/ Focus Potion / Potion of Capacity, Watcher gets Ambrosia / Bottled Miracle /
Stance Potion.

Relics: over 5 walked runs per character, each received its correct starter
(Burning Blood, Ring of the Snake, Cracked Core, Pure Water) and otherwise only
shared relics. No cross-character leakage — in particular no Watcher relic on an
Ironclad, which is what put Melange on an Ironclad and caused the SCRY segfault
(that was `lightspeed/relics.py`'s synthetic pool, not the engine's). Small
sample: A20 runs die early, so only 4–6 distinct relics per character were seen.
Suggestive rather than conclusive.

## Open: unplayable status and curse cards can be played, for free

**Found 2026-08-01. Affects training and evaluation, not just the live bridge.**

`Void` is unplayable in the real game. In this engine it is a legal action, and
playing it **removes the card from hand and costs nothing**:

```
played VOID:
  hand   ['STRIKE_RED', 'BLOODLETTING', 'APPARITION', 'VOID', 'APPARITION']
      -> ['STRIKE_RED', 'BLOODLETTING', 'APPARITION', 'APPARITION']
  energy 2 -> 2   hp 44 -> 44
```

That is a **free discard of a dead card** — strictly beneficial, and a move the
game never permits. The search finds it: in a 32,728-decision collection over the
train split, `WOUND` was offered as a playable card 562 times, `BURN` 201,
`DAZED` 84, `VOID` 17. An earlier pick-rate pass measured the search *choosing*
these roughly 30% of the time they were available, which is what a free
hand-cleanup should look like.

**The cause is the cost sentinel, and this file's previous claim about it was
wrong.** The entry below on card energy costs states: "Both are negative so every
`energy >= cost` test fails identically, and nothing in our engine compares
against the literal." Both halves fail. The card carries `cost = -2` and
`cost_for_turn = -2`, not the −3 that entry describes, and `energy >= -2` is
**true** for any non-negative energy — so the playability test passes rather than
failing. Unplayable cards are therefore playable whenever a cost comparison is
the only gate.

`Slimed` is *not* part of this: it is genuinely playable (cost 1, "Exhaust"), and
was a false positive on first inspection.

**Scope.** Any fight whose deck contains Wound, Burn, Dazed or Void — which in
the human benchmark is ~2.6% of decisions offering at least one. It is the same
class of defect as draw-order clairvoyance: the search exploits an advantage the
real game does not grant, so every number measured on a status-bearing deck is
optimistic. Unlike the clairvoyance, this one is a **correctness bug with no
upside**, and fixing it should cost nothing but the inflation it removes.

**Not fixed.** The fix is to gate playability on a non-negative cost rather than
on `energy >= cost` alone, and it changes engine behaviour, so it invalidates
baselines exactly as the Armaments and monster fixes did. It should land with a
fresh re-baseline in the same batch as any other behaviour change.

## Open: the combat search is draw-order clairvoyant (found 2026-07-31)

`nativeRunMctsSearch` roots its tree in a full copy of the live `BattleContext`
(`arena.newNode(BattleContext(bc))`, both entry points), and that copy includes
the **ordered draw pile**. Verified directly: dump `bc.draw_pile` before
searching and the tail of that list is exactly what END_TURN then draws. Every
simulation therefore plays against the true future draw order — the search
knows which card is coming, always.

This is not the Runic Dome intent issue below (measured <1 HP/battle). Silverbot
found this same defect in their own engine on 2026-06-03 — their log's headline
is "WE'VE BEEN CHEATING: draw-order clairvoyance ≈ +34pp" (69.4% → 35.2% honest
at 1k sims; their canonical-CardPile belief search recovered 56.2%) — and their
`SearchAgent.h` comments attribute their huge default sim budget to the honest
engine's noisier chance nodes. The engine we compare against is the honest one.

Consequences for our numbers:

- Every combat comparison run 2026-07-31 pits our clairvoyant search against
  their honest search. Our tier-deck lead (473/504 vs 463/504) and human-deck
  parity are measured with an information advantage they deliberately removed.
- "We pay 1.29x a top human's HP" is optimistic — the human cannot see draw
  order either.
- Combat looking "saturated" at ~100 sims is partly a clairvoyance artifact:
  perfect draw information collapses most chance-node variance, so few sims
  suffice. An honest search would be weaker AND hungrier for simulations.

What survives it: the 2026-07-31 layer-swap result (see
[03-combat-search.md](03-combat-search.md)) — both arms share our combat, so the
clairvoyance cancels, and the +15.71-floor overworld-policy attribution stands.

### Measured 2026-07-31: the advantage is worth 3.78 HP/fight

`lightspeed/_clairvoyance_cost.py` permutes the draw pile's ORDER before every
search — identical contents, identical everything else — so the only variable
removed is knowledge of what comes next. On 250 test fights at 100 sims:

| | objective | deaths |
|---|---|---|
| clairvoyant (current) | −8.472 | 39/250 |
| blind draw order | −12.252 | 48/250 |

**−3.78 ± 0.84 HP (t = −4.50)**, with deaths up 23%. For scale that is 75% of
the entire +4.99 HP the day's config tuning bought.

**This overturns "our combat beats Silverbot".** That comparison ran our
clairvoyant search against their honest one. Blind, we score −12.25 against their
−6.58: their honest engine is comfortably better than our honest engine, and the
apparent parity was the cheat. Every ours-vs-theirs combat number from 2026-07-31
carries this caveat.

The figure bounds the fix's cost from two sides: a real belief search averages
over orders in-tree (Silverbot measured that as +21pp over committing to one
sampled order), which recovers part of the loss; but this harness resamples a
fresh wrong order every decision, which is harsher than a live bridge holding one
arbitrary order.

**Live play is where this actually bites.** `build_battle_context` takes
`draw_pile_cards` as externally reported state, and CommunicationMod does not
expose the shuffle order — so the bridge hands the search an arbitrary order
which it then trusts completely. Every number in
[09-live-play-bridge.md](09-live-play-bridge.md) assumes a search that knows the
future, and real live degradation is at least this 3.78 HP/fight.

Fix direction, per Silverbot's own migration: canonical CardPile belief search
(in-tree belief averaging beat committing to one sampled order per decision by
+21pp in their A/B). This is a significant engine change, and every combat
number measured before it lands will need re-baselining afterwards — same class
of invalidation as the Armaments fix.

### Implemented 2026-08-01 behind `honest_draw_order` (default off)

The mechanism is more specific than "the root copy carries the order".
`nativeDpwChanceChild` already treats END_TURN as a chance node with DPW
widening; what leaked was **every other draw**. Drawing from an ordered pile
consumes no RNG, so `nativeRngCounterSum` — the probe deciding whether an action
is stochastic — sees zero consumption for Battle Trance, Pommel Strike, Shrug It
Off, Acrobatics, Warcry and friends, classifies them deterministic, caches one
child, and never samples another order.

`honest_draw_order` permutes order (never contents) at the three points the
future actually enters: each DPW chance sample, seeded from that sample's own CRN
seed; each rollout, on its own stream so toggling this cannot shift the Gumbel
sequence; and any in-tree action that drew, which is then reclassified as a
chance node so widening samples several orders. That is the in-tree averaging
shape (Silverbot's 56.2%), not one determinization per decision (their 35.2%).

Verified: 250 val fights byte-identical at 0.0, 79/120 change at 1.0, and two
runs at 1.0 match exactly — the reproducibility check the Gumbel-RNG incident
above makes mandatory for any new sampling stream.

**Price: −4.78 ± 0.71 HP (t = −6.69), deaths 76 → 109**, 500 paired train fights
at 100 sims. Consistent with `_clairvoyance_cost.py`'s −3.78, slightly larger as
expected since this removes the information in more places.

**The simulation curve does NOT clearly un-flatten.** 300 paired train fights,
paired within each condition:

| | 100 sims | 300 sims | 900 sims |
|---|---:|---:|---:|
| clairvoyant | −4.490 (37 deaths) | −3.063, **+1.43 ± 0.76** | −2.307, **+2.18 ± 0.82** |
| honest | −9.593 (57 deaths) | −7.457, **+2.14 ± 0.96** | −9.820, **−0.23 ± 1.19** |

- **The clairvoyant curve was never flat**, corroborating the retraction recorded
  in [03](03-combat-search.md): 100 → 900 buys +2.18 ± 0.82 (t = 2.66), same
  direction as the +3.31 HP measured for 100 → 1600. "Budget is not a lever"
  rests on the FLOOR measurements, not on this one.
- The honest arm's non-monotonic 900-sim point above (−0.23, deaths back to 59)
  **was noise and did not replicate.** Re-run on `train[300:600]`, disjoint from
  the set above, it is monotone: +0.44 ± 0.75 at 300 and +0.75 ± 0.79 at 900.

### Replicated on a disjoint set, and the result reverses the hypothesis

Both arms, `train[300:600]`, 300 fights, paired within condition:

| | 100 sims | 300 sims | 900 sims |
|---|---:|---:|---:|
| clairvoyant | −6.950 (55 deaths) | −4.683, **+2.27 ± 0.70** (t = 3.22) | −3.477, **+3.47 ± 0.74** (t = 4.71) |
| honest | −9.890 (65 deaths) | −9.453, **+0.44 ± 0.75** (t = 0.58) | −9.143, **+0.75 ± 0.79** (t = 0.94) |

**The clairvoyant search converts budget into HP; the honest one barely does.**
That is the opposite of the reasoning that motivated building this. The
expectation — stated in `honestDrawOrder`'s own comment, and taken from
Silverbot's large default simulation budget — was that removing draw knowledge
would restore chance-node variance and make the search hungrier for simulations,
reviving the search-research axis that measured flat all day. It does the
reverse: 100 → 900 buys **+3.47 clairvoyant against +0.75 honest**, on identical
fights and identical seeds, and the honest arm has the worse baseline (−9.89 vs
−6.95) so it had more room to gain, not less.

The plausible mechanism is the one previously invoked for the blip, which
survives as an explanation for the flat response even though the blip itself was
noise: reclassifying draws as chance nodes greatly enlarges the chance-node
population, and `wc_chance`/`wa_chance` (3.66 / 0.667) were tuned when END_TURN
was the *only* chance node. Extra simulations may therefore be spent widening
across draw orders rather than deepening lines. **Untested.** Re-tuning those two
parameters under honest draws is the experiment that would settle it, and nothing
about the honest arm's budget response should be quoted as a property of honest
search in general until it is run.

The gap also does not close with compute — 2.94 / 4.77 / 5.67 HP at 100 / 300 /
900 on this set — so the information cannot be bought back with search at these
budgets.

**Net: the case for turning this on is honest measurement and a trustworthy live
bridge. It is not a floors play, and the "honest search un-flattens the budget
curve, so the shelved search ideas deserve a second look" argument is refuted.**

**Priority note.** Fixing this would cost floors, not gain them: combat is
saturated above ~100 sims for floor performance ([03](03-combat-search.md)) while
being steep below it, and an honest engine is both weaker and noisier (which is
why Silverbot's default budget is 50,000 sims). Canonical CardPile buys honesty
and a trustworthy live bridge; it does not buy floors, and the binding constraint
is the run policy.

## Open: Runic Dome is clairvoyance, not phantom moves (tested 2026-07-30)

`RUNIC_DOME` appears in the engine in exactly three places, and none of them
hides anything: `BattleContext.cpp:218` grants +1 energy per turn,
`ExpertKnowledge.cpp:205` ranks it for boss-relic drafting, and
`SimpleAgent.cpp:158` — a legacy agent that is **not** on the search path —
substitutes a flat `5 × act` damage estimate when Dome is held. The bindings'
own `nativePredictedIncomingDamage`, which is what the search uses, has no Dome
branch at all.

Measured on a Time Eater state built twice, identical except for the relic:

| | no Dome | Dome |
|---|---|---|
| `move_history` | `[189, 0]` | `[189, 0]` |
| `get_monster_move_damage(0)` | `(0, 0)` | `(0, 0)` |
| `leaf_features()` | `[60, 0, 3, 0, 0, 0, 456, 0, 1, 1]` | identical |

**So the search sees the true intent under Runic Dome.** That is a real
divergence from the game — a human under Dome cannot see intents — but it is
*clairvoyance*, not the phantom-move class of bug. Silverbot closed the same
clairvoyance in June 2026 by deferring monster move rolls while intents are
hidden.

Note the *damage* prediction itself is correct:
`nativePredictedIncomingDamage` applies the engine's real resolution (Strength
added, monster Weak ×0.75, player Vulnerable ×1.5), not the raw
`getMoveBaseDamage` table lookup. Earlier documentation claiming otherwise was
wrong.

### Why fixing it is not worth doing (measured 2026-07-30)

Removing the clairvoyance takes information *away* from the search, so it makes
the agent slightly **worse**, not better. It is a fidelity fix, and its value
scales with how often the relic is actually held. Measured over 100 A20 seeds
with v31:

| | |
|---|---|
| Runs acquiring any boss relic | 68/100 |
| Runs holding **Runic Dome** | **3/100** |

Dome sits mid-pack among boss relics, at the same 3% as Velvet Choker, Pandora's
Box, Ectoplasm, Astrolabe, Calling Bell and Coffee Dripper; Busted Crown and
Slaver's Collar lead at 6%. So the clairvoyance is **not** causing the policy to
over-draft Dome — the boss-relic distribution is nearly flat, which is its own
finding (the same marginal-collapse pattern seen at campfires).

An effect confined to 3% of runs, in the direction of making them harder, is far
below the ±0.55 noise of a 200-seed paired eval. Reproduce with
`python -m lightspeed._relic_uptake 100`. **Leave it**, unless live play
becomes the priority — and note that even then it changes nothing, for the reason
in the next entry.

## Open (live bridge): telegraphed monster intents are discarded

`native_recommend.py:359` never sets `NativeMonsterSpec.move_name`, for any
monster, in any fight. `build_battle_context` therefore falls back to rolling a
plausible move from this engine's own AI model, so **every live recommendation is
computed against a guessed monster intent rather than the one the game is
actually showing.**

The stated reason is that CommunicationMod's `move_id` is a per-monster-class
numeric id with no name in the protocol. That is true, but the data to resolve it
is already present. Sampled from this project's own capture,
`slay-sim/sts_raw_states.log` (4,734 monster records):

| Field | Present |
|---|---|
| `intent`, `move_id`, `move_base_damage`, `move_adjusted_damage`, `move_hits` | 4,734 / 4,734 |
| `last_move_id` | 2,395 |
| `second_last_move_id` | 1,002 |

`intent` is `UNKNOWN` on only **72 of 4,734 records (1.5%)** — which is the
genuinely hidden case, i.e. Runic Dome. The other 98.5% of the time the game is
telling us the move and we throw it away.

The mapping is also small: that capture holds **21 distinct
`(monster, move_id)` pairs across 12 monsters**, and each is uniquely
fingerprinted by `(intent, move_base_damage, move_hits)` — e.g. Snecko's
`move_id=2 / ATTACK / 18 / 1` versus `move_id=3 / ATTACK_DEBUFF / 10 / 1`. A
lookup table keyed on that tuple is mechanical to derive against the engine's own
`MonsterMoveDamage` tables.

**Measured: the guess is wrong 87.5% of the time**
(`lightspeed/_bridge_intent_audit.py`). Replaying 1,000
single-monster states out of the capture, building each the way the bridge does
(no `move_name`) and comparing the engine's rolled move against the damage the
game was telegraphing in that same state:

| | |
|---|---|
| Rolled move matched the telegraphed one | **125 / 1000 = 12.5%** |
| Rolled move was wrong | **875 / 1000 = 87.5%** |

The errors are large and directional, not rounding:

| Monster | Telegraphed | Guessed |
|---|---|---|
| Book of Stabbing | 24 × 1 | 7 × 1 |
| Snecko | 18 × 1 | **0 × 0** |
| Snake Plant | 8 × 3 | **0 × 0** |
| Blue Slaver | 8 × 1 | 13 × 1 |
| Taskmaster | 9 × 1 | 7 × 1 |

The `0 × 0` cases are the damaging ones. With predicted incoming damage at zero,
`dangerFraction` is zero, so the SKILL branch scores at its floor *and*
`blockSufficient` is trivially true, so `defensiveCardSuppressionPenalty` fires
against every defensive card. The search will recommend attacking into a 24-point
hit while actively suppressing the Defend in hand.

Note the inversion this creates: `predict.py`, the *fallback* damage-only layer,
reads the telegraphed intents correctly. The primary recommendation layer, which
supersedes it, does not — so the bridge's cheap fallback is better informed about
incoming damage than its expensive search.

Caveat on the measurement: it covers single-monster states from one capture
session (largely Act 2 — Slavers, Snecko, Book of Stabbing), and passes empty
piles, since the move roll does not depend on the player's deck. The 12.5% is
about what a coincidental match would give, which is the point: the bridge has no
intent information at all.

**This is the Dome question's real answer.** On the live path there are no true
intents to leak — the bridge never had them — so fixing the search-side
clairvoyance would change nothing there. What would change live play is using the
intents the protocol already provides.

## Fixed: segfault on `InputState::SCRY` (engine rebuilt 22:59)

**Status: fixed 2026-07-30.** Two defects, one crash.

**Root cause.** `sts::py::getLegalActions` (`bindings/bindings-util.cpp:420`)
handled exactly two input states, `CARD_SELECT` and `PLAYER_NORMAL`. Anything
else fell through and returned an **empty** action vector. For
`InputState::SCRY` that meant every caller assuming at least one legal action
broke — `nativeHeuristicPick`'s `return legal[0]`
(`bindings/slaythespire.cpp:1277`) segfaulted mid-rollout, reached via
`nativeHeuristicPickFast`'s non-`PLAYER_NORMAL` fallback. The tree was equally
blind: `nativeExpandLeaf` (`:2022`) and `nativeRunMctsSearch` (`:2257`) both
assign `node->actions` from this function, so a Scry node got **zero edges** and
could not be searched at all. `BattleScumSearcher2.cpp:189` had handled SCRY
correctly all along — only the bindings' enumeration was missing it.

Confirmed under gdb against a debug build (`-UNDEBUG -fno-lto -Wa,-mbig-obj`),
which named the frame directly: `legal=std::vector of length 0, capacity 0`,
`sim.inputState = sts::InputState::SCRY`.

**Fix.** Enumerate the scry subsets as a bitmask, identical to
`BattleScumSearcher2`'s, so the two searchers agree on what a Scry decision is.
Plus an `sts_asserts`-guarded assert at the `legal[0]` site, so a future missing
enumeration fails with a message instead of a segfault.

**Why an Ironclad was scrying at all.** `lightspeed/relics.py`'s pool contained
13 character-specific relics from the Silent, Defect and Watcher pools that an
Ironclad can never obtain. `MELANGE` — "whenever you shuffle your draw pile,
Scry 3" — fires constantly on an Ironclad deck. `_EXCLUDED_NAMES` had been
assembled per-mechanic and missed these; the pool is now 109 relics, down from
122.

**Reproducer, now passing**: `COLLECTOR`, seed `6600013`, A20, act2/boss
resources, `relic_generator=weighted_ironclad_relics`. The full 360-fight sweep
that previously died at fight 134 completes clean.

**Note for anyone reading old logs**: a segfaulting worker presents as a hung or
vanished job, not an error — six pool workers sat at near-zero CPU for 13
minutes before this was diagnosed. Any past training stall on a relic-enabled
script is a candidate.

## Not ours: Time Eater phantom moves on select-openings (tested 2026-07-30)

This was mis-attributed. The entry lives in
`silverbot-reference/CORRECTNESS_ISSUES.md` and describes a bug in **Silverbot's
`pbc` shadow-replay bridge**: because their engine defers monster move rolls
while Dome hides intents, a Time Warp trigger that lands on a play parked at a
card select cannot defer, and materializes guessed moves.

Our fork has no deferral machinery, because it never hides intents in the first
place — so there is nothing to materialize. Tested directly on the exact trigger
condition (Time Eater, `TIME_WARP` at 11, Armaments as the 12th play, which parks
at `SINGLE_CARD_SELECT`), with and without Runic Dome:

| | Time Warp 10 (no trigger) | Time Warp 11 (triggers) |
|---|---|---|
| after Armaments | parked at `SINGLE_CARD_SELECT`, counter unchanged | parked at `SINGLE_CARD_SELECT`, counter unchanged |
| after resolving the select | counter 10 → 11, turn 1 | counter → 0, monster +2 Strength, turn 1 → 2 |
| Dome vs no Dome | traces identical | traces identical |

The trigger fires correctly through the parked select, and Dome changes nothing.
No phantom, no crash. **Remove this from our backlog** — the residual Dome issue
we do have is the clairvoyance above.

## Mostly fixed 2026-07-31: half the search's scoring terms were switched off

Thirteen `TunableParams` scoring terms sat at 0.0 in `tuned_search_params.json`,
existing in the code and contributing nothing. Eight are now on, tuned against
the human benchmark (`tune_search_human.py`): both Power-value terms,
`vulnerableApplyBonus`, `weakApplyBonus`, `energyWasteWeight`,
`enemyBlockWeight`, `directBlockScoreWeight`, and `rolloutTemperature` (2.199).
The applied config is worth **+4.99 ± 0.69 HP (t = 7.27)** over its predecessor on
528 held-out fights, both measured on the current engine.

Still off: `silentPoisonApplyBonus` (Silent-only), `policyNetWeight`
(no net exists on disk), and the ten `vf*` weights (live only in `leaf_eval_mode`
value/truncated — measured a dead end, see [03](03-combat-search.md)).

Corrected 2026-07-31: `attackDamageScoreWeight`, `selfDamageScorePenalty`,
`blockWeight` and `winHpFractionWeight` were listed here as off and are all four
tuned on in the shipped `tuned_search_params.json` (42 overrides, 16:31). See
[03](03-combat-search.md) for the values and for the test that caught it.

A sixth term was found switched off in a different sense on the same day and is
now tunable: the rollout scored **every non-CARD action a flat 5.0**, so drinking
a potion, discarding one and ending the turn were indistinguishable to it
(`nativeScoreAction`'s opening branch). `potionScoreWeight` prices potions in the
terminal evaluation only, which the rollout never reaches. Five parameters now
cover that branch — `rolloutPotionBase`, `rolloutNonCardBase`,
`rolloutPotionDangerScale`, `rolloutPotionFinishOffScale`,
`rolloutPotionDiscardPenalty` — defaulting to a verified no-op (250 val fights
byte-identical across the rebuild), and now in `tune_search_human.py`'s space
(47 parameters). All three fire: forcing each one changes 42–48 of 120 fights.

**They are untuned, and the one hand-picked setting measures null.** A discard
penalty of 50 — enough to put discarding below every drink, which is the one
setting with a prior worth stating in advance, since there is no slot pressure
inside a fight — scores **−0.58 ± 0.56 HP (t = −1.04)** against the shipped
config on 500 paired train fights, with deaths 76 → 82 and 226 of 500 fights
playing differently. So the rollout does discard potions constantly and it does
not measurably cost anything. Worth recording as a near-miss in method: the same
arm looked like **+1.0 HP** on a 120-fight val spot-check, which is precisely the
best-of-three selection high this file warns about elsewhere. Sweep on train;
let the tuner find the values.

**Turning one of them on exposed a reproducibility bug that had been latent for
as long as the parameter was zero**, which is the part worth remembering.

## Fixed 2026-07-31: the rollout's sampling RNG ignored the search seed

`nativeGumbelNoise` held:

```cpp
static thread_local std::mt19937_64 gumbelRng(std::random_device{}());
```

The rollout policy's Gumbel-max sampling drew from an RNG seeded outside every
reproducibility guarantee the file otherwise makes — not from `search_seed`, not
from the CRN base. It was unreachable while `rolloutTemperature` was 0.0, because
argmax never samples, so it sat dormant from the day it was written.

The moment tuning moved that parameter off zero, identical `BattleContext` plus
identical `search_seed` began returning different actions. **Every "paired"
comparison silently lost its common random numbers**, and a repeated 200-fight
evaluation of one fixed config moved 0.49 HP with 97 of 200 fights differing.

Now seeded per search call from that call's seed, SplitMix64-mixed because
`play`-style callers derive highly correlated seeds (`(i << 20) ^ crc32(fight)`)
and mt19937_64 seeded with near-identical small values produces visibly
correlated early output. Verified: eight searches at one seed identical, a
528-fight evaluation byte-identical across runs.

No prior conclusion was invalidated — every standard error quoted was computed
empirically from observed paired differences, so the noise was already inside it.
But comparisons made before the fix are wider than they needed to be.

## Trap: `gc.deck` returns a copy, so mutating it silently does nothing

Not a bug, but it cost a full wrong result before being caught, and anything that
builds a `GameContext` by hand will hit it.

```python
gc.deck.clear()          # no effect
gc.deck.append(card)     # no effect
```

pybind returns the underlying vector by value. The real mutators are
`gc.obtain_card(card)` and `gc.remove_card(idx)`, and both preserve upgrade
state. The first version of `_human_deck_combat.py` used `clear()`/`append()`,
so 40 fights that were supposed to use a human's reconstructed deck were played
with the default starter deck — and the resulting 14/40 win rate was reported
before the harness was checked. It now raises on a deck-size mismatch after
rebuilding.

## Engine rebuilt 2026-07-31 14:36: numbers from before it are on a different game

The production `sts_lightspeed/build/` predated the day's engine edits (its first
rebuild recompiled every translation unit when only `bindings/slaythespire.cpp`
had been touched). Eight changes landed, six of which alter Ironclad behaviour:
Lagavulin, Champ, Darkling, Writhing Mass, Toy Ornithopter, Dolly's Mirror.

**Every floor count and model evaluation on disk predates this**, including
v37's 23.675. They are not comparable to anything measured after, exactly as with
the Armaments fix (which moved v28 from 23.70 to 21.73).

Measured, rather than assumed: on the 528-fight human benchmark the six Ironclad
changes are worth **0.04 HP** net — the pre-tuning config scores −11.831 on the
new engine against −11.79 on the old. They largely offset (Lagavulin and Champ
easier, Darkling harder, Writhing Mass easier, both relics helpful), and each
touches a small slice of the fights. So combat-benchmark numbers spanning the
rebuild happen to be safe; **run-level numbers are not**, since a boss or Act 3
elite behaving differently compounds across a whole run.

Rebuild note: `build/CMakeCache.txt` carries `STS_PGO:STRING=use`, so a plain
`cmake --build . --target slaythespire` in that directory *is* the PGO build —
`-fprofile-use -fprofile-correction -fprofile-dir=<src>/pgo`, confirmed in
`flags.make`, against 31 `.gcda` files. Do not "fix" this by reconfiguring
without `STS_PGO`. `build_classfix/` is superseded and stale as of 06:19.

## Won't fix (measured 2026-07-30): no tree reuse

`nativeRunMctsSearch` allocates a fresh `MctsArena` and a fresh transposition
table per decision, so nothing carries across decisions.

**The two prior failures were in the Python search, not this one.**
`az_search.py:796` records both: attempt 1 matched a real post-`END_TURN` state
against a cached DPW sample and hit a hard assertion on Awakened One, traced to
its `_state_key` omitting monster `move_history`/`misc_info` (Time Eater's
has-used-Haste flag, Awakened One's isPhase2). Attempt 2 added those, retried,
and segfaulted on the same fight without the cause being isolated. Both were in
`az_search.py`, which is deprecated and not on the whole-run path.

The C++ side is in better shape than that history suggests: `NativeStateKey`
already carries `moveHistory0/1` and `miscInfo`, preserves draw-pile order, and
the search already classifies actions by an RNG-consumption probe rather than by
action type — which is exactly the fix the Python attempts converged on.

**But the payoff is too small to justify it.** Rerooting can only recycle the
subtree under the action actually taken. Measured over 339 real decisions across
six encounters at 300 sims with the shipped config (`lightspeed/_reuse_ceiling.py`):

| | |
|---|---|
| Mean share of the tree under the chosen action (`N_best/N_total`) | **0.356** |
| Chosen actions that are deterministic (so a reroot could match) | **326/339 = 96.2%** |
| Mean reusable fraction | **0.342** |
| Effective budget if reuse were perfect | **1.34×** — 300 → ~402 sims |

The match rate is not the limiter; concentration is. And a 1.34× effective budget
has to be read against the measured return on budget: tripling sims 300 → 900
buys +0.96 floors for v28 and **+0.21 for v31**, the current best checkpoint. A
1.34× multiplier is a small fraction of a 3× one, on a sub-linear curve — call it
under +0.1 floors for v31, well inside the ±0.55 noise of a 200-seed paired eval.

Against that: rerooting needs the arena's ownership model reworked (or the
subtree deep-copied), needs the transposition table invalidated on every reroot
(the most likely source of the Python segfault), and needs `NativeStateKey`'s
gaps closed first — it does not key card upgrade state or the exhaust pile. That
is a lot of memory-safety-sensitive work for an effect this project cannot
measure.

**Recommendation: leave it.** Revisit only if the search budget itself ever
becomes the binding constraint, which today's evidence says it is not.

## Refuted 2026-07-31: imitating the human archive makes the policy WORSE

`lightspeed/replay_human_runs.py` recovers Baalorlord's own decisions by
converting his base-35 seed, regenerating the same map, and solving his recorded
room sequence back into the node he clicked -- 4,044 demonstrations across
routing (2,625), drafting (653), campfires (441) and smith targets (325), from 97
of 100 runs. As a **diagnostic** it is excellent and its findings stand (see the
routing table above, and the drafting comparison below). As **training data it
does not work**, measured twice:

| clone | scope | paired floors vs v31 (120 seeds) |
|---|---|---|
| routing + campfire + smith | `experts` | **-15.80 +/- 0.74** (t=-21.5), worse on 118/120 |
| drafting only (`rewards`) | `experts` | **-5.42 +/- 0.81** (t=-6.69), worse on 72/120 |

Two causes, and the second is the general one:

**Elite-taking is capability-dependent.** He takes 73% of offered elites because
his deck beats them cheaply. v31 cloned the preference (ELITE coefficient
-2.55 -> +0.28, elite capture 22% -> 75%) and then died at floor 7.08 -- exactly
where act 1 elites appear. Elite avoidance was *rational* for a policy whose
combat is more expensive than his; imitation removed a defence it needed.

**The extraction pins his deck at every floor**, which is what makes replays
survive (yield 23 -> 52 decisions/run) and simultaneously makes every observation
show HIS deck, relics and HP -- states our policy never occupies. The model
learns "given Baalorlord's deck, pick Feel No Pain" and then picks it holding a
deck with no exhaust engine. Textbook behaviour-cloning distribution mismatch,
built in deliberately. It explains both failures with one mechanism, including
why drafting failed despite being capability-INDEPENDENT in principle.

**Beware the proxies.** The routing clone had every intermediate signal pointing
the right way -- validation NLL falling monotonically over 18 epochs, argmax
agreement with the human 65% -> 69%, and the target coefficient moving exactly as
designed -- while dying at floor 7. Imitation metrics measure agreement with a
teacher whose actions are only good given the teacher's capability. **Run the
120-seed paired floor eval FIRST**; it costs 90 seconds and is the only
measurement that caught either failure.

## Refuted 2026-08-01: survival-weighted route planning costs 3.68 floors

Harnesses: `lightspeed/_route_planner.py` (the planner) and
`lightspeed/_eval_route_planner.py` (the paired eval). Both reusable; the eval
substitutes any map-decision rule for v31's while leaving the policy untouched
elsewhere, the same isolation `_routing_audit.py --randomize-paths` uses.

The idea was to treat routing as a small planning problem rather than a learning
one. An act map is a ~54-node DAG with at most 3 successors per node — the shape
the orienteering literature calls stochastic orienteering with survival
constraints — and the input that a capability-independent human demonstration
could never supply, *our own* probability of surviving a fight, is sitting in
v31's `next_combat_survival` head at AUC 0.817 (see the aux-head entry above).
Node values come from one backward pass in descending y:

    V(n) = room_value(n) + p_fight(n) * max over successors of V(s)

REST value scales with missing HP, encoding structurally the sign v31 has
backwards (hp_frac × REST +1.19 against the human's −1.93).

**Result on 120 paired A20 seeds at 100 sims, elite weight 3.0:**

| | mean floor | wins | Act 1+ elite capture |
|---|---:|---:|---:|
| v31 stock | 24.12 | 0 | 9/273 (3.3%) |
| v31 + planner | **20.43** | 0 | 224/251 (89.2%) |

**−3.68 ± 1.10 floors (t = −3.34)**, 109/120 runs differ, planner overriding
65.6% of map decisions.

Two things worth separating, because they point different directions.

**This is not the imitation collapse repeating.** Cloning Baalorlord's routing
took capture to 75% and died at floor **7.08**; the planner takes it to 89.2% and
still reaches **20.43**. Conditioning elite appetite on our own survival estimate
is evidently doing real work — it just is not doing enough of it to pay.

**The likely defect is the approximation the planner was built on, not the
framing.** `next_combat_survival` predicts P(surviving the NEXT combat) from the
CURRENT state and cannot be queried at a hypothetical node three floors ahead, so
one probability is applied homogeneously to every fight on the route. An elite
therefore carries the same modelled risk as an ordinary monster while being worth
strictly more, and the planner takes essentially all of them — 89.2% against the
~59% of offered Act 1 elites that winning runs in the 1,008,636-run study take.
The homogeneous-survival assumption is documented in `_route_planner.py`'s own
docstring as deliberate; this result is what it costs.

**Not tested, and the obvious next steps if anyone resumes this**: sweeping the
elite weight downward (3.3% → 89.2% capture is an enormous jump, and nothing was
measured in between), and giving ELITE nodes a harsher survival factor than
MONSTER nodes, which is the principled repair of the flaw above rather than a
recalibration around it. Neither was run, so **the framing is untested — only
this parameterisation of it is refuted.**

## Refuted 2026-07-31: the map representation is not the routing problem

Silverbot's own log recommends an R5b map encoding -- path option += destination
room type, map nodes += ego-relative coords + reachable flag + scaled
`(minE, maxE, distRest)` aggregates -- and credits aux heads with lifting their
routing 91.8 -> 95.5. It is natural to suspect v31's ELITE coefficient of -2.55
means it cannot SEE the structure.

It can. `whole_run_env.map_route_features` already computes per path option:
`min_elites`, `max_elites`, `rest_distance`, `elite_distance`, `shop_distance`,
`min_monsters`, `max_monsters`, alongside `action_target_rooms`,
`action_target_coords` and the raw map. Its docstring states the same intent --
"avoiding the need for the policy to learn multi-hop graph traversal merely to
compare two routes". `dest_room` aux is separately refuted (the model already
embeds `action_target_rooms`, so the head would predict its own input).

So v31 has the features silverbot credits and has still learned the wrong
preference from them. This is a learning problem, not a representation one.

## What Silverbot actually does (read 2026-07-31)

Not imitation. `rl_train.py` with pipelined collection, driven by
`run_heart1_supervised.sh`, checkpoints to **iter 2575** at ~650s per iteration
-- roughly **three weeks of continuous on-policy RL**, with LR and entropy decay
schedules (heart-kill 0.30 @1295 -> 0.367 @1862; they thinned checkpoints 47G ->
13G). Their bare agent WITHOUT the trained net reaches mean floor 14.1-17.0 at
100-2000 sims, below our v31's 22.4 -- so their advantage is the trained policy,
not their search.

Given imitation and representation are both closed, on-policy RL against our own
state distribution is the remaining path to the +15.71 floors the layer swap
proved available. Our episodes are cheap (0.8s per A20 run at 100 combat sims,
which is all combat needs), so the wall is iteration count, not per-episode cost.

## Open: two more archived runs segfault the engine during overworld replay

Runs at sorted-index 51 and 76 (`1762887633`: Runic Dome, Calling Bell, no
Prismatic Shard) reproducibly crash the engine while replaying legitimate game
states, and a third appears in indices 89-99 only after card-reward capture was
added -- suggesting that one is in the `REWARDS` path rather than the same
defect. Hard crashes, not catchable exceptions.

`replay_human_runs.py` works around them by streaming output per run and taking
`--start` to step over a crasher. That streaming is load-bearing: an earlier
buffered version had a segfault silently destroy an entire extraction while the
shell reported **exit 0** (that was `tail`'s status, not python's) and wrote no
output file at all.

## Coverage gaps

**Character.** Ironclad only on the training path. `sts.CharacterClass.IRONCLAD`
is hardcoded in five non-test files: `whole_run_env.py:257`, `env.py:822`,
`benchmark_full_runs.py:40`, `compare_tier_combat.py:126`,
`eval_heart1_hybrid.py:23`.

The C++ engine implements **every playable card of all four characters** — 75/75
Ironclad, 75/75 Silent, 75/75 Defect, 75/75 Watcher, measured 2026-07-31 by
cross-referencing `cardTypes`/`cardColors` against the three type switches in
`BattleContext.cpp` (`lightspeed/_class_card_audit.py`, dispatch pass). Orbs,
stances and per-character starting decks are all wired
(`BattleContext.cpp:66`, `GameContext.cpp:440-520`).

The claim in earlier revisions that "Defect and Watcher are largely
unimplemented" was **wrong**. What was actually missing lived in the search
bindings, not the engine — see the entry below.

Silent, Defect and Watcher remain **untuned**: `cardPickRateWeight` was learned
from ~5,500 Ironclad decisions and everything else falls back to a 0.05
smoothing floor, and `sts.CharacterClass.IRONCLAD` is still hardcoded in five
non-test files. Playable is not the same as competitive.

**Alternate agents.** `SimpleAgent.cpp`, `BattleScumSearcher2.cpp` and
`ScumSearchAgent2.cpp` have no cases for the newer `CardSelectTask::DISCARD` and
`RETAIN` tasks and will fall through their `default`, picking nothing. They are
legacy heuristic agents, not the path the training pipeline or the bridge uses.
Note that the parenthetical this entry used to carry — "both go through
`sim/search/Action.cpp`, which is fully wired" — was false when written; that is
what the next entry is about.

**Python engine choices.** `slay-sim/sts/` has no choice-resolution mechanism, so
every card whose real text involves a player choice ("choose 1 of 3", "choose a
card from your hand") falls back to a random pick. Each such factory documents
this in its own docstring. Also approximated there: Apotheosis/Transmutation do
not generically upgrade existing `Card` instances; Sadistic Nature's reactive
damage is unwired; Hand of Greed's gold has no economy layer to attach to; Dark
Shackles' Strength loss does not restore at end of turn.

## Behavioural weaknesses (not defects)

- **Elite avoidance.** Over 60 seeds the policy fights ~1 elite per run against
  3.15 available in Act 1 alone. Elites are the main relic source. Measured as a
  capture rate: **42 of 189 offered Act 1 elites, 22%**, with 24 of 60 runs
  taking none. v28 and v31 are *identical* here — same 42/189, same
  `0:24 1:30 2:6` distribution — so 10× the training labels moved Act 2 survival
  without moving Act 1 routing at all. No run has ever taken three.

  **It is not a valuation error, and it is not fixable by correcting the
  coefficient** (measured 2026-07-31, `lightspeed/_route_bias_probe.py`, v31,
  120 paired A20 seeds at 300 sims on the post-rebuild engine). The probe adds a
  plain additive bias to the policy's logits at MAP_SCREEN only — two or three
  scalars, nothing trained, combat and seeds shared — and asks whether moving
  ELITE toward Silverbot's +0.22 or the human's +1.93 buys floors:

  | arm | elites/run | paired vs baseline |
  |---|---:|---:|
  | baseline | 0.85 | — |
  | elite +1 | 2.22 | **−1.95 ± 0.72** (t=−2.69) |
  | elite +2 | 2.23 | −1.93 ± 0.72 |
  | elite +3 | 2.23 | −1.93 ± 0.72 |
  | elite +4.5 | 2.23 | −1.93 ± 0.72 |
  | elite −1 | 0.81 | −0.38 ± 0.23 |

  Forcing elite capture to 2.2 per run — past the 1.85-per-Act-1 figure winning
  human runs show — **costs two floors**, and pushing avoidance further also
  costs. v31 sits at a local optimum in both directions. This is the
  1-parameter, distribution-matched version of the human-clone experiment
  (elite capture 22% → 75%, death at floor 7.08) with the two confounds removed,
  and it reaches the same verdict: **elite avoidance is a rational response to
  combat that is more expensive than the reference agents', not a preference
  inherited from bad labels.** The ELITE row in the conditional-logit table below
  is therefore a *description* of the capability gap and **not a fix target** —
  closing it requires a better policy or cheaper combat, not a better
  coefficient.

  Two further readings. Every arm from +1 to +4.5 is identical to three decimal
  places on floors and on W/T/L (27/46/47), so **the net's routing margins are
  under one logit** — a bias of this kind is a switch, not a dial. And the
  rest-side coefficients do not replicate: `hurtrest`/`rest` arms measured
  +0.50 ± 0.44 on the first 120 seeds and **+0.10 to +0.23 (all t < 1) on 240
  fresh seeds** (`runs/route_bias_rest_confirm.jsonl`), which is a textbook
  example of why this file insists on a second seed set.

- **Routing is real but miscalibrated** (`lightspeed/_routing_audit.py`, 60 seeds,
  635 path decisions). Two measurements borrowed from Silverbot's map program:

  A `--randomize-paths` intervention — replace every path choice with a uniform
  random legal one, keep the policy everywhere else — costs **4.87 mean floors**
  (28.45 → 23.58, sem 1.62, t=3.00). So the routing policy is doing real work,
  and is *not* in the ±0.1-nat "no detectable preference" state Silverbot's
  cheat-era policy was in.

  A conditional logit over path decisions, MONSTER as reference, against
  Silverbot's published honest1 coefficients:

  | Room | Silverbot | v31 | Baalorlord (human) |
  |---|---:|---:|---:|
  | REST | **+1.51** (z=56) | +0.02 (z=0.0) | **+3.19** (z=4.6) |
  | SHOP | +1.34 (z=43) | +1.84 (z=5.6) | +0.43 (z=2.3) |
  | EVENT | +0.64 (z=41) | +0.55 (z=3.6) | +0.44 (z=3.9) |
  | **ELITE** | **+0.22** (z=8) | **−2.55** (z=−6.4) | **+1.93** (z=10.2) |
  | hp_frac × REST | **−1.72** (z=−13) | +1.19 (z=1.2, n.s.) | **−1.93** (z=−2.2) |

  The human column was added 2026-07-31 from 998 multi-option decisions recovered
  by `lightspeed/replay_human_runs.py` (his base-35 seed regenerates the same map
  in our engine, so his recorded room sequence can be solved back into the node he
  clicked), fitted with this same `conditional_logit`. It is an independent
  confirmation of the diagnosis above from a completely different source:

  - **He and Silverbot agree on HP-conditioned resting** — −1.93 against −1.72,
    derived from 998 and 32.8k decisions respectively. Both rest *more* when hurt.
    v31's +1.19 is the wrong sign.
  - **He seeks elites harder than Silverbot** (+1.93 vs +0.22, z=10.2), matching
    the 1,008,636-run study below. v31 is 4.48 logits away — an inverted
    preference, not a miscalibrated one.
  - **The cheap preferences already match.** SHOP and EVENT agree across all three
    agents, exactly as this section says.

  So the two failing coefficients now have supervised labels with the correct sign
  and magnitude, on the decision class the 2026-07-31 layer swap proved carries
  the entire 15.71-floor gap ([03-combat-search.md](03-combat-search.md)).

  Per-room capture: SHOP 82%, REST 64%, EVENT 58%, **ELITE 4.6%**.

  So the policy learned the cheap preferences at roughly Silverbot's strength and
  failed to learn the two that govern act-boss survival: elite tolerance, and
  resting when hurt. Arriving under-relic'd *and* without HP management is exactly
  the profile that dies at floors 16–17 and 33–34, where 47.5% of deaths land.

  Caveat: 635 decisions against Silverbot's 32.8k, so our z-scores are far weaker.
  The ELITE and REST gaps are large enough to survive that; the smaller effects
  should not be over-read.
- **Drafting builds the wrong deck** (measured 2026-07-31, 60 runs vs his 100).
  Deck SIZE is not the problem -- at floor 10 v31 holds 16.6 cards against his
  16.2. The CARDS are. Nine of v31's top fifteen picks appear nowhere in his top
  twenty-five:

  | v31 picks | he picks |
  |---|---|
  | Perfected Strike 42, Anger 35, Twin Strike 30, Iron Wave 26, **Clash 25**, Carnage 18, Clothesline 17, Uppercut 15 | Shrug It Off 103, Feel No Pain 97, Dark Embrace 83, **Singing Bowl 82**, True Grit 58, Burning Pact 56, Offering 52, Second Wind 43 |

  v31 drafts raw attacks that read well alone; he drafts an exhaust engine of
  individually mediocre cards. **Clash is close to unplayable** (castable only
  with an all-attack hand) and v31 takes it 25 times; Perfected Strike is dead
  weight without a Strike-heavy deck. He also takes Singing Bowl -- +2 max HP --
  82 times, i.e. deliberately declining the cards offered.

  This is the upstream cause of the elite gap: a pile of vanilla attacks beats
  early monsters and then loses to act 2 elites, which is where runs die. Note
  the deeper-floor deck comparisons in any such audit are survivorship-biased
  (v31 reaches floor 34 in 3 of 60 runs; his figures are all n=100).

- **Campfire marginal collapse.** The net emits roughly P(REST)=0.41 /
  P(SMITH)=0.35 in every state — close to the label marginal — and argmax turns
  that into REST 100% of the time. Sampling preserves the proportions and plays
  much worse (26.30 → 14.82 at T=0.5). Part of this is downstream of the
  Armaments bug making upgrades free.

- **The policy is near-indifferent almost everywhere, and this is the general
  case of the entry above** (measured 2026-07-31 on v37 over 699 decisions from
  `lightspeed/ppo_collect.py` batches). Top-1 minus top-2 logit gap:

  | screen | n | median gap | frac < 0.1 | frac < 0.5 |
  |---|---:|---:|---:|---:|
  | REWARDS | 356 | 0.185 | 0.30 | 0.97 |
  | MAP | 131 | 0.199 | 0.29 | 0.83 |
  | EVENT | 68 | 0.057 | 0.94 | 1.00 |
  | REST | 48 | 0.088 | 0.85 | 1.00 |
  | SHOP | 43 | 0.117 | 0.40 | 0.91 |
  | CARD_SELECT | 32 | 0.067 | 0.62 | 1.00 |
  | **ALL** | **699** | **0.129** | **0.44** | **0.95** |

  **95% of decisions are decided by less than half a logit**, and on events,
  rests and card selects the model is effectively indifferent. Three separate
  observations collapse into this one fact: argmax at campfires is arbitrary but
  deterministic; the route-bias probe saturated at +1 because that is ~8x the
  median margin; and the whole 22.89-floor greedy policy is balanced on ~0.13
  nats. It is also what the label-SNR result predicts — the model faithfully
  learned near-ties from targets that were themselves near-tied on two-thirds of
  decisions.

  Consequence for on-policy RL: sampling costs floors steeply, because a 0.13
  gap is close to a coin flip. Measured against v37's 22.80 argmax baseline
  (96 episodes per arm, 100 sims):

  | temperature | mean floor | entropy |
  |---:|---:|---:|
  | argmax | 22.80 | — |
  | 0.2 | 18.12 ± 0.69 | 0.713 |
  | 0.4 | 16.00 ± 0.54 | 0.835 |
  | 0.6 | 14.42 ± 0.49 | 0.872 |
  | 0.8 | 13.48 ± 0.52 | 0.886 |
  | 1.0 | 12.94 ± 0.49 | 0.891 |

  Entropy moves only 25% across a 5x temperature range, which is itself the
  signature of near-uniform logits. The practical reading is favourable:
  **exploration is nearly free — T=0.2 still gives 0.713 nats** — so collection
  should run cold, near 0.2, and an RL curve starts from ~18 floors rather than
  from the 22.80 greedy number.
### What winning runs actually do (external, 1,008,636 runs)

The two weaknesses above have a measured target, from outside this project.
Porenius & Hansson, *Using machine learning to help find paths through the map
in Slay the Spire* (Malmö University, 2021), analysed **1,008,636 runs from the
official Slay the Spire developer dataset** — 93,242 wins against 915,394
losses, a 10.2% base win rate — and counted room types on winning versus losing
paths:

| Room type | Winning runs | Losing runs | Δ |
|---|---:|---:|---:|
| Elites, Act 1 | **1.85** | 1.32 | +0.53 |
| Elites, Act 2 | **1.68** | 1.00 | +0.68 |
| Elites, Act 3 | **1.59** | 1.03 | +0.56 |
| Campfires, Act 1 | **2.88** | 2.07 | +0.81 |
| Campfires, Act 2 | **2.93** | 1.81 | +1.13 |
| Campfires, Act 3 | **2.84** | 1.85 | +0.99 |
| Shops, Act 1 | 0.91 | 0.74 | +0.18 |
| Shops, Act 2 | 1.18 | 0.88 | +0.29 |
| Shops, Act 3 | 1.26 | 0.91 | +0.35 |

Winners take more elites in every act and roughly **one extra campfire per
act**. The paper's own ANN converged on the same preference independently — 8 of
its 10 predicted paths tied for most elites — and beat all three human controls
in its user study.

**These are exactly the two rooms this policy handles worst**, and the numbers
turn both from a direction into a target. Winning runs take ~1.85 elites in Act
1 alone; ours takes ~1.08 across an entire run, capturing 22% of those offered.
Campfires show the largest gap in the whole table, and ours resolves to REST
100% of the time.

Two caveats, and they matter. This is the full player population at a 10.2% win
rate, so it is mostly low-ascension play and its absolute numbers should not be
read as A20 targets. And it is **correlational** — a stronger deck both survives
elites and wins, so some of the gap is reverse causation, and the table does not
show that taking more elites *causes* wins. What it does establish is that our
agent is nowhere near the winning-run figures on either axis, which is not in
doubt from either direction.

Worth contrasting with the death-floor clustering below: `bottled_ai`, an
independent rules-based bot, also dies mostly at floors 16 and 33. Those are the
act-boss floors, so that clustering is likely structural to the game rather than
diagnostic of this policy in particular. The elite and campfire gaps are the
measurements that carry weight here; the death floors are not independent
evidence for them.

- **Zero A20 victories.** Across every checkpoint and every evaluation file, the
  only recorded A20 victory anywhere is a single v26 run out of 500. At A0, v31
  wins 13/100.

## Build note

The post-fix rebuild emitted LTO plugin warnings (`plugin needed to handle lto
object`). It links and imports correctly, and the test suite passes, but this
project's own convention is not to assume warnings are harmless. A clean rebuild
is advisable before a long run.

### `assert()` does nothing in any release build — including the SCRY guard

Confirmed 2026-07-31, and it is why the card-select defects above presented as
silence rather than as crashes. `include/sts_common.h:8` defines `sts_asserts`
unconditionally, so every `#ifdef sts_asserts` block compiles in — but
`CMakeLists.txt:17` passes `-DNDEBUG`, which makes `assert()` a no-op, as line
11 of that same file states outright.

The consequences are specific and worth knowing:

- The `assert(!legal.empty())` added by the SCRY fix
  (`slaythespire.cpp:1284`) never fires. In release, an empty enumeration is
  still a bare access violation at `legal[0]`, which is exactly what that
  assert was written to prevent. It works only in the debug build.
- `enumerateCardSelectActions`'s `default: assert(false)` likewise never fires;
  it returns an empty vector silently.
- The three card-type switches in `BattleContext.cpp` print
  `attempted to use unimplemented card: <name>` and then assert. **The
  `std::cerr` write is not inside the `assert`, so it does still happen** —
  which makes stderr the only runtime signal that a card is unimplemented, and
  is how the four dispatch bugs above were finally caught. `Action::execute`
  writes a similar dump when `isValidAction` rejects an action.

Practical upshot: watch stderr. `lightspeed/_class_card_audit.py` captures fd 2
rather than `sys.stderr` for this reason — C++ `std::cerr` does not go through
Python's stream objects.

### The test suite ignores `PYTHONPATH` for the native module

`sts/bridge/native_recommend.py:42` does
`sys.path.insert(0, <project>/sts_lightspeed/build)` with a hardcoded absolute
path. Three test files import it transitively — `test_native_recommend.py`,
`test_autobattle.py`, `test_engine_card_identity.py` — so as soon as any of them
is collected, `import slaythespire` resolves to `build/` for the **whole
session**, whatever `PYTHONPATH` says.

This matters when validating a build somewhere other than `build/`: the suite
will silently test the old binary and report a pass. Verify with
`python -c "import slaythespire; print(slaythespire.__file__)"` *after*
importing the bridge, not before.

### Resolved: shipped `.pyd` did not match the source tree (2026-07-30, 23:26)

`build/slaythespire.cp313-win_amd64.pyd` was built at **23:23**. At that moment
`bindings/slaythespire.cpp` contained a short-lived empty-legal-actions guard in
`nativeHeuristicPick` and both rollout loops, added while the SCRY crash was
still being diagnosed. That guard was **reverted at 23:26**, after the build.

**Confirmed present, 2026-07-31**: calling `heuristic_playout` on a state whose
enumeration is empty returns normally against the shipped binary instead of
crashing. So the binary does carry the reverted guard.

Two things follow, and the second is the one that matters:

- The guard is **unreachable on the Ironclad path** — the source trace plus a
  57,908-decision fuzz found no empty enumeration for an Ironclad — so no label
  generated against this binary is affected by it.
- The guard only ever covered `nativeHeuristicPick`, i.e. the rollout. It does
  **not** cover tree expansion, so even with it, `native_playout_battle` on a
  Silent deck segfaulted. It was never the safety net it looked like.

Superseded in practice by the enumeration fix above: with all 26 tasks
enumerable, the empty case the guard existed for cannot arise.

**Fix: one rebuild.** The compile from current source is already verified clean;
only the link is outstanding, and it was blocked by a running `lightspeed`
training job holding the `.pyd` (`ld.exe: cannot open output file ... Permission
denied`). Rebuild once that job finishes, before trusting any further evaluation
to match the tree.

Reminder for anyone hitting the same block: the link fails, the compile does not,
so `Error 2` from `cmake --build` here means "file locked", not "code broken".
