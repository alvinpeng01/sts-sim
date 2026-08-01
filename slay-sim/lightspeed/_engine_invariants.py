"""Engine invariants: properties that must hold whatever cards get played.

The two most expensive engine bugs this project has found were both silent
wrong-behaviour, not crashes, and neither needed a reference implementation to
detect -- each violated a property that is obviously true:

  * cardOnExit wrote combat-scoped upgrades into the MASTER DECK. Armaments
    upgrades a card "for the rest of combat"; the upgrade persisted. Worth 3.2
    mean floors, larger than v31's entire gain over v28, and it inflated every
    measurement taken before 2026-07-30. Property violated: a battle must not
    change the master deck.
  * build_battle_context left every card at uniqueId -1, so removeFromHandById
    never matched and a played card stayed in hand. One Bash killed a 44 HP Jaw
    Worm. Property violated: playing a card must not increase how many copies of
    it exist.

That is the value here: the property IS the oracle, so no second engine and no
live capture is needed, and it covers all four characters at once because the
properties do not care which cards were played.

What this does NOT catch: a card with a wrong constant. A card dealing 8 damage
where the real game deals 9 satisfies every property below. That needs a real
oracle -- see docs/10-other-characters.md.

usage:  python -m lightspeed._engine_invariants [runs_per_character] [--sims N]

NOTE: requires an engine that can enumerate every character's card selects.
The pre-2026-07-31 build segfaults on Silent/Defect/Watcher -- see
docs/07-known-issues.md.
"""
from __future__ import annotations

import collections
import random
import sys

import slaythespire as sts

from lightspeed._class_card_audit import captured_cxx_stderr
from lightspeed.whole_run_env import partition_legal_actions

CHARACTERS = ["IRONCLAD", "SILENT", "DEFECT", "WATCHER"]

# Cards allowed to change in the master deck as a result of a battle. Both
# change only THEMSELVES, via `misc`, which is why they can be whitelisted
# without blunting the invariant.
DECK_CHANGE_WHITELIST = {
    "RITUAL_DAGGER",       # keeps the damage it gained on a kill
    "GENETIC_ALGORITHM",   # keeps the block it gained
}

# Lesson Learned upgrades a RANDOM card in the deck on a kill, so any card may
# legitimately change while it is in the deck. Whitelisting every card would
# make the invariant vacuous -- exactly the mistake of whitelisting Armaments,
# the card the whole invariant exists to catch. Instead, a deck holding it makes
# that battle unverifiable, and unverifiable battles are counted and reported
# rather than quietly skipped.
DECK_CHECK_DISABLED_BY = {"LESSON_LEARNED"}

# Stacked into the deck before walking a run. Random overworld play picks up
# almost no cards, so an unstacked run stays near the 10-card starting deck and
# would never draw the cards whose effects are permanent -- the invariant would
# pass by never being exercised. These are the permanent-effect cards plus
# upgrade targets for Armaments to act on.
STACK = {
    "IRONCLAD": ["ARMAMENTS", "ARMAMENTS", "RITUAL_DAGGER", "STRIKE_RED",
                 "STRIKE_RED", "BASH", "ANGER", "SHRUG_IT_OFF", "IRON_WAVE"],
    "SILENT":   ["GENETIC_ALGORITHM", "FEED", "SURVIVOR", "ACROBATICS",
                 "PREPARED", "DAGGER_THROW", "NEUTRALIZE"],
    "DEFECT":   ["SELF_REPAIR", "ZAP", "DUALCAST", "COOLHEADED", "SCRAPE",
                 "BLIZZARD", "HOLOGRAM"],
    "WATCHER":  ["ERUPTION", "VIGILANCE", "PRESSURE_POINTS", "SETUP",
                 "WELL_LAID_PLANS", "NIGHTMARE"],
}


def deck_key(gc):
    """Multiset of the master deck. `misc` carries Ritual Dagger's damage and
    Genetic Algorithm's block, so it has to be part of the identity or those
    permanent changes -- and any bug imitating them -- are invisible."""
    return collections.Counter(
        (str(c.id).split(".")[-1], c.upgrade_count, c.misc) for c in gc.deck)


def pile_counts(bc):
    """Per-card-id totals across all four piles."""
    c = collections.Counter()
    for pile in (bc.hand, bc.draw_pile, bc.discard_pile, bc.exhaust_pile):
        for card in pile:
            c[str(card.id).split(".")[-1]] += 1
    return c


# A card reward must come from the character's own pool. CardPools.h uses
# explicit hard-coded per-character lists and never reads cardColors, so this is
# structurally immune to the colour-table bug class -- but that is a claim about
# the code, and this checks it against what the game actually offers.
# COLORLESS is legitimate: verified 2026-07-31 that every colorless offer landed
# on floor 0 holding only the starting relic, i.e. Neow.
CHARACTER_COLOR = {"IRONCLAD": "RED", "SILENT": "GREEN",
                   "DEFECT": "BLUE", "WATCHER": "PURPLE"}
STARTER_CARDS = {"STRIKE_RED", "STRIKE_GREEN", "STRIKE_BLUE", "STRIKE_PURPLE",
                 "DEFEND_RED", "DEFEND_GREEN", "DEFEND_BLUE", "DEFEND_PURPLE",
                 "ASCENDERS_BANE"}


class Findings:
    def __init__(self):
        self.deck_leaks = []
        self.duplications = []
        self.sanity = []
        self.reward_pool = []

    def total(self):
        return (len(self.deck_leaks) + len(self.duplications)
                + len(self.sanity) + len(self.reward_pool))


def check_state_sanity(bc, where, found):
    if bc.player_hp < 0 or bc.player_hp > bc.player_max_hp:
        found.sanity.append(f"{where}: hp {bc.player_hp}/{bc.player_max_hp}")
    if bc.player_block < 0:
        found.sanity.append(f"{where}: block {bc.player_block}")
    if bc.player_energy < 0:
        found.sanity.append(f"{where}: energy {bc.player_energy}")


def pass_master_deck(character, runs, sims, rng, found):
    """A battle must not change the master deck.

    Must go through the whole-run path: new_battle() builds a BattleContext
    decoupled from GameContext.deck, so a write-through is invisible there --
    verified, 0/4 fights mutated gc.deck that way. Only
    native_playout_current_battle_result synchronises back, which is why the
    Armaments leak showed up in run economy and not in any combat test.
    """
    cc = getattr(sts.CharacterClass, character)
    battles = unverifiable = offers = 0
    # The measurement that originally exposed the Armaments leak: _run_audit.py
    # reported 7.6 upgrades per run while the policy took REST at 137 of 137
    # campfires. Splitting upgrades by whether they appeared across a battle or
    # outside one makes a zero here interpretable -- it says the check was
    # exercised, not merely that nothing happened to change.
    upgrades_in_battle = upgrades_outside = 0
    for _ in range(runs):
        gc = sts.GameContext(cc, rng.randint(1, 10 ** 8), 20)
        for name in STACK.get(character, []):
            gc.obtain_card(sts.Card(getattr(sts.CardId, name)))
        for _ in range(400):
            if gc.outcome != sts.GameOutcome.UNDECIDED:
                break
            if gc.screen_state == sts.ScreenState.BATTLE:
                before = deck_key(gc)
                floor = gc.floor_num
                deck_ids = {k[0] for k in before}
                blocked = deck_ids & DECK_CHECK_DISABLED_BY
                ups_before = sum(c.upgrade_count for c in gc.deck)
                sts.native_playout_current_battle_result(gc, sims)
                battles += 1
                after = deck_key(gc)
                upgrades_in_battle += sum(c.upgrade_count for c in gc.deck) - ups_before
                if blocked:
                    unverifiable += 1
                elif before != after:
                    delta = (after - before) + (before - after)
                    unexplained = {k: v for k, v in delta.items()
                                   if k[0] not in DECK_CHANGE_WHITELIST}
                    if unexplained:
                        found.deck_leaks.append(
                            f"{character} seed={gc.seed} floor={floor}: {unexplained}")
                continue
            if gc.screen_state == sts.ScreenState.REWARDS:
                offers += check_reward_pool(gc, character, found)
            actions, _ = partition_legal_actions(gc)
            if not actions:
                break
            ups_before = sum(c.upgrade_count for c in gc.deck)
            rng.choice(actions).execute(gc)
            upgrades_outside += sum(c.upgrade_count for c in gc.deck) - ups_before
    return battles, unverifiable, upgrades_in_battle, upgrades_outside, offers


BENIGN_REWARD_NOISE = "was not in a state with card rewards"


def check_reward_pool(gc, character, found):
    """Every offered card is the character's own colour, or colorless.

    A REWARDS screen can carry gold, a potion or a relic and no cards, and
    get_card_reward writes a message to std::cerr in that case. Capture fd 2
    around the call so the expected message stays out of the report, but keep
    anything else it prints -- silently swallowing engine stderr is how the
    unimplemented-card diagnostics went unnoticed in the first place.
    """
    with captured_cxx_stderr() as cap:
        try:
            cards = list(gc.get_card_reward())
        except Exception:
            cards = []
    noise = cap.text.strip()
    if noise and BENIGN_REWARD_NOISE not in noise:
        found.sanity.append(f"{character} floor={gc.floor_num} stderr: {noise[:200]}")
    if not cards:
        return 0
    want = CHARACTER_COLOR[character]
    names = [str(c.id).split(".")[-1] for c in cards]
    if len(set(names)) != len(names):
        found.reward_pool.append(f"{character} floor={gc.floor_num}: "
                                 f"duplicate card in one reward {names}")
    for card, name in zip(cards, names):
        color = str(sts.get_card_color(card.id)).split(".")[-1]
        if color not in (want, "COLORLESS"):
            found.reward_pool.append(
                f"{character} floor={gc.floor_num}: offered {name} ({color})")
        if name in STARTER_CARDS:
            found.reward_pool.append(
                f"{character} floor={gc.floor_num}: offered starter {name}")
    return len(cards)


def pass_card_accounting(character, fights, rng, found):
    """Playing a card must not increase how many copies of it exist.

    Checked per play across all four piles. Cards that legitimately create
    copies (Dual Wield) or add cards (Immolate's Burn, Blade Dance's Shivs)
    increase OTHER ids, so attributing the increase to the id actually played
    isolates duplication from generation.
    """
    cc = getattr(sts.CharacterClass, character)
    encounters = [e for e in ("CULTIST", "JAW_WORM", "TWO_LOUSE", "GREMLIN_NOB",
                              "THE_GUARDIAN", "BOOK_OF_STABBING")
                  if hasattr(sts.MonsterEncounter, e)]
    plays = 0
    for _ in range(fights):
        gc = sts.GameContext(cc, rng.randint(1, 10 ** 8), 20)
        bc = sts.new_battle(gc, getattr(sts.MonsterEncounter,
                                        rng.choice(encounters)))
        for _ in range(300):
            if bc.outcome != sts.BattleOutcome.UNDECIDED:
                break
            legal = list(bc.get_legal_actions())
            if not legal:
                break
            check_state_sanity(bc, f"{character} pre-action", found)
            action = rng.choice(legal)
            played_id = None
            if (action.action_type == sts.ActionType.CARD
                    and 0 <= action.source_idx < len(bc.hand)):
                played_id = str(bc.hand[action.source_idx].id).split(".")[-1]
            before = pile_counts(bc)
            action.execute(bc)
            if played_id is not None:
                plays += 1
                after = pile_counts(bc)
                if after[played_id] > before[played_id]:
                    found.duplications.append(
                        f"{character}: playing {played_id} raised its own count "
                        f"{before[played_id]} -> {after[played_id]}")
    return plays


def self_test(rng):
    """Prove the deck invariant actually fires.

    A harness that never reports anything is indistinguishable from one that
    cannot report anything, so corrupt a deck deliberately and confirm the
    comparison catches it.
    """
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 4242, 20)
    before = deck_key(gc)
    gc.obtain_card(sts.Card(sts.CardId.BASH))          # unexplained addition
    after = deck_key(gc)
    delta = (after - before) + (before - after)
    caught = {k: v for k, v in delta.items()
              if k[0] not in DECK_CHANGE_WHITELIST}
    return bool(caught), caught


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    runs = int(args[0]) if args else 3
    sims = 30
    if "--sims" in sys.argv:
        sims = int(sys.argv[sys.argv.index("--sims") + 1])

    ok, caught = self_test(random.Random(0))
    print(f"self-test: deck invariant fires on a deliberate change -> "
          f"{'PASS' if ok else '*** FAIL, harness is blind ***'} {caught}")
    if not ok:
        return 2
    print()

    found = Findings()
    print(f"{'character':10s} {'battles':>8s} {'unverif':>8s} {'plays':>8s} "
          f"{'offers':>7s} {'upg_in_battle':>14s} {'upg_outside':>12s}")
    print("-" * 76)
    for ch in CHARACTERS:
        rng = random.Random(0xA11CE + sum(ch.encode()))
        battles, unverifiable, ups_in, ups_out, offers = pass_master_deck(
            ch, runs, sims, rng, found)
        plays = pass_card_accounting(ch, 3, rng, found)
        print(f"{ch:10s} {battles:8d} {unverifiable:8d} {plays:8d} "
              f"{offers:7d} {ups_in:14d} {ups_out:12d}")

    print()
    for label, items in (("master-deck leaks", found.deck_leaks),
                         ("card duplications", found.duplications),
                         ("state sanity", found.sanity),
                         ("reward pool", found.reward_pool)):
        print(f"{label}: {len(items)}")
        for it in items[:8]:
            print(f"   {it}")
    print()
    if found.total() == 0:
        print("PASS -- no invariant violations.")
        return 0
    print(f"FAIL -- {found.total()} violation(s).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
