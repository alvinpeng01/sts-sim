"""Monster identity index for a learned embedding table, mirroring cards.py's
card_index/OTHER_CARD_INDEX/EMBEDDING_VOCAB_SIZE pattern.

sts_lightspeed's MonsterId enum isn't exposed via pybind (unlike CardId), so
this keys off Monster::name instead (already bound as `.name`) -- the 47
distinct strings actually returned by bc.monsters across every encounter in
ALL_ACT_TIER_GROUPS, collected by direct enumeration (sts.new_battle(...) for
every encounter, not guessed from the wiki or the C++ enum source). Includes
'INVALID = 0', the placeholder Monster::getName() returns for the reserved-
but-not-yet-spawned-into slots that Bronze Automaton/Collector/Gremlin
Leader/Reptomancer encounters carry before their mid-fight spawn triggers --
a real, recurring, meaningful state (distinct from every actual monster) that
deserves its own embedding row rather than falling into OTHER_MONSTER_INDEX.

Added specifically because the observation had no way to tell the network
WHICH monster it's fighting -- only anonymous aggregates (HP, strength,
vulnerable, weak, block, incoming damage). Donu & Deca's focus-fire logic,
Reptomancer's dagger-priority, Time Eater's card-count discipline, and
Awakened One's revive-expectation are all boss-*specific* strategies the
network previously had to re-infer from scratch each time from numbers alone.
"""

from __future__ import annotations

MONSTER_NAMES = (
    "ACID_SLIME_M",
    "AWAKENED_ONE",
    "BLUE_SLAVER",
    "BOOK_OF_STABBING",
    "BRONZE_AUTOMATON",
    "BYRD",
    "CENTURION",
    "CHOSEN",
    "CULTIST",
    "DAGGER",
    "DARKLING",
    "DECA",
    "DONU",
    "EXPLODER",
    "FAT_GREMLIN",
    "FUNGI_BEAST",
    "GREEN_LOUSE",
    "GREMLIN_LEADER",
    "GREMLIN_NOB",
    "GREMLIN_WIZARD",
    "HEXAGHOST",
    "INVALID = 0",
    "JAW_WORM",
    "LAGAVULIN",
    "LOOTER",
    "MYSTIC",
    "NEMESIS",
    "ORB_WALKER",
    "RED_LOUSE",
    "RED_SLAVER",
    "REPTOMANCER",
    "REPULSOR",
    "SENTRY",
    "SHELLED_PARASITE",
    "SLIME_BOSS",
    "SNAKE_PLANT",
    "SNEAKY_GREMLIN",
    "SNECKO",
    "SPHERIC_GUARDIAN",
    "SPIKER",
    "SPIKE_SLIME_M",
    "SPIKE_SLIME_S",
    "THE_CHAMP",
    "THE_COLLECTOR",
    "THE_GUARDIAN",
    "TIME_EATER",
    "WRITHING_MASS",
)

NUM_MONSTER_NAMES = len(MONSTER_NAMES)
assert NUM_MONSTER_NAMES == 47, f"expected 47 monster names, found {NUM_MONSTER_NAMES}"

_NAME_TO_INDEX = {name: i for i, name in enumerate(MONSTER_NAMES)}

# One extra embedding slot for any monster name outside this fixed set --
# same rationale as OTHER_CARD_INDEX: a future encounter or a spawn-slot
# monster this enumeration missed shouldn't crash, just embed as "unknown".
OTHER_MONSTER_INDEX = NUM_MONSTER_NAMES
MONSTER_EMBEDDING_VOCAB_SIZE = NUM_MONSTER_NAMES + 1


def monster_index(name: str) -> int:
    """0..46 index for a known monster name, or OTHER_MONSTER_INDEX for
    anything outside the set -- never raises."""
    return _NAME_TO_INDEX.get(name, OTHER_MONSTER_INDEX)
