"""Potion pool for training, mirroring relics.py's structure (Ironclad-only
scope, broad-inclusion-then-crash-test philosophy -- see relics.py's own
docstring for why that's the approach this project uses rather than
hand-verifying every entry against source up front).

sts_lightspeed's Potion enum spans the full base-game potion pool (43
entries: 2 sentinels + 41 real potions). Unlike relics, there's no real-data
usage CSV for potions in lightspeed/data/ (the MaT1g3R dataset this project's
card/relic weighting draws from doesn't break out potion choices) -- potions
are sampled UNIFORMLY among the included pool instead, same fallback this
project already uses for encounter_weights when nothing more specific is
set. Revisit if a real potion-usage data source turns up.

Excluded categories:

  - Other classes' exclusive potions (potionPool[class][0:3] in
    sts_lightspeed's Potions.h -- Ironclad's own three exclusive slots are
    Blood Potion/Elixir Potion/Heart Of Iron, included below): Poison
    Potion/Cunning Potion/Ghost In A Jar (Silent), Focus Potion/Potion Of
    Capacity/Essence Of Darkness (Defect), Bottled Miracle/Stance Potion/
    Ambrosia (Watcher). Verified directly from the engine's own
    potionPool table, not guessed -- Ironclad can never actually draw
    these in a real run.
  - Smoke Bomb: BattleContext::drinkPotion's case for it is a bare
    "// todo" with no effect at all (BattleContext.cpp:3026-3028) -- a
    confirmed engine no-op, not a scope judgment call, same category as
    Nilry's Codex/Ninja Scroll's unimplemented-card crashes in relics.py.

Not excluded despite initial suspicion, on closer reading of
BattleContext::drinkPotion (verified against source, not guessed):
  - Attack/Colorless/Power/Skill Potion (Discovery mechanic) and Liquid
    Memories (BetterDiscardPileToHandAction): both route through
    InputState::CARD_SELECT-style follow-up choices, which
    getLegalActions() already handles via Action::enumerateCardSelectActions
    (the same mechanism Exhume/Warcry/Armaments-style cards already use) --
    plausibly fine, not proven broken.
  - Entropic Brew: its drink case is a plain loop of obtainPotion calls,
    no SetState/special input-state call visible in the switch case
    despite the enum name suggesting a discard-choice sub-state exists
    somewhere in the engine -- plausibly fine.
  - Gambler's Brew: routes through Actions::GambleAction(), which is NOT
    obviously covered by the CARD_SELECT handling above -- higher
    suspicion of an unhandled input state, but not yet confirmed broken.

None of the four bullets above are hand-verified end-to-end the way card
costs were -- this is exactly what the broad randomized smoke test (see
test_relics.py-style verification, same idea applied to potions) is for:
a genuinely broken one surfaces as a real crash/exception during that test,
same safety net as relics.
"""

from __future__ import annotations

import slaythespire as sts

_EXCLUDED_NAMES = {
    "INVALID", "EMPTY_POTION_SLOT",
    # other classes' exclusive potions (verified against sts_lightspeed's
    # own potionPool[class] table in Potions.h, not guessed)
    "POISON_POTION", "CUNNING_POTION", "GHOST_IN_A_JAR",  # Silent
    "FOCUS_POTION", "POTION_OF_CAPACITY", "ESSENCE_OF_DARKNESS",  # Defect
    "BOTTLED_MIRACLE", "STANCE_POTION", "AMBROSIA",  # Watcher
    # confirmed engine no-op -- BattleContext::drinkPotion's SMOKE_BOMB
    # case is a bare "// todo", no effect implemented at all
    "SMOKE_BOMB",
}

ALL_POTION_NAMES = sorted(
    n for n in dir(sts.Potion)
    if not n.startswith("_") and n not in ("name", "value") and n not in _EXCLUDED_NAMES
)
POTION_IDS = [getattr(sts.Potion, n) for n in ALL_POTION_NAMES]


def uniform_ironclad_potions(rng, count: int) -> list:
    """Sample `count` distinct potions (no duplicates -- matches
    weighted_ironclad_relics' same no-duplicate convention, though unlike
    relics a real run CAN hold duplicate potions; kept distinct anyway for
    training variety, revisit if duplicate-potion states turn out to
    matter). Uniform rather than weighted -- see module docstring for why."""
    count = min(count, len(POTION_IDS))
    return rng.sample(POTION_IDS, count)
