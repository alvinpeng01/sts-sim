"""Relic pool for training, mirroring cards.py's real-data-weighted sampling.

sts_lightspeed's RelicId enum spans the full base-game relic pool (181
entries, all classes + shared). Excluded categories, based on real game
mechanics (not guessed at random -- each has a concrete reason tied to
what an isolated single-combat sim can and can't represent):

  - Other classes' starting relics (Cracked Core/Defect, Pure Water/Watcher,
    Ring of the Snake/Silent) -- this project trains Ironclad only, whose
    own starting relic (Burning Blood) is granted unconditionally rather
    than sampled, matching its ~98% real-run presence (general.json's
    other-metric-derived data; see ironclad_relic_wins.csv).
  - Pure gold/economy relics (Maw Bank, Old Coin, Membership Card, Ceramic
    Fish, N'loth's Gift, N'loth's Hungry Face, The Courier, Meal Ticket) --
    this project doesn't model gold or shops at all, so these would be
    silent no-ops; excluded rather than included-but-inert, to keep the
    pool's real slots meaningful.
  - Rest-site-specific relics (Ancient Tea Set, Shovel, Singing Bowl,
    Dolly's Mirror, Regal Pillow, Girya, Peace Pipe) -- their entire
    mechanic depends on a rest-site interaction this project never
    simulates (no map/floor progression, only isolated fights).
  - Obtain-time-only relics whose effect is a one-shot deck/relic
    modification at pickup (Frozen/Molten/Toxic Egg, Empty Cage, Tiny
    House, Orrery, Matryoshka, Pandora's Box, Calling Bell, Cauldron,
    Question Card, Prayer Wheel, Tiny Chest, White Beast Statue, Spirit
    Poop, Bloody Idol, Astrolabe) -- nothing left to simulate mid-combat
    once the deck for this episode is already generated.
  - Potion-referencing relics (Sozu, Potion Belt, Toy Ornithopter, Strange
    Spoon) -- this project never grants potions, so these would be inert.
  - Nilry's Codex specifically -- its mid-combat effect (peek the top 3
    draw-pile cards, choose one) needs a CARD_SELECT-style input state;
    get_legal_actions() only recognizes PLAYER_NORMAL and CARD_SELECT
    already-wired cases (see bindings-util.cpp's own comment on this), and
    this hasn't been verified to route through cleanly -- excluded until
    checked, rather than risk an unhandled input state mid-episode.

Max-HP-on-pickup relics (Mango, Strawberry, Pear, Lee's Waffle) ARE
included -- env.py grants relics BEFORE setting the tier's calibrated
player_hp/max_hp (see IroncladFightEnv.reset()), so the explicit
tier-resource assignment always wins regardless of what a relic added,
avoiding any conflict with ACT_TIER_RESOURCES' own calibration.

This exclusion list is a best-effort categorization from real game
mechanics, not exhaustively hand-verified against source for all 181
entries the way card costs were -- the broad randomized smoke test (see
test_relics.py-style verification run alongside this) is the actual
safety net for anything mis-categorized here: a genuinely broken relic
would surface as a real crash/exception during that test, not silently.
"""

from __future__ import annotations

import json as _json
import os as _os

import slaythespire as sts

_EXCLUDED_NAMES = {
    "INVALID", "CIRCLET", "RED_CIRCLET",
    # other classes' starting relics
    "CRACKED_CORE", "PURE_WATER", "RING_OF_THE_SNAKE",
    # Other-class mechanic relics -- this project is Ironclad-only, so
    # anything whose entire effect references Orbs/Focus (Defect), Stance/
    # Mantra (Watcher), or Shiv/start-of-combat-Poison (Silent) is either
    # meaningless noise (Ironclad has no orb slots or stances to trigger
    # these) or actively crash-prone: NINJA_SCROLL grants 3 Shiv, and Shiv
    # is not fully implemented as a playable card in this engine --
    # confirmed via a real crash during the broad smoke test ("attempted
    # to use unimplemented card: Shiv", assertion failure at
    # BattleContext.cpp:1424). Scoped to Ironclad only "for now" per
    # explicit instruction -- revisit if Silent/Defect/Watcher training
    # ever gets added.
    "NINJA_SCROLL", "TWISTED_FUNNEL",  # Silent (Shiv / start-of-combat Poison)
    "HOLY_WATER",  # Watcher (grants Miracle, also unimplemented as playable here -- confirmed via a real
                   # crash: "attempted to use unimplemented card: Miracle", BattleContext.cpp:2014)
    "FROZEN_CORE", "DATA_DISK", "RUNIC_CAPACITOR", "EMOTION_CHIP", "GOLD_PLATED_CABLES",  # Defect (Orbs)
    "CLOAK_CLASP", "TEARDROP_LOCKET", "VIOLET_LOTUS", "DAMARU",  # Watcher (Stance/Mantra)
    # The exclusions above were assembled per-mechanic and left 13 gaps: these are
    # character-specific relics from RelicPools.h's Silent/Defect/Watcher namespaces,
    # so an Ironclad can never obtain any of them in a real run and sampling them is
    # pure fidelity error. Found 2026-07-30 by diffing this pool against those pools.
    #
    # MELANGE is the one that bit. "Whenever you shuffle your draw pile, Scry 3" fires
    # constantly on an Ironclad deck, driving the engine into InputState::SCRY -- a state
    # sts::py::getLegalActions had no case for, so it returned an EMPTY action list and
    # nativeHeuristicPick's `return legal[0]` segfaulted mid-rollout. The engine gap is
    # fixed (bindings-util.cpp), but these still should not be here.
    "SNECKO_SKULL", "PAPER_KRANE", "TINGSHA", "TOUGH_BANDAGES",
    "RING_OF_THE_SERPENT", "HOVERING_KITE", "WRIST_BLADE",       # Silent
    "SYMBIOTIC_VIRUS", "INSERTER", "NUCLEAR_BATTERY",            # Defect
    "DUALITY", "GOLDEN_EYE", "MELANGE",                          # Watcher
    # pure gold/economy
    "MAW_BANK", "OLD_COIN", "MEMBERSHIP_CARD", "CERAMIC_FISH",
    "NLOTHS_GIFT", "NLOTHS_HUNGRY_FACE", "THE_COURIER", "MEAL_TICKET",
    # rest-site specific
    "ANCIENT_TEA_SET", "SHOVEL", "SINGING_BOWL", "DOLLYS_MIRROR",
    "REGAL_PILLOW", "GIRYA", "PEACE_PIPE",
    # obtain-time-only deck/relic modification, nothing left to simulate
    "FROZEN_EGG", "MOLTEN_EGG", "TOXIC_EGG", "EMPTY_CAGE", "TINY_HOUSE",
    "ORRERY", "MATRYOSHKA", "PANDORAS_BOX", "CALLING_BELL", "CAULDRON",
    "QUESTION_CARD", "PRAYER_WHEEL", "TINY_CHEST", "WHITE_BEAST_STATUE",
    "SPIRIT_POOP", "BLOODY_IDOL", "ASTROLABE", "SMILING_MASK",
    # potion-referencing, inert (no potions modeled)
    "SOZU", "POTION_BELT", "TOY_ORNITHOPTER", "STRANGE_SPOON",
    # unverified mid-combat CARD_SELECT-style input state
    "NILRYS_CODEX",
    # THE_SPECIMEN (yes, also a relic, not just a monster): triggers
    # SELECT_ENEMY_THE_SPECIMEN_APPLY_POISON on any monster death --
    # confirmed via a real crash ("unhandled InputState raw=28",
    # Monster.cpp:331) -- get_legal_actions() doesn't recognize this
    # input state, same category as Nilry's Codex above.
    "THE_SPECIMEN",
    # Watcher-specific relic (grants Shiv, a token card this engine doesn't
    # fully implement as playable -- confirmed via a real crash during the
    # broad smoke test: "attempted to use unimplemented card: Shiv",
    # assertion failure at BattleContext.cpp:1424). Not Ironclad-obtainable
    # in the real game anyway (Watcher-class-restricted relic pool), so
    # this was a real categorization mistake, not just a defensive exclude.
    "NINJA_SCROLL",
    # NEOWS_LAMENT: forces every monster to 1 HP at battle start (for the
    # relic's first 3 uses). Root-caused as the cause of TWO separate hard
    # C++ asserts during the broad smoke test, both at the same line
    # (MonsterSpecific.cpp:1875, the generic `case MMID::INVALID:`
    # fallback) -- once against REPTOMANCER, once against AUTOMATON.
    # Bisection (re-running with each of a crashing draw's other relics
    # individually omitted, via a wrapper around weighted_ironclad_relics
    # that preserves the real rng consumption and only filters the
    # returned list post-hoc) isolated NEOWS_LAMENT as the one relic whose
    # omission fixed both crashes; every other relic in both draws was
    # confirmed innocent this way. Root cause: some bosses' move-selection
    # logic branches on HP thresholds/turn-number state that assumes a
    # normal starting HP and never expects to already be at 1 HP turn one
    # -- forcing that produces an uninitialized/out-of-range move index.
    # Also a realism mismatch, not just a safety one: a real player spends
    # this relic on trivial early fights, never on a randomly-drawn boss
    # like this project's per-episode sampling can produce.
    "NEOWS_LAMENT",
}

BURNING_BLOOD = sts.RelicId.BURNING_BLOOD

ALL_RELIC_NAMES = sorted(
    n for n in dir(sts.RelicId)
    if not n.startswith("_") and n not in ("name", "value") and n not in _EXCLUDED_NAMES
    and n != "BURNING_BLOOD"
)
RELIC_IDS = [getattr(sts.RelicId, n) for n in ALL_RELIC_NAMES]

with open(_os.path.join(_os.path.dirname(__file__), "data", "ironclad_relic_wins.csv")) as _f:
    import csv as _csv
    _counts = {}
    for _row in _csv.DictReader(_f):
        _counts[_row["Relic"]] = _counts.get(_row["Relic"], 0) + 1
_TOTAL_GAMES = 50  # ironclad_general.json's total_games_played

# Same smoothing rationale as cards.py's PICK_RATE_SMOOTHING -- this sample
# is even thinner per-relic than the card data (134 unique relics across
# only 50 games, most single-digit counts), so the floor matters more here,
# not less.
RELIC_FREQUENCY_SMOOTHING = 0.05

# Relic display names in the CSV don't always match the enum name exactly
# (spacing/casing/apostrophes) -- normalize both sides to a bare-alnum key
# for matching rather than requiring an exact manual alias table for all 181.
def _normalize(name: str) -> str:
    return "".join(ch for ch in name.upper() if ch.isalnum())

_NORMALIZED_COUNTS = {_normalize(k): v for k, v in _counts.items()}

RELIC_WEIGHTS = {
    rid: (_NORMALIZED_COUNTS.get(_normalize(name), 0) / _TOTAL_GAMES) + RELIC_FREQUENCY_SMOOTHING
    for rid, name in zip(RELIC_IDS, ALL_RELIC_NAMES)
}


# BOSS-rarity relics (the 3-choice pool offered after killing an act boss),
# queried from the engine's own authoritative relicTiers[] table (see
# sts_lightspeed's bindings/slaythespire.cpp:is_boss_relic) rather than a
# hand-maintained list -- filtered against RELIC_IDS so any already-excluded
# relic (e.g. a boss relic this engine can't simulate) stays excluded here
# too, rather than needing its own separate check.
BOSS_RELIC_IDS = [r for r in RELIC_IDS if sts.is_boss_relic(r)]


def _weighted_sample_without_replacement(rng, pool: list, count: int) -> list:
    pool = list(pool)
    weights = [RELIC_WEIGHTS[r] for r in pool]
    chosen = []
    for _ in range(min(count, len(pool))):
        total = sum(weights)
        r = rng.random() * total
        acc = 0.0
        for i, w in enumerate(weights):
            acc += w
            if r <= acc:
                chosen.append(pool.pop(i))
                weights.pop(i)
                break
        else:
            chosen.append(pool.pop())
            weights.pop()
    return chosen


def weighted_ironclad_relics(rng, count: int, n_boss: int = 0) -> list:
    """Sample `count` distinct relics (no duplicates -- a real run never
    has two copies of the same relic) weighted by real frequency. Returns
    RelicId values; Burning Blood is NOT included here, it's granted
    unconditionally by the caller (see env.py).

    `n_boss`: how many of the `count` relics must be BOSS-rarity, mirroring
    a real run's guaranteed boss-relic pickup after each act boss kill (1 by
    Act 2, 2 by Act 3 -- see ACT_TIER_RESOURCES' 6th element in env.py).
    Without this, general-pool frequency sampling would only include a boss
    relic by chance, understating how often the strongest relics actually
    show up by that point in a real run."""
    n_boss = min(n_boss, count, len(BOSS_RELIC_IDS))
    boss_chosen = _weighted_sample_without_replacement(rng, BOSS_RELIC_IDS, n_boss)
    # Excludes ALL boss-tier relics, not just the ones just chosen -- the
    # "rest" draw is meant to come from the non-boss pool only, otherwise it
    # could independently pick one of the remaining (unchosen) boss relics
    # too, silently exceeding n_boss.
    boss_set = set(BOSS_RELIC_IDS)
    rest_pool = [r for r in RELIC_IDS if r not in boss_set]
    rest_chosen = _weighted_sample_without_replacement(rng, rest_pool, count - n_boss)
    return boss_chosen + rest_chosen
