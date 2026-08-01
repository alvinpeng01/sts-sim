"""The Ironclad card pool and its index mapping for the embedding table.

sts_lightspeed's CardId enum spans all 4 classes (371 entries); we filter to
CardColor.RED (Ironclad) via the get_card_color() binding added for this
project. That table has 4 known data-entry errors (verified directly against
the real game's card list, not guessed): BULLET_TIME and CONCENTRATE (real
Defect/blue cards) are mistagged RED, while COMBUST and BRUTALITY (real
Ironclad cards) are mistagged GREEN. The two mistakes happened to cancel out
count-wise -- both sides of the swap are exactly 2 cards -- which is how the
pool passed a "== 75 cards" sanity check for so long while actually
containing 2 wrong cards and missing 2 real ones. BULLET_TIME/CONCENTRATE
being in a training deck is not just wrong-flavor, it's a real bug: the
native engine has no execution logic for them in an Ironclad context, so
drawing and playing one silently no-ops (logged as "attempted to use
unimplemented card") instead of doing anything.
"""

from __future__ import annotations

import json as _json
import os as _os

import slaythespire as sts

_MISTAGGED_AS_RED_BUT_NOT_IRONCLAD = {"BULLET_TIME", "CONCENTRATE"}
_MISTAGGED_AS_NON_RED_BUT_ACTUALLY_IRONCLAD = {"COMBUST", "BRUTALITY"}

IRONCLAD_CARD_IDS = sorted(
    (getattr(sts.CardId, name) for name in dir(sts.CardId)
     if not name.startswith("_") and name not in ("name", "value")
     and name not in _MISTAGGED_AS_RED_BUT_NOT_IRONCLAD
     and (sts.get_card_color(getattr(sts.CardId, name)) == sts.CardColor.RED
          or name in _MISTAGGED_AS_NON_RED_BUT_ACTUALLY_IRONCLAD)),
    key=lambda cid: int(cid),
)

NUM_IRONCLAD_CARDS = len(IRONCLAD_CARD_IDS)
assert NUM_IRONCLAD_CARDS == 75, f"expected 75 Ironclad cards, found {NUM_IRONCLAD_CARDS}"

_ID_TO_INDEX = {cid: i for i, cid in enumerate(IRONCLAD_CARD_IDS)}

# One extra embedding slot for any card outside the 75-card pool that can
# still legitimately end up in hand -- Status cards like Wound/Dazed
# (shuffled in by Wild Strike/Reckless Charge, same mechanic our own sts/
# engine implements) aren't "Ironclad cards" by color, but they're real
# possible hand contents once such a card's been played. Total embedding
# table size is NUM_IRONCLAD_CARDS + 1, not NUM_IRONCLAD_CARDS.
OTHER_CARD_INDEX = NUM_IRONCLAD_CARDS
EMBEDDING_VOCAB_SIZE = NUM_IRONCLAD_CARDS + 1


def card_index(card_id: "sts.CardId") -> int:
    """0..74 index for a pool card, or OTHER_CARD_INDEX for anything outside
    the pool (Status cards, etc.) -- never raises, since what ends up in
    hand isn't limited to what we intentionally put in the deck."""
    return _ID_TO_INDEX.get(card_id, OTHER_CARD_INDEX)


# CardInstance (what bc.hand actually returns) doesn't expose .type, only
# .id -- but sts.Card(id).type does, and type is a fixed property of the
# id, not per-instance state, so this is precomputed once here instead of
# constructing a throwaway Card per lookup in the (per-action, per-decision)
# hot path. Used by env.py's action encoding: the policy had no way to tell
# an Attack from a Skill from a Power, which meant it structurally could not
# learn "this Skill will feed Gremlin Nob's Enrage" no matter how much
# training happened -- the information just wasn't in its input.
_TYPE_LOOKUP = {cid: sts.Card(cid).type for cid in IRONCLAD_CARD_IDS}


def card_type(card_id: "sts.CardId"):
    """CardType for a pool card, or None for anything outside the pool."""
    return _TYPE_LOOKUP.get(card_id)


STARTER_CARD_IDS = [sts.CardId.STRIKE_RED, sts.CardId.DEFEND_RED, sts.CardId.BASH]


def _make_card(rng, cid, upgrade_chance):
    card = sts.Card(cid)
    if rng.random() < upgrade_chance and card.upgradable:
        card.upgrade()
    return card


def random_ironclad_deck(rng, extra_cards: int = 15, upgrade_chance: float = 0.3,
                          exclude=None, force_include=None):
    """Starter deck (5 Strike, 4 Defend, 1 Bash -- via GameContext's default
    construction) plus `extra_cards` sampled from the 75-card pool (including
    possible extra Strike/Defend/Bash, matching real decks that often have
    duplicates), each independently upgraded with probability
    `upgrade_chance`. Returns a list of sts.Card to obtain via
    GameContext.obtain_card(), NOT including the starter deck itself.

    `exclude`: card IDs to never sample -- for training decks in a held-out-
    card generalization test.
    `force_include`: card IDs guaranteed to appear (each still independently
    upgraded per `upgrade_chance`) -- for eval decks in that same test, so
    held-out cards are actually observed in play, not just possibly present.
    """
    cards = []
    force_include = list(force_include or [])
    for cid in force_include:
        cards.append(_make_card(rng, cid, upgrade_chance))

    pool = IRONCLAD_CARD_IDS
    if exclude:
        exclude_set = set(exclude)
        pool = [cid for cid in IRONCLAD_CARD_IDS if cid not in exclude_set]

    remaining = max(0, extra_cards - len(force_include))
    for _ in range(remaining):
        cid = rng.choice(pool)
        cards.append(_make_card(rng, cid, upgrade_chance))
    return cards


# --- Real-deck-informed weighted sampling ---
#
# Source: MaT1g3R/Slay-the-Spire-data, results/200-rotating-sample/IRONCLAD
# (github.com/MaT1g3R/Slay-the-Spire-data) -- streamer run history (Run
# History Plus mod format), 50 games, 58% win rate (general.json). NOT a
# per-run master-deck dataset -- that repo only has aggregated CSVs, no raw
# per-combat deck state -- so this uses card_picks.csv: one row per card
# OFFERED at a reward screen with whether it was Picked. pick_rate[card] =
# P(picked | offered) across ~5500 real reward decisions, computed once via
# a one-off script and saved to data/ironclad_pick_rates.json (72 of the 75
# pool cards appear; BASH/DEFEND_RED/STRIKE_RED never show up since they're
# never offered as rewards, only ever starting cards).
#
# This is explicitly a MARGINAL signal -- how much real, moderately-skilled
# players value each card in isolation -- not a synergy/co-occurrence model;
# it won't specifically pair Limit Break with a Strength engine more than
# chance the way an actual archetype-aware generator would. It's the cheap,
# real first step: sampling weighted by genuine human preference instead of
# uniform, without needing full deck reconstruction this data doesn't have.
#
# Smoothing: a few real, playable cards (Searing Blow, Clash, Fire
# Breathing, Berserk) show 0/N picks in this specific 50-game sample --
# using the raw rate as a sampling weight would make the generator NEVER
# draw them, which contradicts this project's own established principle
# (random_ironclad_deck's docstring) that every pool card needs training
# exposure. PICK_RATE_SMOOTHING is added to every weight so the floor is
# "still sampled sometimes, just rarely" rather than zero.
PICK_RATE_SMOOTHING = 0.05

with open(_os.path.join(_os.path.dirname(__file__), "data", "ironclad_pick_rates.json")) as _f:
    _PICK_RATES_BY_NAME = _json.load(_f)

# Keyed by CardId (not the raw string), and defaults to 0.0 (-> just the
# smoothing floor) for any pool card absent from the data file -- covers
# BASH/DEFEND_RED/STRIKE_RED and is also forward-safe if IRONCLAD_CARD_IDS
# ever grows without a matching data refresh.
PICK_RATE_WEIGHTS = {
    cid: _PICK_RATES_BY_NAME.get(str(cid).replace("CardId.", ""), 0.0) + PICK_RATE_SMOOTHING
    for cid in IRONCLAD_CARD_IDS
}

# random.choices() builds its cumulative-weight table from `weights` fresh
# on every call -- fine for a one-off, but weighted_ironclad_deck runs once
# per episode (millions of times over a real training run), and the
# no-exclude case (the actual training path; exclude is only used by the
# held-out-card generalization evals) uses the exact same weights every
# single time. Precomputing this once and passing cum_weights= instead of
# weights= skips re-deriving it per call. Measured: 3.49x slower than plain
# rng.choice() in isolation (57.7 vs 16.6 microseconds/deck, 5000-deck
# benchmark) before this change; deck generation is a small fraction of a
# full episode either way (6.28ms/episode uniform vs 6.88ms weighted, ~10%
# episode-level overhead), but free to claw back so it's worth doing before
# a long run pays that cost millions of times.
import itertools as _itertools

_FULL_POOL_CUM_WEIGHTS = list(_itertools.accumulate(PICK_RATE_WEIGHTS[cid] for cid in IRONCLAD_CARD_IDS))


def weighted_ironclad_deck(rng, extra_cards: int = 15, upgrade_chance: float = 0.3,
                            exclude=None, force_include=None):
    """Same contract/signature as random_ironclad_deck (starter deck is
    still obtained separately by the caller), except the `extra_cards`
    sample is drawn weighted by PICK_RATE_WEIGHTS instead of uniformly --
    see the module comment above this function for the data source and its
    real limitation (marginal pick preference, not deck synergy)."""
    cards = []
    force_include = list(force_include or [])
    for cid in force_include:
        cards.append(_make_card(rng, cid, upgrade_chance))

    remaining = max(0, extra_cards - len(force_include))
    if remaining:
        if exclude:
            exclude_set = set(exclude)
            pool = [cid for cid in IRONCLAD_CARD_IDS if cid not in exclude_set]
            weights = [PICK_RATE_WEIGHTS[cid] for cid in pool]
            picks = rng.choices(pool, weights=weights, k=remaining)
        else:
            picks = rng.choices(IRONCLAD_CARD_IDS, cum_weights=_FULL_POOL_CUM_WEIGHTS, k=remaining)
        for cid in picks:
            cards.append(_make_card(rng, cid, upgrade_chance))
    return cards


# --- Archetype-aware sampling ---
#
# PICK_RATE_WEIGHTS is a MARGINAL signal (how much real players value each
# card in isolation) -- it has no notion of which cards belong together, so
# a weighted_ironclad_deck() deck can (and often does) draw strong cards
# from unrelated archetypes that don't actually combo, same as uniform
# sampling would. This is the real limitation flagged when that generator
# was built: not a bug, just a ceiling the pick-rate data alone can't break,
# since the source repo's raw per-run deck lists were never published (only
# aggregated CSVs -- confirmed by reading analyze.py's own generation code
# for card_wins.csv, which groups rows by run but drops the boundary once
# flattened, so reconstructing real co-occurrence from it isn't reliable).
#
# ARCHETYPES below is NOT derived from that data -- it's hand-encoded from
# well-established, verifiable Ironclad synergies (the actual card text and
# community-standard archetype names), the same kind of fact-checking
# rigor as a card's real cost or a boss's real HP elsewhere in this
# project, just sourced from game-mechanics knowledge instead of a wiki
# fetch. It's honest about being a DIFFERENT kind of signal than
# PICK_RATE_WEIGHTS: real-player-data vs. real-game-mechanics-knowledge,
# not a replacement for genuine per-run co-occurrence statistics.
ARCHETYPES = {
    # Strength sources + the attacks that specifically benefit from a large
    # Strength stack (Heavy Blade multiplies it 3x/5x; multi-hit attacks
    # apply it per hit).
    "strength": [
        "INFLAME", "FLEX", "DEMON_FORM", "SPOT_WEAKNESS", "BERSERK",
        "HEAVY_BLADE", "SWORD_BOOMERANG", "TWIN_STRIKE", "WHIRLWIND", "CARNAGE", "RAMPAGE",
    ],
    # Block that persists/scales instead of resetting -- Barricade is the
    # enabler, everything else either retains block or converts it directly
    # into a payoff (Body Slam's damage, Juggernaut's proc).
    "block_tank": [
        "BARRICADE", "ENTRENCH", "BODY_SLAM", "JUGGERNAUT", "METALLICIZE",
        "GHOSTLY_ARMOR", "IMPERVIOUS", "SHRUG_IT_OFF", "TRUE_GRIT",
    ],
    # Exhaust-matters shell: Dark Embrace/Feel No Pain/Sentinel all trigger
    # off a card leaving play via exhaust, Corruption forces every Skill to
    # exhaust to feed them.
    "exhaust": [
        "DARK_EMBRACE", "FEEL_NO_PAIN", "CORRUPTION", "SENTINEL", "SECOND_WIND", "FIEND_FIRE", "IMMOLATE",
    ],
    # HP spent as a resource for tempo/damage -- Rupture turns the HP loss
    # itself into Strength, the rest trade HP directly for energy/damage/draw.
    "self_damage": [
        "RUPTURE", "COMBUST", "HEMOKINESIS", "BLOODLETTING", "OFFERING", "SEEING_RED", "BRUTALITY", "BLOOD_FOR_BLOOD",
    ],
    # Big single burst hits, usually set up by Weak/Vulnerable or a second
    # copy via Double Tap.
    "burst": [
        "DOUBLE_TAP", "REAPER", "BLUDGEON", "UPPERCUT", "LIMIT_BREAK", "FEED",
    ],
}
_ARCHETYPE_NAME_ORDER = sorted(ARCHETYPES)
_ARCHETYPE_CARD_SETS = {
    name: {getattr(sts.CardId, n) for n in names} for name, names in ARCHETYPES.items()
}

ARCHETYPE_BONUS = 2.5  # multiplier on PICK_RATE_WEIGHTS for cards in the chosen archetype
NO_ARCHETYPE_PROB = 0.25  # fraction of decks generated with no archetype bias at all (pure pick-rate)

# One cum_weights table per archetype (plus "none"), precomputed once for
# the same reason _FULL_POOL_CUM_WEIGHTS was -- this runs once per episode,
# and there are only 6 possible weight vectors (5 archetypes + unbiased),
# so there's no need to ever rebuild one at call time in the no-exclude path.
_ARCHETYPE_CUM_WEIGHTS = {
    name: list(_itertools.accumulate(
        PICK_RATE_WEIGHTS[cid] * (ARCHETYPE_BONUS if cid in members else 1.0)
        for cid in IRONCLAD_CARD_IDS
    ))
    for name, members in _ARCHETYPE_CARD_SETS.items()
}
_ARCHETYPE_CUM_WEIGHTS["_none"] = _FULL_POOL_CUM_WEIGHTS


def synergy_ironclad_deck(rng, extra_cards: int = 15, upgrade_chance: float = 0.3,
                           exclude=None, force_include=None):
    """weighted_ironclad_deck plus an archetype bias: one archetype is
    chosen per deck (or none, with probability NO_ARCHETYPE_PROB, so
    training still sees plenty of non-archetypal/mixed decks) and its
    member cards get ARCHETYPE_BONUS x their real pick-rate weight for
    that deck's sampling. Same contract/signature as the other two deck
    generators otherwise. See the ARCHETYPES module comment for what this
    is (and isn't) -- hand-encoded real synergy knowledge, not a
    replacement for genuine per-run co-occurrence data this project
    doesn't have access to."""
    cards = []
    force_include = list(force_include or [])
    for cid in force_include:
        cards.append(_make_card(rng, cid, upgrade_chance))

    remaining = max(0, extra_cards - len(force_include))
    if remaining:
        chosen_name = "_none" if rng.random() < NO_ARCHETYPE_PROB else rng.choice(_ARCHETYPE_NAME_ORDER)
        if exclude:
            exclude_set = set(exclude)
            pool = [cid for cid in IRONCLAD_CARD_IDS if cid not in exclude_set]
            members = _ARCHETYPE_CARD_SETS.get(chosen_name, set())
            weights = [PICK_RATE_WEIGHTS[cid] * (ARCHETYPE_BONUS if cid in members else 1.0) for cid in pool]
            picks = rng.choices(pool, weights=weights, k=remaining)
        else:
            picks = rng.choices(IRONCLAD_CARD_IDS, cum_weights=_ARCHETYPE_CUM_WEIGHTS[chosen_name], k=remaining)
        for cid in picks:
            cards.append(_make_card(rng, cid, upgrade_chance))
    return cards
