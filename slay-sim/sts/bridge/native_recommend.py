"""Recommendation layer backed by THIS PROJECT'S native combat engine
(sts_lightspeed's `slaythespire` pybind11 module + lightspeed/tuned_search_params.json)
instead of sts.mcts's own Python MCTS + value_net_weights_v2.npz. Same job as
_try_recommend's old sts.mcts.mcts_choose_action path in communication_mod.py, built
to be a drop-in replacement: returns the exact (description, action, combat) shape
communication_mod.py's autobattle/_build_command plumbing already expects, so none of
that needs to change.

FULLY SELF-SUFFICIENT -- does NOT call state_mapper.build_combat_state(). An earlier
version of this module did, reusing it purely for hand/monster object identity
(card.name, target.json_index). That was found to be a real, costly mistake by
replaying this project's own captured live data (sts_raw_states.log, see
test_native_recommend.py): 257 of 300 sampled real states were a Taskmaster fight,
which sts/enemies.py (the older, pure-Python engine state_mapper wraps) has no class
for at all -- build_combat_state raised UnmappedContentError on every single one,
even though THIS module's own _MONSTER_ID_MAP already covers "SlaverBoss" (the real
CommunicationMod id) -> TASKMASTER (this engine's own, more complete roster) just
fine. Requiring the older engine's reconstruction to succeed first was silently
gating the native engine behind a SMALLER roster than the native engine itself
actually has -- so this module now builds its own lightweight _ShadowCombat directly
from the raw JSON (hand/monster .name fields are already real display names in the
JSON, no translation needed for THAT purpose) and never touches sts.enemies.py at all.

The actual DECISION still comes from sts.build_battle_context() + sts.run_mcts_search()
built directly from the same raw JSON -- this project's own C++ engine, tuned/
validated all last session at 83%+ win rate against the encounters it was tuned on,
vs. the old value-net path's own (lower, never fully validated end-to-end) win rate.

CRITICAL INVARIANT: the monster filter used to build bc's monster list here MUST
exactly match _ShadowCombat's own filter (is_gone OR current_hp<=0 -> skip), or a
target_idx returned by search would resolve to the WRONG monster. Both come from the
SAME single _live_monster_jsons() call in native_recommend(), by construction, so
this can't drift the way it could when two independent reconstructions were involved.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

_STS_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_STS_PROJECT_ROOT / "sts_lightspeed" / "build"))

import slaythespire as sts  # noqa: E402

import json as _json  # noqa: E402

_TUNED_PARAMS_PATH = _STS_PROJECT_ROOT / "slay-sim" / "lightspeed" / "tuned_search_params.json"
_params_loaded = False


def _ensure_params_loaded():
    global _params_loaded
    if _params_loaded:
        return
    try:
        with open(_TUNED_PARAMS_PATH) as f:
            sts.set_search_params(_json.load(f)["params"])
    except FileNotFoundError:
        print(f"[native_recommend] WARNING: {_TUNED_PARAMS_PATH} not found, "
              f"using un-tuned C++ defaults", file=sys.stderr)
    sts.set_leaf_eval_mode("rollout")  # the quality default -- see this session's own
                                        # speed-vs-quality investigation (29-60% win in
                                        # the faster "value" mode vs rollout's 83%+ --
                                        # not an acceptable trade for live recommendations)
    _params_loaded = True


N_SIMULATIONS = 200


def _normalize(name: str) -> str:
    return name.upper().replace(" ", "").replace("_", "")


# --- status mapping: CommunicationMod power "id" string -> our canonical
# PlayerStatus/MonsterStatus enum-name. Confirmed against a REAL live capture
# (sts_raw_states.log, this project's own prior run) that power ids are TitleCase
# WITH SPACES for multi-word powers ("Feel No Pain", "Demon Form") and single words
# for the rest ("Strength", "Entangled") -- _normalize's upper+strip-spaces matches
# our own SNAKE_CASE enum names directly for the overwhelming majority, so this only
# needs explicit entries for the CONFIRMED exceptions (Weakened != Weak,
# IntangiblePlayer is a distinct player-only id) plus a modest hand list for
# anything not yet observed live.
_PLAYER_STATUS_OVERRIDES = {
    _normalize(k): v for k, v in {
        "Weakened": "WEAK", "IntangiblePlayer": "INTANGIBLE",
    }.items()
}
_MONSTER_STATUS_OVERRIDES = {
    _normalize(k): v for k, v in {
        "Weakened": "WEAK",
    }.items()
}
# Canonical enum-name spellings this bridge knows to match _normalize()'d ids
# against directly (covers "Strength"->STRENGTH, "DemonForm"->DEMON_FORM, etc.
# automatically without needing an explicit entry per status).
_PLAYER_STATUS_NAMES = [
    "VULNERABLE", "WEAK", "FRAIL", "STRENGTH", "DEXTERITY", "ARTIFACT", "INTANGIBLE",
    "BARRICADE", "METALLICIZE", "PLATED_ARMOR", "REGEN", "RITUAL", "THORNS", "RAGE",
    "RUPTURE", "COMBUST", "DARK_EMBRACE", "DEMON_FORM", "FEEL_NO_PAIN", "FIRE_BREATHING",
    "FLAME_BARRIER", "JUGGERNAUT", "NOXIOUS_FUMES", "EVOLVE", "BRUTALITY", "CORRUPTION",
    "BUFFER", "DOUBLE_TAP", "ENVENOM", "CONFUSED", "NO_DRAW", "ENTANGLED", "CONSTRICTED",
    "WRAITH_FORM", "ENERGIZED", "PANACHE", "VIGOR", "ACCURACY", "AFTER_IMAGE",
    "BATTLE_HYMN", "BURST", "CREATIVE_AI", "DEVA", "DEVOTION", "ECHO_FORM",
    "ESTABLISHMENT", "FOCUS", "FORESIGHT", "LIKE_WATER", "LOOP", "MAGNETISM", "MAYHEM",
    "OMEGA", "PHANTASMAL", "SADISTIC", "STATIC_DISCHARGE", "THOUSAND_CUTS",
    "TOOLS_OF_THE_TRADE", "WAVE_OF_THE_HAND", "WELL_LAID_PLANS", "EQUILIBRIUM",
]
_MONSTER_STATUS_NAMES = [
    "VULNERABLE", "WEAK", "STRENGTH", "ARTIFACT", "METALLICIZE", "PLATED_ARMOR",
    "POISON", "REGEN", "RITUAL", "THORNS", "TIME_WARP", "MODE_SHIFT", "CURL_UP",
    "ANGRY", "ENRAGE", "INTANGIBLE", "SHACKLED", "CHOKED", "MARK", "LOCK_ON",
    "BLOCK_RETURN", "CORPSE_EXPLOSION", "SPORE_CLOUD", "THIEVERY", "MALLEABLE",
    "ASLEEP", "BARRICADE", "MINION", "PAINFUL_STABS", "STASIS", "FLIGHT",
]

_unmapped_logged = set()


def _match_status(power_id: str, overrides: dict, names: list, log_fn) -> Optional[str]:
    key = _normalize(power_id)
    if key in overrides:
        return overrides[key]
    for name in names:
        if _normalize(name) == key:
            return name
    if power_id not in _unmapped_logged:
        _unmapped_logged.add(power_id)
        log_fn(f"[native_recommend] unmapped power id '{power_id}' -- skipped")
    return None


# --- monster id mapping: CommunicationMod raw "id" -> our canonical MonsterId
# enum-name. Built from state_mapper.py's own EXTENSIVELY VERIFIED real-capture
# comments (Mugger, SlaverBlue, Healer, Donu/Deca, FuzzyLouseNormal/Defensive) plus a
# direct check of this project's own sts_raw_states.log (confirmed SlaverRed,
# SlaverBoss->Taskmaster), cross-referenced against our C++ engine's own (more
# precise) MonsterId roster -- e.g. our engine has REAL_LOUSE/GREEN_LOUSE as distinct
# types and ACID_SLIME_S/M/L, SPIKE_SLIME_S/M/L as three distinct sizes each, unlike
# the older sts/enemies.py's single approximating classes, so this mapping is a real
# precision improvement over the old engine, not just a port.
_MONSTER_ID_MAP = {
    _normalize(k): v for k, v in {
        "JawWorm": "JAW_WORM", "Cultist": "CULTIST",
        "FuzzyLouseNormal": "RED_LOUSE", "FuzzyLouseDefensive": "GREEN_LOUSE",
        "AcidSlime_S": "ACID_SLIME_S", "AcidSlime_M": "ACID_SLIME_M", "AcidSlime_L": "ACID_SLIME_L",
        "AcidSlimeS": "ACID_SLIME_S", "AcidSlimeM": "ACID_SLIME_M", "AcidSlimeL": "ACID_SLIME_L",
        "SpikeSlime_S": "SPIKE_SLIME_S", "SpikeSlime_M": "SPIKE_SLIME_M", "SpikeSlime_L": "SPIKE_SLIME_L",
        "SpikeSlimeS": "SPIKE_SLIME_S", "SpikeSlimeM": "SPIKE_SLIME_M", "SpikeSlimeL": "SPIKE_SLIME_L",
        "SlaverBlue": "BLUE_SLAVER", "SlaverRed": "RED_SLAVER", "SlaverBoss": "TASKMASTER",
        "Looter": "MUGGER", "Mugger": "MUGGER",
        "GremlinNob": "GREMLIN_NOB", "Lagavulin": "LAGAVULIN",
        "TheGuardian": "THE_GUARDIAN", "Guardian": "THE_GUARDIAN",
        "Sentry": "SENTRY", "MadGremlin": "MAD_GREMLIN", "SneakyGremlin": "SNEAKY_GREMLIN",
        "FatGremlin": "FAT_GREMLIN", "ShieldGremlin": "SHIELD_GREMLIN",
        "Byrd": "BYRD", "Mystic": "MYSTIC", "Healer": "MYSTIC",
        "Centurion": "CENTURION", "GremlinLeader": "GREMLIN_LEADER",
        "Champ": "THE_CHAMP", "TheChamp": "THE_CHAMP",
        "BronzeOrb": "BRONZE_ORB", "Automaton": "BRONZE_AUTOMATON", "BronzeAutomaton": "BRONZE_AUTOMATON",
        "TorchHead": "TORCH_HEAD", "Collector": "THE_COLLECTOR", "TheCollector": "THE_COLLECTOR",
        "GremlinWizard": "GREMLIN_WIZARD", "Darkling": "DARKLING", "OrbWalker": "ORB_WALKER",
        "WrithingMass": "WRITHING_MASS", "Spiker": "SPIKER", "Repulsor": "REPULSOR",
        "Exploder": "EXPLODER", "SphericGuardian": "SPHERIC_GUARDIAN", "Nemesis": "NEMESIS",
        "Dagger": "DAGGER", "Reptomancer": "REPTOMANCER",
        "AwakenedOne": "AWAKENED_ONE", "AwakenedOneCultist": "AWAKENED_ONE",
        "TimeEater": "TIME_EATER", "Donu": "DONU", "Deca": "DECA", "Hexaghost": "HEXAGHOST",
        "SlimeBoss": "SLIME_BOSS", "Chosen": "CHOSEN", "BookOfStabbing": "BOOK_OF_STABBING",
        "GiantHead": "GIANT_HEAD", "ShelledParasite": "SHELLED_PARASITE",
        "Bear": "BEAR", "CorruptHeart": "CORRUPT_HEART", "SnakePlant": "SNAKE_PLANT",
        "Snecko": "SNECKO", "SpireGrowth": "SPIRE_GROWTH", "SpireShield": "SPIRE_SHIELD",
        "SpireSpear": "SPIRE_SPEAR", "TheMaw": "THE_MAW", "Transient": "TRANSIENT",
        "Pointy": "POINTY", "Romeo": "ROMEO", "FungiBeast": "FUNGI_BEAST",
    }.items()
}


# --- relic mapping: RelicId is already a bound pybind11 enum (sts.RelicId), so this
# only needs raw relic-id-string -> our enum's exact name string, then a getattr --
# same convention as _map_potion in native_search_agent.py's own bridge. Confirmed
# against this project's own real capture (sts_raw_states.log): raw ids look like
# {"name": "Burning Blood", "id": "Burning Blood", "counter": -1} -- "id" is close to
# our enum spelling after normalizing, EXCEPT a stacked/multi-copy relic can carry a
# trailing " <N>" (observed: "Toxic Egg 2") that must be stripped first. NOTE: relics
# with ordinal>=128 (includes VAJRA) are silently no-op'd by the C++ side regardless
# of what's passed here -- see build_battle_context's own docstring for the
# pre-existing engine limitation this runs into (Player::relicBits0/relicBits1
# capacity, not something this Python layer can work around).
_unmapped_relic_warned = set()


def _map_relic(raw_id: str, log_fn) -> Optional[str]:
    import re
    stripped = re.sub(r"\s+\d+$", "", raw_id).strip()  # "Toxic Egg 2" -> "Toxic Egg"
    key = _normalize(stripped)
    for value_name in dir(sts.RelicId):
        if value_name.startswith("_"):
            continue
        if _normalize(value_name) == key:
            return value_name
    if raw_id not in _unmapped_relic_warned:
        _unmapped_relic_warned.add(raw_id)
        log_fn(f"[native_recommend] unmapped relic id '{raw_id}' -- skipped (search will "
               f"not account for it)")
    return None


# --- potion mapping: Potion is already a bound pybind11 enum (sts.Potion), same
# normalize-then-getattr convention as _map_relic above. Confirmed against this
# project's own real capture: raw entries look like {"id": "Potion Slot", ...} for an
# empty slot (the literal sentinel string, same as spirecomm's own convention) or
# {"id": "SpeedPotion", ...} for a real potion.
_unmapped_potion_warned = set()


def _map_potion(raw_id: str, log_fn) -> str:
    if raw_id == "Potion Slot":
        return "EMPTY_POTION_SLOT"
    key = _normalize(raw_id)
    for value_name in dir(sts.Potion):
        if value_name.startswith("_"):
            continue
        if _normalize(value_name) == key:
            return value_name
    if raw_id not in _unmapped_potion_warned:
        _unmapped_potion_warned.add(raw_id)
        log_fn(f"[native_recommend] unmapped potion id '{raw_id}' -- treating as empty "
               f"slot (search will not consider using/discarding it)")
    return "EMPTY_POTION_SLOT"


class UnmappedMonsterError(Exception):
    """Raised when a monster's raw id has no entry in _MONSTER_ID_MAP -- unlike an
    unmapped status (skipped, non-fatal), an unmapped monster means the whole
    reconstruction is unreliable, so this propagates and native_recommend's caller
    falls back exactly like an UnmappedContentError from state_mapper would."""


def _map_monster_id(raw_id: str) -> str:
    mapped = _MONSTER_ID_MAP.get(_normalize(raw_id))
    if mapped is None:
        raise UnmappedMonsterError(f"unrecognized monster id {raw_id!r} -- add it to "
                                    f"_MONSTER_ID_MAP in native_recommend.py")
    return mapped


def _live_monster_jsons(monsters_json: list) -> list:
    """SAME filter state_mapper.build_combat_state ALSO uses for its own CombatState
    (kept identical on purpose, even though this module no longer depends on
    state_mapper's output -- state_mapper is still the module a live JSON's
    UnmappedContentError check runs through, so the two must keep agreeing on which
    monsters are "real" for that to mean anything consistent). Returns (json_entry,
    original_index) pairs -- the original index (position in the FULL, unfiltered
    monsters_json array) is what CommunicationMod's own `play <card> <target>`
    command expects, and what _ShadowMonster.json_index carries forward."""
    return [(m, i) for i, m in enumerate(monsters_json) if not (m.get("is_gone") or m.get("current_hp", 0) <= 0)]


def _log_default(msg: str) -> None:
    print(msg, file=sys.stderr)


class _ShadowCard:
    """A hand card, built directly from the raw JSON (which already carries a real
    display .name and .has_target -- no id translation needed for these two fields).
    Deliberately NOT a dataclass/namedtuple: default (identity-based) equality matters
    here, since _build_command's combat.hand.index(card) must find the EXACT object
    this module selected, not merely an equal-looking duplicate (e.g. one of several
    identical Strikes in hand)."""
    def __init__(self, json_entry: dict):
        self.name = json_entry.get("name") or json_entry.get("id", "?")
        self.has_target = bool(json_entry.get("has_target", False))

    def __repr__(self):
        return f"<ShadowCard {self.name}>"


class _ShadowMonster:
    def __init__(self, json_entry: dict, json_index: int):
        self.name = json_entry.get("name") or json_entry.get("id", "?")
        self.json_index = json_index

    def __repr__(self):
        return f"<ShadowMonster {self.name} json_index={self.json_index}>"


class _ShadowCombat:
    """Everything _build_command/the overlay's description need (combat.hand,
    combat.monsters), built directly from the raw JSON. Deliberately independent of
    state_mapper.build_combat_state()/sts.enemies.py's own (smaller) monster roster --
    a real gap was found via this project's own captured live data
    (sts_raw_states.log): 257 of 300 sampled real states were a Taskmaster fight,
    which sts.enemies.py has no class for at all (state_mapper raised
    UnmappedContentError on every one), even though THIS module's own _MONSTER_ID_MAP
    covers it fine (SlaverBoss -> TASKMASTER, this engine's real enum entry). Requiring
    build_combat_state to succeed first was blocking the vast majority of that real
    session's states from ever reaching the native engine at all -- see
    test_native_recommend.py's own real-capture-replay note for how this was found."""
    def __init__(self, hand_json: list, live_monsters_json_with_index: list):
        self.hand = [_ShadowCard(c) for c in hand_json]
        self.monsters = [_ShadowMonster(m, idx) for m, idx in live_monsters_json_with_index]


def native_recommend(combat_state_json: dict, game_state: Optional[dict] = None, log_fn=_log_default):
    """Fully self-sufficient: builds its own lightweight _ShadowCombat directly from
    the raw JSON (see its own docstring for why this doesn't reuse
    state_mapper.build_combat_state), so it never depends on the older sts.enemies.py
    roster's own coverage gaps.

    game_state: the OUTER payload["game_state"] dict (NOT combat_state_json, which is
    game_state["combat_state"]) -- ascension_level and relics both live one level up
    from combat_state_json, so this is needed to read them. Optional (defaults to
    "no relics, ascension 0") purely so direct unit tests / the old real-capture
    replay script (which only ever had combat_state_json handy) still work -- every
    real call from communication_mod.py's handle_state() has the full payload and
    should always pass this.

    Returns (description, action, combat) -- same shape communication_mod.py's
    _try_recommend already returns, so its callers (_build_command/_describe_action's
    replacement, autobattle) don't need to change. Raises on any failure (unmapped
    monster, missing search params, etc.) -- callers already catch broadly and fall
    back to predict.py's v1 mode."""
    _ensure_params_loaded()
    game_state = game_state or {}

    player_json = combat_state_json.get("player", {})
    monsters_json = combat_state_json.get("monsters", [])
    hand_json = combat_state_json.get("hand", [])
    draw_json = combat_state_json.get("draw_pile", [])
    discard_json = combat_state_json.get("discard_pile", [])
    exhaust_json = combat_state_json.get("exhaust_pile", [])
    ascension = game_state.get("ascension_level", 0)
    relic_names = [
        name for name in (_map_relic(r.get("id", ""), log_fn) for r in game_state.get("relics", []))
        if name is not None
    ]
    relics = [getattr(sts.RelicId, name) for name in relic_names]

    player_statuses = []
    for p in player_json.get("powers", []):
        name = _match_status(p.get("id", ""), _PLAYER_STATUS_OVERRIDES, _PLAYER_STATUS_NAMES, log_fn)
        if name is not None:
            player_statuses.append((name, p.get("amount", 1)))

    live_monsters_json_with_index = _live_monster_jsons(monsters_json)
    monster_specs = []
    for live, _json_idx in live_monsters_json_with_index:
        spec = sts.NativeMonsterSpec()
        spec.monster_id_name = _map_monster_id(live.get("id", ""))
        spec.cur_hp = live.get("current_hp", 0)
        spec.max_hp = live.get("max_hp", spec.cur_hp)
        spec.block = live.get("block", 0)
        statuses = []
        for p in live.get("powers", []):
            name = _match_status(p.get("id", ""), _MONSTER_STATUS_OVERRIDES, _MONSTER_STATUS_NAMES, log_fn)
            if name is not None:
                statuses.append((name, p.get("amount", 1)))
        spec.statuses = statuses
        # Monster move NAME mapping is intentionally not attempted here (spirecomm/
        # CommunicationMod's move_id is a per-monster-class-local numeric id with no
        # name attached anywhere in the protocol) -- build_battle_context's own
        # fallback (rolls a plausible move via this engine's own AI model) handles
        # this the same way native_search_agent.py's bridge does.
        monster_specs.append(spec)

    def _cards(entries):
        return [(c.get("id", ""), c.get("upgrades", 0)) for c in entries]

    potion_slots = [getattr(sts.Potion, _map_potion(p.get("id", ""), log_fn))
                    for p in game_state.get("potions", [])]

    bc = sts.build_battle_context(
        player_hp=player_json.get("current_hp", 1), player_max_hp=player_json.get("max_hp", 1),
        player_block=player_json.get("block", 0), player_energy=player_json.get("energy", 3),
        player_statuses=player_statuses, monsters=monster_specs,
        hand_cards=_cards(hand_json), draw_pile_cards=_cards(draw_json),
        discard_pile_cards=_cards(discard_json), exhaust_pile_cards=_cards(exhaust_json),
        potion_slots=potion_slots, relics=relics, turn=combat_state_json.get("turn", 1),
        ascension=ascension, rng_seed=1,
    )
    search_action, _ = sts.run_mcts_search(bc, N_SIMULATIONS)

    combat = _ShadowCombat(hand_json, live_monsters_json_with_index)

    if search_action.action_type == sts.ActionType.END_TURN:
        action = ("end",)
        description = "-> end turn (native)"
        return description, action, combat

    if search_action.action_type != sts.ActionType.CARD:
        raise NotImplementedError(
            f"native_recommend: no translation for action_type={search_action.action_type} "
            f"(potion/card-select actions not wired into this path yet)")

    card = combat.hand[search_action.source_idx]
    target = None
    # target_idx is only meaningful when the chosen card actually targets a monster --
    # bindings-util.cpp still reports a (meaningless) default 0 for untargeted cards,
    # so only resolve a target when the card itself needs one AND the live monster
    # list actually has an entry at that index (same defensive pattern as
    # native_search_agent.py's own _native_action_to_spirecomm).
    if card.has_target and search_action.target_idx < len(combat.monsters):
        target = combat.monsters[search_action.target_idx]
    action = ("play", card, target)
    description = f"-> play {card.name}" + (f" -> {target.name}" if target else "") + "  (native)"
    return description, action, combat
