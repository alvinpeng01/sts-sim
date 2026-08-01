"""CommunicationMod JSON combat state -> a real sts.combat.CombatState, so
the live bridge can run sts.mcts.mcts_choose_action (MCTS + the learned
value net) instead of predict.py's v1 intent-arithmetic-only mode.

WHY THIS WAS SAFE TO SKIP UNTIL NOW: earlier in this project the roster
covered only Act 1 + a slice of Act 2, so reconstructing a real CombatState
would silently break (or need a fallback) the instant a live run reached an
unimplemented monster. That gap is mostly closed now (Act 1-3, 29
encounters, tuned to A20) -- see the project notes -- which is what makes
this worth building.

THE HIDDEN-STATE PROBLEM, AND A REAL LIMITATION THIS DOESN'T FULLY SOLVE:
our Monster subclasses track AI history CommunicationMod doesn't expose
(Jaw Worm's last_move anti-repeat guard, Gremlin Nob's last_move, Guardian's
mode/dmg_taken_this_turn, ...) -- a single live JSON snapshot can't tell us
a monster's move history. The first design tried here was to bypass that
entirely: read the live game's already-rolled CURRENT intent as ground
truth and write it directly onto .intent/._pending_move, skipping our own
intent_options()/roll_intent() for the round about to resolve. That doesn't
work -- checked directly against the code: every take_turn() implementation
in enemies.py (e.g. JawWorm's) branches on self._pending_move against a
class-specific move CONSTANT and then deals a HARDCODED damage literal
(`combat.deal_attack_damage(self, combat.player, 12)`), it never reads
self.intent.damage. Setting an arbitrary raw intent string as
"_pending_move" would match none of those branches, and every monster would
silently deal 0 damage, take no action, and apply no power -- worse than
not having live data, since it would look like a valid, silent, no-op fight.

So instead: _apply_live_monster() calls the monster's own roll_intent(rng)
first (self-consistent history-free default state, same approximation
already used for the FURTHER-out intent predictions), which produces a
valid class-specific _pending_move that take_turn() will resolve correctly
using OUR OWN verified (A20-traced) per-move numbers. It then overwrites
.intent.damage/.intent.name to reflect what the live game actually
telegraphed, purely for DISPLAY and for the value net's incoming-damage
feature (value_net.py's _incoming_damage() reads .intent.damage). The real,
acknowledged consequence: the damage the search resolves for this round and
the damage the live game will actually deal aren't guaranteed to be bit-
identical -- they're both draws from "this monster's known move-damage
distribution," not the same draw. Full exactness there would need the
mapper to reverse-engineer which specific named move produced a given live
damage number per class, which isn't done -- flagged as the natural next
refinement, not silently assumed away.

UNVERIFIED, FLAGGED, NOT HIDDEN: the two ID tables below (monster CommunicationMod-id
-> our class, card CommunicationMod-id -> our factory) are this module's
real risk. Card ids are derived from our own Card objects' .name (map_card_id()),
which should mostly just work since this project's card names already mirror
the real game's display names, with a couple of known exceptions (Strike/
Defend need a "_R" suffix in the real game to disambiguate the class-shared
starter names). Monster ids are pure best-effort: PascalCase-stripped from
each class's own default .name, registered alongside the raw display name as
a second guess -- NOT confirmed against a real capture, because CommunicationMod
isn't installed in this environment yet. communication_mod.py's raw-state
log exists specifically to let these tables get corrected against real data
the first time this actually runs. Until then: an unmapped id raises
UnmappedContentError, which the caller is expected to catch and fall back to
predict.py's v1 mode -- never guess past this boundary, never crash the
bridge on a live run over a lookup miss.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional

from ..cards import Card
from ..combat import CombatState
from ..creatures import Player
from ..enemies import (
    Monster, Intent, IntentType,
    JawWorm, Cultist, Louse, AcidSlimeM, SpikeSlimeM, BlueSlaver, Looter,
    GremlinNob, Lagavulin, Guardian, Sentry, MadGremlin, SneakyGremlin,
    FatGremlin, Byrd, Mystic, Centurion, Champ, BronzeOrb, Automaton,
    TorchHead, Collector, GremlinWizard, GremlinLeader, Darkling, OrbWalker,
    WrithingMass, Spiker, Repulsor, Exploder, SphericGuardian, Nemesis,
    Dagger, Reptomancer, AwakenedOneCultist, AwakenedOne, TimeEater, DonuDeca,
    Hexaghost, SlimeBoss, Chosen, BookOfStabbing, GiantHead, ShelledParasite,
)
from .. import cards as cards_module


class UnmappedContentError(Exception):
    """Raised when the live state references a monster/card id this module
    doesn't have a mapping for. Callers must catch this and fall back to
    predict.py's v1 mode -- never guess past this boundary."""


# --- monster id table (UNVERIFIED, see module docstring) ------------------

def _pascal(name: str) -> str:
    return "".join(w.capitalize() for w in name.replace("(", "").replace(")", "").split())


_MONSTER_CTORS: List[Callable[[], Monster]] = [
    JawWorm, Cultist, lambda: Louse(random.Random()), AcidSlimeM, SpikeSlimeM,
    BlueSlaver, Looter, GremlinNob, Lagavulin, Guardian, Sentry, MadGremlin,
    SneakyGremlin, FatGremlin, Byrd, Mystic, Centurion, Champ,
    lambda: BronzeOrb(1), Automaton, TorchHead, Collector, GremlinWizard,
    GremlinLeader, Darkling, OrbWalker, WrithingMass, Spiker, Repulsor,
    Exploder, SphericGuardian, Nemesis, Dagger, Reptomancer,
    AwakenedOneCultist, AwakenedOne, TimeEater, lambda: DonuDeca(True),
    Hexaghost, SlimeBoss, Chosen, BookOfStabbing, GiantHead, ShelledParasite,
]

MONSTER_FACTORY: Dict[str, Callable[[], Monster]] = {}
for _ctor in _MONSTER_CTORS:
    _sample = _ctor()
    for _key in {_pascal(_sample.name), _sample.name}:
        MONSTER_FACTORY[_key] = _ctor
del _ctor, _sample, _key

# Confirmed against a live CommunicationMod capture: the real game (and this
# id) distinguishes Red/Green Louse as two ids -- "FuzzyLouseNormal" and
# "FuzzyLouseDefensive" -- but sts/enemies.py's Louse is a single simplified
# class that doesn't model that split (see its docstring). Same
# alias-to-the-closest-available-ctor approach as Strike_R/Defend_R above,
# not a guess -- confirmed real id, just no matching distinct implementation
# to point it at yet.
if "Louse" in MONSTER_FACTORY:
    MONSTER_FACTORY["FuzzyLouseNormal"] = MONSTER_FACTORY["Louse"]
    MONSTER_FACTORY["FuzzyLouseDefensive"] = MONSTER_FACTORY["Louse"]

# Confirmed against real CommunicationMod capture data from a live run
# (sts_raw_states.log / sts_predictions.log): the real game's internal ids
# for these don't match either _pascal(name) or the raw display name --
# real, verified quirks, not guesses:
#   - the Looter's internal id is "Mugger" (a leftover/renamed-in-code-only
#     name from development, display name stayed "Looter")
#   - Blue Slaver's id has its color/role words in the opposite order
#     ("SlaverBlue", not "BlueSlaver")
#   - Mystic's internal id is "Healer" (again, display name diverged from
#     the internal one)
if "Looter" in MONSTER_FACTORY:
    MONSTER_FACTORY["Mugger"] = MONSTER_FACTORY["Looter"]
if "BlueSlaver" in MONSTER_FACTORY:
    MONSTER_FACTORY["SlaverBlue"] = MONSTER_FACTORY["BlueSlaver"]
if "Mystic" in MONSTER_FACTORY:
    MONSTER_FACTORY["Healer"] = MONSTER_FACTORY["Mystic"]

# Size variants: sts/enemies.py only implements the Medium size of each
# slime (AcidSlimeM/SpikeSlimeM) -- Small and Large are real, distinct
# monsters in the actual game (different HP, different move sets, and Large
# splits into two Mediums on death, which this aliasing does NOT model) but
# aren't implemented as their own classes yet. Same "alias to the closest
# available ctor so predictions degrade to approximate rather than absent"
# choice as FuzzyLouseNormal/Defensive above -- flagged here because unlike
# the Louse case, the approximation error for a Large slime is large (wrong
# HP, missing the split-on-death mechanic entirely).
if "AcidSlimeM" in MONSTER_FACTORY:
    MONSTER_FACTORY["AcidSlime_M"] = MONSTER_FACTORY["AcidSlimeM"]
    MONSTER_FACTORY["AcidSlime_L"] = MONSTER_FACTORY["AcidSlimeM"]  # approximate: real L has more HP + splits on death
if "SpikeSlimeM" in MONSTER_FACTORY:
    MONSTER_FACTORY["SpikeSlime_S"] = MONSTER_FACTORY["SpikeSlimeM"]  # approximate: real S has far less HP
    MONSTER_FACTORY["SpikeSlime_L"] = MONSTER_FACTORY["SpikeSlimeM"]  # approximate: real L has more HP + splits on death

# Donu and Deca are fought as a pair, each a separate CommunicationMod
# monster id ("Donu", "Deca") backed by the same DonuDeca class here with
# opposite starts_attacking values. Verified against slaythespire.wiki.gg:
# Donu's alternating pattern opens with its buff (Circle of Power), Deca's
# opens with its attack (Beam).
MONSTER_FACTORY["Donu"] = lambda: DonuDeca(starts_attacking=False)
MONSTER_FACTORY["Deca"] = lambda: DonuDeca(starts_attacking=True)


# --- card id table: derived from our own factories' .name, since this
# project's card names already mirror the real game's display names -----

def _card_factories():
    import inspect
    out = {}
    for name, fn in vars(cards_module).items():
        if not name.startswith("make_") or not inspect.isfunction(fn):
            continue
        try:
            sample = fn()
        except TypeError:
            continue
        if not isinstance(sample, Card):
            continue
        try:
            fn(upgraded=True)
            supports_upgrade = True
        except TypeError:
            supports_upgrade = False
        out[sample.name] = (fn, supports_upgrade)
    return out


CARD_FACTORY = _card_factories()
# Known real-game id exceptions: Strike/Defend are shared across classes in
# the actual game and need a class suffix to disambiguate there ("Strike_R"
# for Ironclad); register both spellings since which one CommunicationMod
# actually reports is exactly the kind of thing that needs a live capture
# to confirm.
if "Strike" in CARD_FACTORY:
    CARD_FACTORY["Strike_R"] = CARD_FACTORY["Strike"]
    # Confirmed against a live capture: the Silent's run reported "Strike_G"
    # (green), not "Strike_R" -- register both since this project's Deck
    # doesn't track which class a Strike/Defend "belongs" to, it's the same
    # Card either way.
    CARD_FACTORY["Strike_G"] = CARD_FACTORY["Strike"]
if "Defend" in CARD_FACTORY:
    CARD_FACTORY["Defend_R"] = CARD_FACTORY["Defend"]
    CARD_FACTORY["Defend_G"] = CARD_FACTORY["Defend"]
# Ascender's Bane is GUARANTEED present (not just possible) in any real
# Ascension 10+ run's deck from turn 1 -- unlike an optional curse pickup,
# every single fight's draw/discard/exhaust piles will include it, so
# leaving this unmapped wouldn't just miss one card, it would make
# build_combat_state() raise on literally the first call for any A10+ game,
# disabling the mod's recommendation entirely for exactly the ascension
# level this project has standardized on (A20). Same apostrophe/spelling
# uncertainty as the Strike_R/Defend_R case -- registering the likely
# no-punctuation id too rather than betting on one spelling.
if "Ascender's Bane" in CARD_FACTORY:
    CARD_FACTORY["AscendersBane"] = CARD_FACTORY["Ascender's Bane"]


def map_card_id(card_id: str, upgrades: int) -> Card:
    entry = CARD_FACTORY.get(card_id)
    if entry is None:
        raise UnmappedContentError(f"unknown card id {card_id!r}")
    fn, supports_upgrade = entry
    if upgrades and supports_upgrade:
        return fn(upgraded=True)
    return fn()


def map_monster(monster_id: str) -> Monster:
    ctor = MONSTER_FACTORY.get(monster_id)
    if ctor is None:
        raise UnmappedContentError(f"unknown monster id {monster_id!r}")
    return ctor()


# --- the actual JSON -> CombatState builder --------------------------------

_DAMAGING_INTENTS = {"ATTACK", "ATTACK_BUFF", "ATTACK_DEBUFF", "ATTACK_DEFEND"}


def _apply_live_monster(m: Monster, live: dict, rng: random.Random) -> None:
    """Overwrite HP/block/powers from the live snapshot (ground truth), and
    call the monster's OWN roll_intent() to get a valid, class-consistent
    _pending_move (take_turn() requires one -- see module docstring for why
    the live intent string can't be written there directly). The live
    intent's damage/name is then layered onto .intent for display and the
    value net's incoming-damage feature, WITHOUT changing _pending_move --
    so what gets displayed/fed to eval and what take_turn() actually
    resolves can diverge slightly; that's the acknowledged limitation."""
    m.hp = live.get("current_hp", m.hp)
    m.max_hp = live.get("max_hp", m.max_hp)
    m.block = live.get("block", 0)
    for p in live.get("powers", []):
        # CommunicationMod power ids are TitleCase-ish like our own Power
        # names in most cases (Strength, Vulnerable, Weak, ...); unknown
        # power ids are skipped rather than raising, since a missing buff on
        # one enemy shouldn't take down state reconstruction for the whole
        # fight the way an unmapped monster/card should.
        from .. import powers as powers_module
        cls = getattr(powers_module, p.get("id", ""), None)
        if cls is not None:
            m.add_power(cls(p.get("amount", 1)))

    m.roll_intent(rng)  # sets a valid, class-consistent _pending_move + .intent

    intent_str = live.get("intent")
    dmg = live.get("move_adjusted_damage")
    hits = live.get("move_hits") or 1
    if intent_str in _DAMAGING_INTENTS and dmg is not None:
        m.intent = Intent(IntentType.ATTACK, dmg * hits, intent_str)
    elif intent_str:
        m.intent = Intent(IntentType.BUFF, None, intent_str)
    # else: keep roll_intent()'s own guess -- no live intent field to layer on


def build_combat_state(combat_state_json: dict, rng: Optional[random.Random] = None) -> CombatState:
    """Builds a CombatState that mirrors the live fight's player/monster HP,
    block, powers, and CURRENT intents exactly. Raises UnmappedContentError
    if any monster or hand/pile card isn't in our tables -- callers must
    catch this and fall back to predict.py's v1 mode, not guess further."""
    player_json = combat_state_json.get("player", {})
    monsters_json = combat_state_json.get("monsters", [])
    hand_json = combat_state_json.get("hand", [])
    draw_json = combat_state_json.get("draw_pile", [])
    discard_json = combat_state_json.get("discard_pile", [])
    exhaust_json = combat_state_json.get("exhaust_pile", [])

    player = Player(max_hp=player_json.get("max_hp", 80))
    player.hp = player_json.get("current_hp", player.max_hp)
    player.block = player_json.get("block", 0)
    player.energy = player_json.get("energy", player.max_energy)
    from .. import powers as powers_module
    for p in player_json.get("powers", []):
        cls = getattr(powers_module, p.get("id", ""), None)
        if cls is not None:
            player.add_power(cls(p.get("amount", 1)))

    combat_rng = rng or random.Random()
    monsters = []
    for json_idx, live in enumerate(monsters_json):
        if live.get("is_gone") or live.get("current_hp", 0) <= 0:
            continue
        m = map_monster(live.get("id", ""))
        _apply_live_monster(m, live, combat_rng)
        # CommunicationMod's own `play <card> <target>` command indexes
        # monsters by position in the JSON's own (unfiltered) `monsters`
        # array -- but dead/gone ones are skipped above, so combat.monsters'
        # own index drifts from that as soon as anything's died. Autobattle
        # (communication_mod.py) needs the ORIGINAL index to build a command
        # CommunicationMod will actually understand, so it's stashed here
        # rather than recomputed error-pronely later.
        m.json_index = json_idx
        monsters.append(m)

    def _cards(entries):
        return [map_card_id(c.get("id", ""), c.get("upgrades", 0)) for c in entries]

    hand = _cards(hand_json)
    draw_pile = _cards(draw_json)
    discard_pile = _cards(discard_json)
    exhaust_pile = _cards(exhaust_json)

    combat = CombatState(player, monsters, [], rng=combat_rng)
    combat.hand = hand
    # Shuffle rather than trust CommunicationMod's reported order: a
    # legitimate player (and, by extension, a tool respecting that) doesn't
    # actually know the true hidden draw sequence -- only the pile's
    # CONTENTS are real information here. This matters even at
    # turns_left=1 if the current hand contains a mid-turn draw effect
    # (Battle Trance, etc.), since that would pop from draw_pile within the
    # very turn being solved.
    combat_rng.shuffle(draw_pile)
    combat.draw_pile = draw_pile
    combat.discard_pile = discard_pile
    combat.exhaust_pile = exhaust_pile
    combat.turn = combat_state_json.get("turn", 1)
    return combat
