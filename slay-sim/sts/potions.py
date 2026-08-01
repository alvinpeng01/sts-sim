"""Potions: single-use consumables, drunk mid-combat for an immediate effect
(unlike relics, which are always-on).

No community resource gives a rigorous, numeric potion valuation for StS1 --
a background research pass (this session) found only crowd-voted tier lists
(TierMaker) and qualitative Reddit/wiki consensus, no EV-grounded model, and
no published bot/paper that scores potions quantitatively. The heuristic
below is therefore our own construction, informed by that research (in
particular the actual effect magnitudes from the wiki, listed per potion),
not sourced from an existing formula. Treat potion_value()'s output as a
first-pass heuristic to sanity-check and tune against real play, the same
epistemic status as evaluate() in search.py.

Potions aren't wired into legal_actions()/play_card() yet -- there's no
"use potion" action in the engine, and no potion-slot/inventory model on
Player. This module is the valuation piece on its own: given a potion and a
CombatState, how good does drinking it look right now. Wiring an actual
UsePotion action through the engine is a natural follow-up, not done here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from .combat import CombatState


class PotionCategory(Enum):
    DAMAGE = "damage"
    BLOCK = "block"
    BUFF = "buff"
    DEBUFF = "debuff"
    UTILITY = "utility"


@dataclass
class Potion:
    name: str
    category: PotionCategory
    # value(combat) -> float, in [0, ~1.5]: how good drinking this looks
    # right now. Not a strict probability -- see potion_value()'s docstring.
    value: Callable[[CombatState], float]


def _incoming_damage_this_turn(combat: CombatState) -> int:
    return sum(m.intent.damage for m in combat.living_monsters
              if m.intent and m.intent.damage)


def _turns_left_estimate(combat: CombatState, per_turn_damage: float = 8.0) -> float:
    """Crude remaining-fight-length estimate: total living enemy HP divided
    by a flat assumed damage-per-turn. Deliberately simple -- a real
    estimate would use search's own evaluate()/expectimax, which is
    circular if potion_value() is meant to feed evaluation. Clamped to
    [1, 6] since a longer implied fight shouldn't make buffs look
    unboundedly better (this session's next fight also matters)."""
    total_hp = sum(m.hp for m in combat.living_monsters)
    if total_hp <= 0:
        return 1.0
    return max(1.0, min(6.0, total_hp / per_turn_damage))


# --- Fire Potion: 20 flat damage to one enemy (target = lowest-HP living enemy) ---

def _fire_potion_value(combat: CombatState) -> float:
    living = combat.living_monsters
    if not living:
        return 0.0
    target = min(living, key=lambda m: m.hp)
    dmg = min(20, target.hp)
    lethal_bonus = 0.4 if dmg >= target.hp else 0.0  # denies the enemy's whole turn
    # Damage-pool-relative value: bigger swing against a weaker overall
    # board is worth more per point of damage.
    total_hp = max(1, sum(m.hp for m in living))
    return min(1.5, dmg / total_hp * 2.0 + lethal_bonus)


# --- Block Potion: 12 Block ---

def _block_potion_value(combat: CombatState) -> float:
    incoming = _incoming_damage_this_turn(combat)
    needed = max(0, incoming - combat.player.block)
    if needed <= 0:
        return 0.05  # already safe; near-worthless right now
    covered = min(12, needed)
    return min(1.2, covered / max(1, combat.player.hp) * 3.0)


# --- Weak Potion: 3 Weak to all enemies ---

def _weak_potion_value(combat: CombatState) -> float:
    living = combat.living_monsters
    if not living:
        return 0.0
    # Weak reduces the WEAKENED creature's outgoing damage by 25%; value it
    # against the damage those enemies are currently telegraphing, over a
    # few turns of it mattering (duration 3, but only while they keep
    # attacking -- estimate against current intents as a proxy).
    total_intent_dmg = sum(m.intent.damage for m in living if m.intent and m.intent.damage)
    mitigated = total_intent_dmg * 0.25 * min(3, _turns_left_estimate(combat))
    return min(1.2, mitigated / max(1, combat.player.hp))


# --- Fear Potion: 3 Vulnerable to one enemy ---

def _fear_potion_value(combat: CombatState) -> float:
    living = combat.living_monsters
    if not living:
        return 0.0
    target = max(living, key=lambda m: m.hp)  # biggest remaining threat
    # Vulnerable's 50% extra damage taken is worth more the more damage
    # you're about to deal it and the more turns it has left to matter.
    turns = min(3, _turns_left_estimate(combat))
    return min(1.2, 0.5 * turns * (target.hp / max(1, sum(m.hp for m in living))))


# --- Strength/Dexterity Potion: +2 stat for the rest of combat ---

def _stat_potion_value(per_stack_worth: float):
    def _value(combat: CombatState) -> float:
        turns = _turns_left_estimate(combat)
        gain = 2 * per_stack_worth * turns
        return min(1.3, gain / max(1, sum(m.hp for m in combat.living_monsters) or 1))
    return _value


# --- Speed Potion: +5 Str & +5 Dex, lost at end of turn (burst only) ---

def _speed_potion_value(combat: CombatState) -> float:
    # Fixed 1-turn window regardless of how long the fight might run.
    dmg_swing = 5 * 1.0  # rough: 5 Strength ~ 5 extra damage per attack this turn
    block_swing = 5 * 1.0
    living = combat.living_monsters
    total_hp = max(1, sum(m.hp for m in living)) if living else 1
    return min(1.2, (dmg_swing / total_hp) + (block_swing / max(1, combat.player.hp)))


# --- Energy Potion: +2 Energy ---

def _energy_potion_value(combat: CombatState) -> float:
    # Worth more the more energy-starved / the bigger the hand.
    hand_playable_value = len(combat.hand) / 5.0
    return min(1.0, 0.3 + 0.4 * hand_playable_value)


# --- Fairy in a Bottle: passive death insurance, revive at 30% max HP ---

def _fairy_bottle_value(combat: CombatState) -> float:
    p = combat.player
    hp_frac = p.hp / max(1, p.max_hp)
    incoming = _incoming_damage_this_turn(combat)
    near_death = incoming >= p.hp + p.block
    # Spikes hard near death; otherwise it's just held (low but nonzero,
    # since it's never "wasted" by holding -- passive, no action needed).
    base = 0.15
    if hp_frac < 0.3 or near_death:
        base = 1.4
    elif hp_frac < 0.5:
        base = 0.6
    return base


POTION_VALUE_FNS = {
    "Fire Potion": _fire_potion_value,
    "Block Potion": _block_potion_value,
    "Weak Potion": _weak_potion_value,
    "Fear Potion": _fear_potion_value,
    "Strength Potion": _stat_potion_value(1.0),
    "Dexterity Potion": _stat_potion_value(1.0),
    "Speed Potion": _speed_potion_value,
    "Energy Potion": _energy_potion_value,
    "Fairy in a Bottle": _fairy_bottle_value,
}


def potion_value(potion_name: str, combat: CombatState) -> float:
    """Heuristic "how good does drinking this potion look right now",
    roughly in [0, 1.5] (not a probability -- see module docstring for what
    this is and isn't grounded in). Raises KeyError for potions without an
    implemented heuristic (Attack/Skill/Power Potion, Ambrosia, Entropic
    Brew, Liquid Bronze -- effect categories not modeled: adding a random
    card of a type to hand, stance-switching, potion-slot refills, and
    Thorns respectively)."""
    fn = POTION_VALUE_FNS[potion_name]
    return fn(combat)
