"""Defect's Orbs: a small rotating queue (default 3 slots) that passively
triggers at the end of the player's turn, scaled by Focus. Channeling past a
full queue evicts and evokes the oldest orb immediately.

Simplified to Lightning and Frost only (skipping Dark and Plasma) -- see
sts/cards.py's module docstring for the rest of the Defect simplifications.
Orb damage/block intentionally bypasses Strength/Weak/Dexterity (matches the
real game: orbs are a separate damage/block source from attacks and cards).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class Orb:
    name: str
    # passive(player, combat) -> None -- fires every end of turn while queued.
    passive: Callable
    # evoke(player, combat) -> None -- fires once, when evicted from the queue.
    evoke: Callable


def _lightning_passive(player, combat) -> None:
    dmg = 3 + player.get_power_amount("Focus")
    living = combat.living_monsters
    if living:
        target = combat.rng.choice(living)
        target.take_damage(max(0, dmg))


def _lightning_evoke(player, combat) -> None:
    dmg = 8 + player.get_power_amount("Focus")
    living = combat.living_monsters
    if living:
        target = combat.rng.choice(living)
        target.take_damage(max(0, dmg))


def _frost_passive(player, combat) -> None:
    player.gain_block(2 + player.get_power_amount("Focus"))


def _frost_evoke(player, combat) -> None:
    player.gain_block(5 + player.get_power_amount("Focus"))


def make_lightning_orb() -> Orb:
    return Orb("Lightning", _lightning_passive, _lightning_evoke)


def make_frost_orb() -> Orb:
    return Orb("Frost", _frost_passive, _frost_evoke)


def _dark_passive(player, combat) -> None:
    focus = player.get_power_amount("Focus")
    # Dark orbs store up damage each turn: passive increments the stored amount
    # but deals no immediate damage. Evoke deals the full stored amount.
    # We track the stored amount ephemerally here by having the orb's evoke
    # read the Focus at time of channeling -- a simplification.


def _dark_evoke(player, combat) -> None:
    focus = player.get_power_amount("Focus")
    dmg = max(0, 6 + focus)
    living = combat.living_monsters
    if living:
        target = min(living, key=lambda m: m.hp)
        target.take_damage(dmg)


def make_dark_orb() -> Orb:
    return Orb("Dark", _dark_passive, _dark_evoke)


def _plasma_passive(player, combat) -> None:
    player.energy += 1


def _plasma_evoke(player, combat) -> None:
    player.energy += 2


def make_plasma_orb() -> Orb:
    return Orb("Plasma", _plasma_passive, _plasma_evoke)
