"""Creatures: the shared base for the Player and Monsters.

Holds HP, block, and the power (status-effect) bag, plus the damage/block math
that both attacks and defends route through. Getting this math right once here
means every card and enemy inherits correct behavior.
"""

from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional

from .powers import Power, Strength, Dexterity, Vulnerable, Weak, Frail
from .orbs import Orb
from .relics import Relic


class Creature:
    def __init__(self, name: str, max_hp: int):
        self.name = name
        self.max_hp = max_hp
        self.hp = max_hp
        self.block = 0
        self.powers: Dict[str, Power] = {}
        # Strength granted this turn that must be clawed back at end of turn
        # (e.g. Flex). Separate from the permanent Strength stack it adds to.
        self.temp_strength_pending = 0
        # Watcher's stance: None, "Calm", or "Wrath". Lives here (not on a
        # Player subclass) so the shared damage math below can apply Wrath's
        # double-damage-dealt-and-taken uniformly; unused by every other
        # class/monster.
        self.stance: Optional[str] = None

    # --- power management ---
    def add_power(self, power: Power) -> None:
        # Artifact: negates the next debuff application entirely (checked
        # here since every debuff-application call site already goes
        # through this one method -- the only centralized hook point that
        # exists for this). Covers Weak/Vulnerable/Frail (Power.is_debuff),
        # the common case; other, rarer negative powers applied via
        # add_power aren't marked is_debuff and so aren't blocked --
        # flagged rather than guessing which of every Power subclass in the
        # real game Artifact does/doesn't cover.
        if getattr(power, "is_debuff", False) and self.has_power("Artifact"):
            artifact = self.powers["Artifact"]
            artifact.amount -= 1
            if artifact.amount <= 0:
                del self.powers["Artifact"]
            return

        existing = self.powers.get(power.name)
        if existing is not None:
            existing.amount += power.amount
        else:
            self.powers[power.name] = power

    def get_power_amount(self, name: str) -> int:
        p = self.powers.get(name)
        return p.amount if p else 0

    def has_power(self, name: str) -> bool:
        return self.get_power_amount(name) > 0

    # --- combat math ---
    def calc_attack_damage(self, base: int, strength_multiplier: int = 1) -> int:
        """Damage this creature deals with a ``base``-damage attack.

        Order (matches StS): add Strength, then Weak (x0.75), then the
        target's Vulnerable is applied later in :meth:`take_damage`.

        ``strength_multiplier`` is 1 for virtually every card -- Heavy
        Blade is the one real exception ("Strength affects this card 3(5)
        times" instead of the usual 1), so rather than duplicate this
        whole method just for that one card, it takes the multiplier as a
        parameter with a default that leaves every other call site
        unaffected.
        """
        dmg = base + self.get_power_amount("Strength") * strength_multiplier
        if self.has_power("Weak"):
            dmg = Weak.modify_damage_dealt(dmg)
        if self.stance == "Wrath":
            dmg *= 2
        if self.stance == "Divinity":
            dmg *= 3
        return max(0, dmg)

    def calc_block_gain(self, base: int) -> int:
        block = base + self.get_power_amount("Dexterity")
        if self.has_power("Frail"):
            block = Frail.modify_block_gained(block)
        return max(0, block)

    def gain_block(self, base: int) -> None:
        self.block += self.calc_block_gain(base)

    def add_temp_strength(self, amount: int) -> None:
        """Gain Strength that expires at the end of this turn (Flex)."""
        self.add_power(Strength(amount))
        self.temp_strength_pending += amount

    def take_damage(self, incoming: int) -> int:
        """Apply ``incoming`` attack damage (already Strength/Weak-adjusted).

        Vulnerable is applied here (it scales damage *received*), then block
        absorbs, then HP drops. Returns HP actually lost.
        """
        dmg = incoming
        if self.has_power("Vulnerable"):
            dmg = Vulnerable.modify_damage_taken(dmg)
        if self.stance == "Wrath":
            dmg *= 2
        if self.stance == "Divinity":
            dmg *= 3
        if self.has_power("Intangible") and dmg > 1:
            dmg = 1

        absorbed = min(self.block, dmg)
        self.block -= absorbed
        hp_loss = dmg - absorbed
        self.hp = max(0, self.hp - hp_loss)
        return hp_loss

    def lose_hp(self, amount: int) -> None:
        """Direct HP loss that ignores block (e.g. self-damage effects)."""
        if self.has_power("Intangible") and amount > 1:
            amount = 1
        self.hp = max(0, self.hp - amount)

    def heal(self, amount: int) -> None:
        """Direct HP gain, capped at max_hp (Bandage Up, Shelled Parasite's
        Life Suck)."""
        self.hp = min(self.max_hp, self.hp + amount)

    @property
    def is_dead(self) -> bool:
        return self.hp <= 0

    def start_turn(self, combat) -> None:
        for power in list(self.powers.values()):
            power.on_start_turn(self, combat)

    def end_turn(self, combat) -> None:
        for power in list(self.powers.values()):
            power.on_end_turn(self, combat)
        if self.temp_strength_pending:
            strength = self.powers.get("Strength")
            if strength is not None:
                strength.amount -= self.temp_strength_pending
                if strength.amount <= 0:
                    del self.powers["Strength"]
            self.temp_strength_pending = 0
        # Powers that last exactly one turn (Rage) are removed outright,
        # regardless of their stack amount.
        for name in list(self.powers.keys()):
            if self.powers[name].expires_end_of_turn:
                del self.powers[name]
        # Duration powers tick down at the end of the owner's turn.
        for name in list(self.powers.keys()):
            power = self.powers[name]
            if power.is_duration:
                power.amount -= 1
                if power.amount <= 0:
                    del self.powers[name]

    def status_str(self) -> str:
        powers = " ".join(str(p) for p in self.powers.values())
        return f"{self.name}: {self.hp}/{self.max_hp} HP, {self.block} block" + (
            f" | {powers}" if powers else ""
        )

    def clone(self):
        """Fast clone for search to branch on. Manually calling __new__ +
        copying __dict__ handles every plain field (ints, strs, the Intent
        dataclass, monster-subclass AI state like last_move/turn_count/
        asleep/mode) in O(1) -- deliberately generic so any field a Monster
        subclass adds is automatically covered without touching this method.
        The `powers` dict is the only field that needs real (still shallow --
        Power objects hold no nested state) copying, since a shared __dict__
        reference would otherwise alias the same dict object between clones.

        NOT copy.copy(self), despite that looking equivalent (and having
        been the original implementation here) -- measured via cProfile on
        a real expectimax search: copy.copy() on a plain object without a
        __copy__/__reduce__ override doesn't just do __dict__.copy(), it
        goes through Python's generic pickle-protocol-based reconstruction
        machinery (__reduce_ex__ -> copyreg.__newobj__ -> _reconstruct),
        which showed up as a genuinely large chunk of total search time.
        A microbenchmark confirmed the gap directly: manual __new__ + dict
        copy measured ~4.5x faster than copy.copy() for an equivalent
        object (89ms vs 405ms for 200k clones). This replaces the earlier
        copy.deepcopy fix for the same underlying lesson -- don't trust a
        generic stdlib convenience function's cost without measuring it,
        even when it looks like it should reduce to the cheap path."""
        new = self.__class__.__new__(self.__class__)
        new.__dict__.update(self.__dict__)
        new.powers = {name: type(p)(p.amount) for name, p in self.powers.items()}
        return new


class Player(Creature):
    def __init__(self, name: str = "Ironclad", max_hp: int = 80, max_energy: int = 3):
        super().__init__(name, max_hp)
        self.max_energy = max_energy
        self.energy = 0
        # Defect's orb queue; unused (stays empty) by every other class.
        self.orbs: List[Orb] = []
        self.orb_slots = 3
        # Fixed for the whole combat (no run layer yet to pick these up at
        # a shop/reward), but individual relics can carry mutable per-combat
        # counter state (Nunchaku, Akabeko, ...) that DOES need real cloning.
        self.relics: List[Relic] = []
        # Blood for Blood's dynamic cost ("costs 1 less for each time you
        # lose HP this combat") is the only thing that needs this -- a
        # plain int field, so Creature.clone()'s generic __dict__ copy
        # already handles it correctly with no changes there.
        self.hp_loss_count_this_combat = 0
        # Panic Button's "cannot gain Block from cards for 2 turns" --
        # counts down in CombatState.start_player_turn, checked/consumed
        # here in gain_block (overridden below).
        self.no_block_from_cards_turns = 0
        # Per-combat Defect tracking counters -- per-instance (cloned with
        # the Player), so they don't leak across search branches the way
        # module-level globals would.  Claw's bonus is tracked per-card via
        # card-level mutable state (closure-over-dict), not here.
        self.lightning_channeled = 0
        self.frost_channeled = 0
        self.orb_channeled_count = 0
        self.powers_played = 0
        # Watcher's Mantra counter (Devotion / Pray / Prostrate / Worship).
        self._mantra = 0
        # Blasphemy: die at start of next turn (set by Blasphemy card).
        self._blasphemer = False

    def gain_block(self, base: int) -> None:
        if self.no_block_from_cards_turns > 0:
            return
        super().gain_block(base)

    def take_damage(self, incoming: int) -> int:
        hp_loss = super().take_damage(incoming)
        if hp_loss > 0:
            self.hp_loss_count_this_combat += 1
        return hp_loss

    def lose_hp(self, amount: int) -> None:
        if amount > 0:
            self.hp_loss_count_this_combat += 1
        super().lose_hp(amount)

    def clone(self) -> "Player":
        new = super().clone()
        # Orb objects are stateless/immutable once created (see orbs.py), so
        # sharing them by reference is safe -- only the list itself, which
        # channel_orb() mutates, needs to be a distinct object per clone.
        new.orbs = list(self.orbs)
        # Relics, unlike Orbs, carry mutable state -- each needs its own
        # clone(), not just a fresh list of shared references.
        new.relics = [r.clone() for r in self.relics]
        return new
