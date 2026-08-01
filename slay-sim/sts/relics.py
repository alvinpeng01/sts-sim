"""Relics: permanent passive items, one hook system per player (mirrors
powers.py's Power class, but relic-scoped: they never expire or decay
within a fight).

Only combat-scoped effects are modeled -- the simulator has no run-level
state layer yet (no map/rest-site/shop), so any relic whose real effect
crosses combat boundaries is out of scope and simply not implemented:
Burning Blood (post-combat heal), Girya (rest-site charges), Meat on the
Bone (rest-site heal), Molten/Toxic/Frozen Egg (upgrade-on-pickup), Red
Skull (run-wide HP threshold), Preserved Insect (room-type conditional).
That's most of what's cut; everything else is a numbers/scope
simplification noted per relic below.

Some relics carry mutable per-combat counter state (Nunchaku's "cards
played" tally, Akabeko's "used" flag). Like Powers, that state MUST be part
of both Player.clone()'s copy (see creatures.py) and search.py's
transposition-table hash (see search.py's _hashable), or search branches
would corrupt/leak into each other, or silently zero out the cache's hit
rate for any fight with relics (a state-varying attribute that isn't hashed
specially falls back to identity-based hashing, meaning no two clones would
ever compare equal). Relics with NO mutable state (Vajra, Anchor, ...) are
free for the cache -- only the counter/once-per-combat relics fragment it,
exactly as predicted before any of this was built.
"""

from __future__ import annotations

from .powers import Strength, Dexterity, Vulnerable


class Relic:
    name: str = "Relic"

    def __init__(self):
        self.counter = 0
        self.used = False

    def clone(self) -> "Relic":
        new = type(self)()
        new.counter = self.counter
        new.used = self.used
        return new

    # --- hooks the engine calls; default to no-ops ---
    def on_combat_start(self, player, combat) -> None:
        pass

    def on_turn_start(self, player, combat) -> None:
        pass

    def on_turn_end(self, player, combat) -> None:
        pass

    def before_attack_card(self, player, combat) -> None:
        """Fired right before an Attack card's play() resolves -- can set
        combat.pending_damage_bonus/pending_damage_multiplier to affect
        THIS card's damage (Akabeko, Pen Nib)."""
        pass

    def on_attack_played(self, player, combat) -> None:
        """Fired right after an Attack card resolves (counters: Nunchaku,
        Kunai, Shuriken, Ornamental Fan)."""
        pass

    def on_player_hp_loss(self, player, combat, attacker, amount: int) -> None:
        """Fired whenever the player loses HP, from ANY source -- combat
        damage, a card's self-damage, Poison, etc. `attacker` is the
        Monster that dealt it, or None if it wasn't a direct attack."""
        pass

    def __repr__(self) -> str:
        return f"{self.name}"


class Vajra(Relic):
    name = "Vajra"

    def on_combat_start(self, player, combat) -> None:
        player.add_power(Strength(1))


class BagOfMarbles(Relic):
    name = "Bag of Marbles"

    def on_combat_start(self, player, combat) -> None:
        for m in combat.living_monsters:
            m.add_power(Vulnerable(1))


class Anchor(Relic):
    name = "Anchor"

    def on_combat_start(self, player, combat) -> None:
        player.gain_block(10)


class OddlySmoothStone(Relic):
    name = "Oddly Smooth Stone"

    def on_combat_start(self, player, combat) -> None:
        player.add_power(Dexterity(1))


class RingOfTheSnake(Relic):
    """Silent's starting relic."""

    name = "Ring of the Snake"

    def on_combat_start(self, player, combat) -> None:
        combat.draw_cards(2)


class CrackedCore(Relic):
    """Defect's starting relic."""

    name = "Cracked Core"

    def on_combat_start(self, player, combat) -> None:
        from .orbs import make_lightning_orb
        combat.channel_orb(make_lightning_orb())


class PureWater(Relic):
    """Watcher's starting relic. Adds a Miracle (0-cost, +1 Energy,
    exhausts) to hand at the start of combat."""

    name = "Pure Water"

    def on_combat_start(self, player, combat) -> None:
        from .cards import make_miracle
        combat.add_card_to_hand(make_miracle())


class Akabeko(Relic):
    """First Attack played each combat deals 8 additional damage."""

    name = "Akabeko"

    def before_attack_card(self, player, combat) -> None:
        if not self.used:
            self.used = True
            combat.pending_damage_bonus += 8


class BronzeScales(Relic):
    """Thorns 3: whenever the player takes damage, retaliate for 3."""

    name = "Bronze Scales"

    def on_player_hp_loss(self, player, combat, attacker, amount: int) -> None:
        if attacker is not None and not attacker.is_dead:
            attacker.take_damage(3)


class Nunchaku(Relic):
    """Every 10th Attack played this COMBAT (counter never resets), gain 1 Energy."""

    name = "Nunchaku"

    def on_attack_played(self, player, combat) -> None:
        self.counter += 1
        if self.counter % 10 == 0:
            player.energy += 1


class _EveryThirdAttackThisTurn(Relic):
    """Shared pattern for Kunai/Shuriken/Ornamental Fan: every 3rd Attack
    played THIS TURN (counter resets each turn, unlike Nunchaku)."""

    def on_turn_start(self, player, combat) -> None:
        self.counter = 0

    def on_attack_played(self, player, combat) -> None:
        self.counter += 1
        if self.counter % 3 == 0:
            self._on_triggered(player, combat)

    def _on_triggered(self, player, combat) -> None:
        raise NotImplementedError


class Kunai(_EveryThirdAttackThisTurn):
    name = "Kunai"

    def _on_triggered(self, player, combat) -> None:
        player.add_power(Dexterity(1))


class Shuriken(_EveryThirdAttackThisTurn):
    name = "Shuriken"

    def _on_triggered(self, player, combat) -> None:
        player.add_power(Strength(1))


class OrnamentalFan(_EveryThirdAttackThisTurn):
    name = "Ornamental Fan"

    def _on_triggered(self, player, combat) -> None:
        player.gain_block(4)


class Orichalcum(Relic):
    """If you end your turn with no Block, gain 6 Block."""

    name = "Orichalcum"

    def on_turn_end(self, player, combat) -> None:
        if player.block == 0:
            player.gain_block(6)


class Torii(Relic):
    """Unblocked damage of 1-5 is reduced to 1. Implemented as a refund
    after the fact (mathematically equivalent, and avoids threading combat
    context into Creature.take_damage)."""

    name = "Torii"

    def on_player_hp_loss(self, player, combat, attacker, amount: int) -> None:
        if 1 <= amount <= 5:
            player.hp = min(player.max_hp, player.hp + (amount - 1))


class RunicCube(Relic):
    """Whenever you lose HP, for any reason, draw 1 card."""

    name = "Runic Cube"

    def on_player_hp_loss(self, player, combat, attacker, amount: int) -> None:
        combat.draw_cards(1)


class CentennialPuzzle(Relic):
    """The first time you lose HP each combat, draw 3 cards."""

    name = "Centennial Puzzle"

    def on_player_hp_loss(self, player, combat, attacker, amount: int) -> None:
        if not self.used:
            self.used = True
            combat.draw_cards(3)


class PenNib(Relic):
    """Every 10th Attack card played deals double damage."""

    name = "Pen Nib"

    def before_attack_card(self, player, combat) -> None:
        self.counter += 1
        if self.counter % 10 == 0:
            combat.pending_damage_multiplier *= 2.0
