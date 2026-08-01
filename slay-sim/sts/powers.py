"""Powers (a.k.a. buffs/debuffs/status effects).

Each Power is a small object holding an ``amount`` (its stack count) plus hooks
the combat engine calls at the right moments. Keeping the logic on the power
object is what makes the engine extensible: adding a new status effect means
writing one class here, not editing the combat loop.

Damage/block math intentionally uses explicit, ordered application in
``creatures.py`` rather than iterating powers in arbitrary order, because
Slay the Spire's order matters (Strength is additive and applies *before*
the multiplicative Weak/Vulnerable). When we add more modifying powers we can
promote that into a priority-sorted hook pipeline; for now explicit is clearer.
"""

from __future__ import annotations

import math


class Power:
    name: str = "Power"
    # Duration powers lose 1 stack at the end of the owner's turn (Weak,
    # Vulnerable). Non-duration powers persist (Strength, Dexterity).
    is_duration: bool = False
    # Removed outright at end of turn regardless of amount -- for powers that
    # last exactly one turn rather than decaying by 1 stack (Rage).
    expires_end_of_turn: bool = False
    # Whether Artifact negates this power on application -- see
    # Creature.add_power. Only the 3 common debuffs are marked; see
    # Artifact's own docstring for why that's a deliberate, flagged
    # simplification rather than an exhaustive real-game mapping.
    is_debuff: bool = False

    def __init__(self, amount: int = 1):
        self.amount = amount

    # --- hooks the engine calls; default to no-ops ---
    def on_start_turn(self, owner, combat) -> None:
        pass

    def on_end_turn(self, owner, combat) -> None:
        pass

    def on_exhaust(self, owner, combat) -> None:
        """Fired whenever the owner exhausts a card (Dark Embrace, Feel No Pain)."""
        pass

    def on_play_attack(self, owner, combat) -> None:
        """Fired after the owner plays an Attack card (Rage)."""
        pass

    def on_draw_status(self, owner, combat) -> None:
        """Fired when the owner draws a Status card (Evolve, Fire Breathing)."""
        pass

    def on_hp_loss_from_card(self, owner, combat) -> None:
        """Fired when the owner loses HP as a direct card effect, not combat
        damage (Rupture) -- see CombatState.player_loses_hp_from_card."""
        pass

    def on_player_attacked(self, attacker, combat) -> None:
        """Fired whenever the player is the target of a monster's attack
        (Flame Barrier) -- regardless of whether any damage got through
        block, since the real card's text is "whenever you are attacked",
        not "whenever you lose HP". See CombatState.deal_attack_damage."""
        pass

    def on_any_card_played(self, owner, combat, card) -> None:
        """Fired whenever the owner plays any card (Thousand Cuts, After Image)."""
        pass

    def on_power_played(self, owner, combat) -> None:
        """Fired whenever the owner plays a Power card (Storm, Heatsinks)."""
        pass

    def on_stance_change(self, owner, combat, new_stance, old_stance) -> None:
        """Fired when the owner's stance changes (Rushdown, Mental Fortress)."""
        pass

    def __repr__(self) -> str:
        return f"{self.name}({self.amount})"


class Strength(Power):
    name = "Strength"
    is_duration = False


class Dexterity(Power):
    name = "Dexterity"
    is_duration = False


class Juggernaut(Power):
    """Whenever you gain Block, deal `amount` damage to a random enemy.
    Checked in CombatState.play_card (a before/after diff on player.block
    around card.play()), not a hook on gain_block() itself -- gain_block()
    isn't given a `combat` reference by any of its ~20+ existing call sites
    across cards.py, and adding one to all of them just for this one power
    would be a lot of surface area for a single card. The diff approach is
    an approximation: a card that calls gain_block() more than once in a
    single play (none currently do) would only trigger this once, not once
    per call -- flagged rather than silently assumed exact."""
    name = "Juggernaut"
    is_duration = False


class Vulnerable(Power):
    """Target takes 50% more attack damage."""

    name = "Vulnerable"
    is_duration = True
    is_debuff = True

    @staticmethod
    def modify_damage_taken(damage: int) -> int:
        return math.floor(damage * 1.5)


class Weak(Power):
    """Attacker deals 25% less attack damage."""

    name = "Weak"
    is_duration = True
    is_debuff = True

    @staticmethod
    def modify_damage_dealt(damage: int) -> int:
        return math.floor(damage * 0.75)


class Frail(Power):
    """Owner gains 25% less block from cards."""

    name = "Frail"
    is_duration = True
    is_debuff = True


class Artifact(Power):
    """Negates the next `amount` debuff applications entirely -- see
    Creature.add_power for where this is actually enforced (the only
    centralized hook point every debuff application already goes through).
    Not itself a duration power (doesn't tick down at end of turn) and not
    removed by expires_end_of_turn -- it's consumed by use, not by time."""

    name = "Artifact"
    is_duration = False

    @staticmethod
    def modify_block_gained(block: int) -> int:
        return math.floor(block * 0.75)


class Ritual(Power):
    """At the end of its turn, owner gains ``amount`` Strength.

    (Cultist-style buff; included to show a start/end-of-turn hook in action.)
    """

    name = "Ritual"
    is_duration = False

    def on_end_turn(self, owner, combat) -> None:
        owner.add_power(Strength(self.amount))


class Poison(Power):
    """At the start of the owner's turn, lose HP equal to the stack (ignores
    block), then the stack decreases by 1. Silent's signature mechanic."""

    name = "Poison"
    is_duration = False

    def on_start_turn(self, owner, combat) -> None:
        lost = self.amount
        owner.lose_hp(lost)
        if owner is combat.player and lost > 0:
            combat._fire_hp_loss_relics(None, lost)
        self.amount -= 1
        if self.amount <= 0:
            owner.powers.pop(self.name, None)


class Focus(Power):
    """Defect's orb-scaling stat -- adds to every orb's passive/evoke value.
    Pure data; read directly by sts/orbs.py, no hooks of its own."""

    name = "Focus"
    is_duration = False


class Combust(Power):
    """At end of turn, lose 1 HP and deal ``amount`` damage to all enemies."""

    name = "Combust"
    is_duration = False

    def on_end_turn(self, owner, combat) -> None:
        owner.lose_hp(1)
        for m in combat.living_monsters:
            m.take_damage(self.amount)


class DarkEmbrace(Power):
    """Whenever a card is exhausted, draw 1 card."""

    name = "Dark Embrace"
    is_duration = False

    def on_exhaust(self, owner, combat) -> None:
        combat.draw_cards(1)


class FeelNoPain(Power):
    """Whenever a card is exhausted, gain ``amount`` Block."""

    name = "Feel No Pain"
    is_duration = False

    def on_exhaust(self, owner, combat) -> None:
        owner.gain_block(self.amount)


class Evolve(Power):
    """Whenever you draw a Status card, draw ``amount`` cards."""

    name = "Evolve"
    is_duration = False

    def on_draw_status(self, owner, combat) -> None:
        combat.draw_cards(self.amount)


class Metallicize(Power):
    """At end of turn, gain ``amount`` Block."""

    name = "Metallicize"
    is_duration = False

    def on_end_turn(self, owner, combat) -> None:
        owner.gain_block(self.amount)


class FlameBarrier(Power):
    """Whenever you are attacked this turn, deal `amount` damage back to
    the attacker. Cleared entirely at end of turn, same "this turn only"
    shape as Rage below (not a decaying duration stack)."""

    name = "Flame Barrier"
    is_duration = False
    expires_end_of_turn = True

    def on_player_attacked(self, attacker, combat) -> None:
        attacker.take_damage(self.amount)


class Panache(Power):
    """Every time you play 5 cards in a single turn, deal `amount` damage
    to ALL enemies -- the actual trigger check lives in
    CombatState.play_card (see its own comment), since there's no generic
    "on any card played" Power hook to attach this to (only the
    Attack-specific on_play_attack exists); this class just holds the
    stack amount."""

    name = "Panache"
    is_duration = False


class TheBomb(Power):
    """At the end of the 3rd turn after being played, deal `amount` damage
    to ALL enemies, then remove itself -- a one-shot countdown, not a
    decaying duration stack (real STS calls this out explicitly: the fixed
    damage isn't affected by Vulnerable/Strength/etc, which `take_damage`
    already doesn't apply here since this bypasses calc_attack_damage)."""

    name = "The Bomb"
    is_duration = False

    def __init__(self, amount: int = 40, turns_remaining: int = 3):
        super().__init__(amount)
        self.turns_remaining = turns_remaining

    def on_end_turn(self, owner, combat) -> None:
        self.turns_remaining -= 1
        if self.turns_remaining <= 0:
            for m in combat.living_monsters:
                m.take_damage(self.amount)
            if self.name in owner.powers:
                del owner.powers[self.name]


class Magnetism(Power):
    """At the start of your turn, add a random Colorless card into your
    hand. Lazy import of the colorless pool to avoid a circular import
    (cards.py already imports this module; powers.py importing back from
    cards.py at module load time would be circular, so the import happens
    inside the method instead, at call time)."""

    name = "Magnetism"
    is_duration = False

    def on_start_turn(self, owner, combat) -> None:
        from .cards import colorless_card_pool
        pool = colorless_card_pool()
        if pool:
            combat.add_card_to_hand(combat.rng.choice(pool))


class SadisticNature(Power):
    """Real effect ("whenever you apply a debuff to an enemy, they take
    `amount` damage") isn't wired to anything -- see make_sadistic_nature's
    docstring in cards.py for why (no hook point that knows the attacker,
    unlike Artifact's Creature.add_power check). Exists as a real,
    grantable power so at least "you have this power" is visible/queryable,
    rather than the card doing nothing observable at all."""

    name = "Sadistic Nature"
    is_duration = False


class Mayhem(Power):
    """At the start of your turn, play the top card of your draw pile
    (same underlying primitive as the Ironclad card Havoc, just recurring
    instead of one-shot, and NOT exhausting the played card -- real Mayhem
    doesn't exhaust it, Havoc does). Auto-targets the lowest-HP living
    monster for cards that need a target, same approximation Havoc already
    uses for this project's non-interactive auto-play cards."""

    name = "Mayhem"
    is_duration = False

    def on_start_turn(self, owner, combat) -> None:
        from .cards import CardType
        if not combat.draw_pile:
            return
        card = combat.draw_pile.pop()
        if card.playable and card.card_type in (CardType.ATTACK, CardType.SKILL):
            auto_target = min(combat.living_monsters, key=lambda m: m.hp, default=None)
            card.play(combat, auto_target)
        combat.discard_pile.append(card)


class Rage(Power):
    """Whenever you play an Attack this turn, gain ``amount`` Block. Cleared
    entirely at end of turn (not a decaying duration stack)."""

    name = "Rage"
    is_duration = False
    expires_end_of_turn = True

    def on_play_attack(self, owner, combat) -> None:
        owner.gain_block(self.amount)


class FireBreathing(Power):
    """Whenever you draw a Status card, deal ``amount`` damage to all enemies.
    (Real card also triggers on Curses; curses aren't modeled here.)"""

    name = "Fire Breathing"
    is_duration = False

    def on_draw_status(self, owner, combat) -> None:
        for m in combat.living_monsters:
            m.take_damage(self.amount)


class Rupture(Power):
    """Whenever you lose HP directly from a card's effect (not combat
    damage), gain ``amount`` Strength."""

    name = "Rupture"
    is_duration = False

    def on_hp_loss_from_card(self, owner, combat) -> None:
        owner.add_power(Strength(self.amount))


class Barricade(Power):
    """Block is not removed at the start of your turn. Pure flag, checked
    directly by CombatState.start_player_turn."""

    name = "Barricade"
    is_duration = False


class DemonForm(Power):
    """At the start of each turn, gain ``amount`` Strength."""

    name = "Demon Form"
    is_duration = False

    def on_start_turn(self, owner, combat) -> None:
        owner.add_power(Strength(self.amount))


class BerserkEnergy(Power):
    """At the start of each turn, gain ``amount`` extra Energy."""

    name = "Berserk Energy"
    is_duration = False

    def on_start_turn(self, owner, combat) -> None:
        owner.energy += self.amount


class Brutality(Power):
    """At the start of each turn, lose 1 HP and draw 1 card."""

    name = "Brutality"
    is_duration = False

    def on_start_turn(self, owner, combat) -> None:
        owner.lose_hp(1)
        combat.draw_cards(1)


class Corruption(Power):
    """Skills cost 0 and exhaust when played. Pure flag; the cost override
    and forced exhaust are handled directly in CombatState.play_card."""

    name = "Corruption"
    is_duration = False


# --- Silent powers ---

class NoxiousFumes(Power):
    """At the start of your turn, apply `amount` Poison to ALL enemies."""

    name = "Noxious Fumes"
    is_duration = False

    def on_start_turn(self, owner, combat) -> None:
        for m in combat.living_monsters:
            m.add_power(Poison(self.amount))


class InfiniteBlades(Power):
    """At the start of your turn, add a Shiv to your hand."""

    name = "Infinite Blades"
    is_duration = False

    def on_start_turn(self, owner, combat) -> None:
        from .cards import make_shiv
        combat.add_card_to_hand(make_shiv())


class Caltrops(Power):
    """Whenever you are attacked, deal `amount` damage back to the attacker."""

    name = "Caltrops"
    is_duration = False

    def on_player_attacked(self, attacker, combat) -> None:
        attacker.take_damage(self.amount)


class ThousandCuts(Power):
    """Whenever you play a card, deal `amount` damage to ALL enemies."""

    name = "A Thousand Cuts"
    is_duration = False

    def on_any_card_played(self, owner, combat, card) -> None:
        for m in combat.living_monsters:
            m.take_damage(self.amount)


class AfterImage(Power):
    """Whenever you play a card, gain 1 Block."""

    name = "After Image"
    is_duration = False

    def on_any_card_played(self, owner, combat, card) -> None:
        owner.gain_block(1)


class Envenom(Power):
    """Whenever an attack deals unblocked damage, apply `amount` Poison."""

    name = "Envenom"
    is_duration = False


class ToolsOfTheTrade(Power):
    """At the start of your turn, discard a card then draw a card."""

    name = "Tools of the Trade"
    is_duration = False

    def on_start_turn(self, owner, combat) -> None:
        if combat.hand:
            card = combat.rng.choice(combat.hand)
            combat.hand.remove(card)
            combat.discard_pile.append(card)
            combat.on_discard(card)
        combat.draw_cards(1)


class WraithForm(Power):
    """Intangible (reduce all damage taken to 1). Lose 1 Dexterity at end of turn."""

    name = "Wraith Form"
    is_duration = False

    def on_end_turn(self, owner, combat) -> None:
        dex = owner.powers.get("Dexterity")
        if dex is not None:
            dex.amount -= 1
            if dex.amount <= 0:
                del owner.powers["Dexterity"]
        else:
            owner.add_power(Dexterity(-1))


# --- Defect powers ---

class Loop(Power):
    """At the start of your turn, trigger the passive of your first orb
    `amount` times."""

    name = "Loop"
    is_duration = False

    def on_start_turn(self, owner, combat) -> None:
        if owner.orbs:
            for _ in range(self.amount):
                owner.orbs[0].passive(owner, combat)


class StaticDischarge(Power):
    """Whenever you receive unblocked attack damage, channel `amount`
    Lightning orb(s)."""

    name = "Static Discharge"
    is_duration = False


class Storm(Power):
    """Whenever you play a Power card, channel `amount` Lightning orb(s)."""

    name = "Storm"
    is_duration = False


class Heatsinks(Power):
    """Whenever you play a Power card, draw `amount` card(s)."""

    name = "Heatsinks"
    is_duration = False


class CreativeAI(Power):
    """At the start of your turn, add a random Power card to your hand."""

    name = "Creative AI"
    is_duration = False

    def on_start_turn(self, owner, combat) -> None:
        from .cards import common_card_pool, silent_card_pool
        from .cards import defect_card_pool, watcher_card_pool
        import random
        all_power_pools = [common_card_pool(), silent_card_pool(),
                          defect_card_pool(), watcher_card_pool()]
        all_cards = [c for pool in all_power_pools for c in pool if c.card_type.value == "Power"]
        if all_cards:
            combat.add_card_to_hand(combat.rng.choice(all_cards))


class HelloWorld(Power):
    """At the start of your turn, add a random common card to your hand."""

    name = "Hello World"
    is_duration = False

    def on_start_turn(self, owner, combat) -> None:
        from .cards import common_card_pool
        pool = [c for c in common_card_pool() if c.cost != -1]
        if pool:
            combat.add_card_to_hand(combat.rng.choice(pool))


class MachineLearning(Power):
    """At the start of your turn, draw `amount` additional card(s)."""

    name = "Machine Learning"
    is_duration = False

    def on_start_turn(self, owner, combat) -> None:
        combat.draw_cards(self.amount)


class BiasedCognition(Power):
    """Gain `initial` Focus on play. Lose 1 Focus at the start of each turn."""

    name = "Biased Cognition"
    is_duration = False

    def __init__(self, initial_focus: int = 4):
        super().__init__(initial_focus)
        self.initial_focus = initial_focus

    def on_start_turn(self, owner, combat) -> None:
        focus = owner.powers.get("Focus")
        if focus is not None:
            focus.amount -= 1
            if focus.amount <= 0:
                del owner.powers["Focus"]
        else:
            owner.add_power(Focus(-1))


class Buffer(Power):
    """Prevent the next `amount` time(s) you would lose HP."""

    name = "Buffer"
    is_duration = False


class Amplify(Power):
    """Your next Power card is played twice. amount = number of doubles."""

    name = "Amplify"
    is_duration = False
    expires_end_of_turn = True


class EchoForm(Power):
    name = "Echo Form"
    is_duration = False

    def on_start_turn(self, owner, combat) -> None:
        self.amount = 1


# --- Watcher powers ---

class LikeWater(Power):
    """At the end of your turn, if you are in Calm, gain `amount` Block."""

    name = "Like Water"
    is_duration = False

    def on_end_turn(self, owner, combat) -> None:
        if owner.stance == "Calm":
            owner.gain_block(self.amount)


class Rushdown(Power):
    """Whenever you enter Wrath, draw `amount` cards."""

    name = "Rushdown"
    is_duration = False


class MentalFortress(Power):
    """Whenever you switch Stances, gain `amount` Block."""

    name = "Mental Fortress"
    is_duration = False


class BattleHymn(Power):
    """At the start of your turn, add a Smite into your hand."""

    name = "Battle Hymn"
    is_duration = False

    def on_start_turn(self, owner, combat) -> None:
        from .cards import make_smite
        combat.add_card_to_hand(make_smite())


class Fasting(Power):
    """Gain `amount` Strength and Dexterity. Lose 1 Energy at the start of
    each turn."""

    name = "Fasting"
    is_duration = False

    def on_start_turn(self, owner, combat) -> None:
        owner.energy = max(0, owner.energy - 1)

    def __init__(self, amount: int = 3):
        super().__init__(amount)


class Devotion(Power):
    """At the start of your turn, gain `amount` Mantra."""

    name = "Devotion"
    is_duration = False

    def on_start_turn(self, owner, combat) -> None:
        owner._mantra = getattr(owner, '_mantra', 0) + self.amount
        if owner._mantra >= 10:
            owner._mantra = 0
            owner.energy += 3

    def __init__(self, amount: int = 2):
        super().__init__(amount)


# --- Additional Silent powers ---

class NextTurnBlock(Power):
    name = "Next Turn Block"
    is_duration = False
    def on_start_turn(self, owner, combat) -> None:
        owner.gain_block(self.amount)
        if self.name in owner.powers:
            del owner.powers[self.name]


class NextTurnEnergy(Power):
    name = "Next Turn Energy"
    is_duration = False
    def on_start_turn(self, owner, combat) -> None:
        owner.energy += self.amount
        if self.name in owner.powers:
            del owner.powers[self.name]


class NextTurnDraw(Power):
    name = "Next Turn Draw"
    is_duration = False
    def on_start_turn(self, owner, combat) -> None:
        combat.draw_cards(self.amount)
        if self.name in owner.powers:
            del owner.powers[self.name]


class AccuracyPower(Power):
    name = "Accuracy"
    is_duration = False


class BlurPower(Power):
    name = "Blur"
    is_duration = False


class ChokePower(Power):
    name = "Choke"
    expires_end_of_turn = True
    def __init__(self, damage_per_card: int, target):
        super().__init__(damage_per_card)
        self.target = target

    def on_any_card_played(self, owner, combat, card) -> None:
        self.target.take_damage(self.amount)


class PhantasmalKillerPower(Power):
    name = "Phantasmal Killer"
    is_duration = False
    def on_start_turn(self, owner, combat) -> None:
        combat.next_turn_double_damage = True
        if self.name in owner.powers:
            del owner.powers[self.name]


class DoppelgangerPower(Power):
    name = "Doppelganger"
    is_duration = False
    def on_start_turn(self, owner, combat) -> None:
        combat.draw_cards(self.amount)
        owner.energy += self.amount
        if self.name in owner.powers:
            del owner.powers[self.name]


class WellLaidPlansPower(Power):
    name = "Well Laid Plans"
    is_duration = False


class BurstPower(Power):
    name = "Burst"
    is_duration = False
    expires_end_of_turn = True


class NightmarePower(Power):
    name = "Nightmare"
    is_duration = False
    def __init__(self, stored_card, amount: int = 1):
        super().__init__(amount)
        self.stored_card = stored_card
    def on_start_turn(self, owner, combat) -> None:
        for _ in range(3):
            combat.add_card_to_hand(self.stored_card)
        if self.name in owner.powers:
            del owner.powers[self.name]


class NextTurnZeroCostPower(Power):
    name = "Next Turn Zero Cost"
    is_duration = False
    def __init__(self, stored_card, amount: int = 1):
        super().__init__(amount)
        self.stored_card = stored_card
    def on_start_turn(self, owner, combat) -> None:
        target = self.stored_card
        if target in combat.hand:
            target.cost = 0
        elif combat.hand:
            combat.rng.choice(combat.hand).cost = 0
        if self.name in owner.powers:
            del owner.powers[self.name]


class CorpseExplosionPower(Power):
    name = "Corpse Explosion"
    is_duration = False


class Mark(Power):
    """Mark stacks on a monster. When Pressure Points is played, the target
    takes HP loss equal to its current Mark amount (Mark is not consumed).
    Not a debuff: in the real game Mark bypasses Artifact, unlike Weak/
    Vulnerable/Frail."""
    name = "Mark"
    is_duration = False


class Intangible(Power):
    """Reduce all attack damage taken and direct HP loss to 1.
    Modified directly in Creature.take_damage and Creature.lose_hp
    (see creatures.py), same explicit-integration approach as
    Vulnerable/Weak/Frail rather than a hook callback."""

    name = "Intangible"
    is_duration = False


class TalkToTheHand(Power):
    name = "Talk to the Hand"
    is_duration = False


class Shackled(Power):
    """Temporary Strength loss that restores at end of turn.

    Lives on the PLAYER. At end of the player's turn, restores the lost
    Strength to the target monster(s) and removes itself.
    target=None means "all living monsters" (Piercing Wail).
    """

    name = "Shackled"
    is_duration = False

    def __init__(self, amount: int, target=None):
        super().__init__(amount)
        self.target = target

    def on_end_turn(self, owner, combat) -> None:
        if self.target is not None and not self.target.is_dead:
            self.target.add_power(Strength(self.amount))
        elif self.target is None:
            for m in combat.living_monsters:
                m.add_power(Strength(self.amount))
        if self.name in owner.powers:
            del owner.powers[self.name]


class SelfRepairPower(Power):
    """Stores the heal amount from Self Repair. Healing fires at end of combat,
    which this single-combat engine doesn't model -- the amount is held here
    so it's observable/queryable as a power, consistent with how the real game
    tracks it."""

    name = "Self Repair"
    is_duration = False
