"""The combat engine.

Owns the card piles, energy, and turn loop, and exposes the primitives cards and
monsters call (``deal_attack_damage``, ``draw_cards``, ...). This is the
simulator surface the AI will eventually drive: read the state, choose from
``legal_actions()``, call ``play_card`` / ``end_turn``, repeat.

Turn structure (StS1):
  combat start -> roll enemy intents
  player turn:  reset block (unless Barricade), refill energy, draw 5,
                play cards, resolve orb passives, end turn
  enemy turn:   each monster resets block, acts on intent, ticks, rolls next
  repeat until one side is dead.
"""

from __future__ import annotations

import copy
import random
from enum import Enum
from typing import List, Optional

from .cards import Card, CardType, make_dazed
from .creatures import Player
from .enemies import Monster
from .powers import Weak, Frail, Strength, Poison


class Result(Enum):
    ONGOING = "ongoing"
    WIN = "win"
    LOSS = "loss"


class CombatState:
    HAND_SIZE = 5

    def __init__(
        self,
        player: Player,
        monsters: List[Monster],
        deck: List[Card],
        rng: Optional[random.Random] = None,
        verbose: bool = False,
    ):
        self.player = player
        self.monsters = monsters
        self.rng = rng or random.Random()  # via the property setter below
        self.verbose = verbose

        self.draw_pile: List[Card] = list(deck)
        self.rng.shuffle(self.draw_pile)
        self.hand: List[Card] = []
        self.discard_pile: List[Card] = []
        self.exhaust_pile: List[Card] = []

        self.turn = 0
        # Set while resolving an X-cost card's play() (Whirlwind); the
        # amount of energy that was spent, i.e. the value of X.
        self._x_value = 0
        # Double Tap: replays the next Attack's play() once more.
        self.double_tap_charges = 0
        # Burst: replays the next Skill's play() once more.
        self.burst_charges = 0
        # Normality ("cannot play more than 3 cards this turn while in
        # hand") and Pain ("lose 1 HP whenever you play a card while in
        # hand") both need this -- neither existed before those two curses.
        self.cards_played_this_turn = 0
        self.cards_discarded_this_turn = 0
        # Time Eater's Time Warp: unlike cards_played_this_turn, this counts
        # cards played across the WHOLE fight and is only reset by the
        # trigger itself (confirmed: "this counter carries over between
        # turns"), so it needs its own field. Set True the instant the 12th
        # card resolves; legal_actions() then only offers ("end",) until
        # start_player_turn() clears it, which is what actually forces the
        # turn to end -- there's no separate "force end turn" primitive in
        # this engine, callers always end a turn by choosing that action.
        self.turn_should_end_early = False
        # Transient bonus/multiplier a relic can apply to the Attack card
        # currently being played (Akabeko, Pen Nib); reset after each
        # Attack card resolves. Only affects the player's own attacks.
        self.pending_damage_bonus = 0
        self.pending_damage_multiplier = 1.0
        self.next_turn_double_damage = False
        self.skip_monster_turn = False

        for m in self.monsters:
            m.roll_intent(self.rng)
        for relic in self.player.relics:
            relic.on_combat_start(self.player, self)

    # --- logging ---
    def log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def clone(self) -> "CombatState":
        """Independent copy for search to branch on without touching the
        real fight. Player/monsters use their own fast clone() (see
        Creature.clone in creatures.py) rather than copy.deepcopy; the card
        piles are shallow-copied lists -- Card objects themselves are
        stateless/immutable, so sharing the same instances across clones is
        safe and far cheaper than deep-copying every card in every branch.
        """
        new = CombatState.__new__(CombatState)
        new.player = self.player.clone()
        new.monsters = [m.clone() for m in self.monsters]
        # copy.deepcopy(random.Random) reconstructs via Random.__reduce__,
        # which calls Random() -- reseeding from OS entropy (a syscall)
        # right before immediately being overwritten by setstate(). __new__
        # + getstate()/setstate() avoided that, but still cost ~11% of total
        # search time on its own (measured via cProfile). Seeding a fresh
        # Random from one draw of the parent's stream is ~2.7x cheaper
        # (microbenchmark: 1.26s vs 3.43s / 300k iterations) and safe here
        # specifically because _state_key() in search.py deliberately
        # excludes combat.rng from the transposition-table cache key -- the
        # search never depends on *which* random samples produced a value,
        # only on the value itself. Note this does make clone() consume one
        # draw from the parent's own rng as a side effect (getstate/setstate
        # didn't touch the source at all); confirmed safe via the full test
        # suite (101/101) plus a real timing comparison (~20.6% faster end
        # to end on a turns_left=2 workload: 41.85s -> 33.23s).
        #
        # A lazy variant (defer seeding until .rng is first read, since many
        # clones -- intra-turn card-play siblings -- never touch it before
        # being discarded) was tried and measured: it saved almost nothing
        # (~3.5% wall time) because most clones DO eventually try "end turn"
        # and resolve their rng anyway, and the resolution bookkeeping ate
        # most of the savings. Reverted rather than keep the extra
        # indirection for a result within noise.
        new.rng = random.Random(self.rng.random())
        new.verbose = False
        new.draw_pile = list(self.draw_pile)
        new.hand = list(self.hand)
        new.discard_pile = list(self.discard_pile)
        new.exhaust_pile = list(self.exhaust_pile)
        new.turn = self.turn
        new._x_value = self._x_value
        new.double_tap_charges = self.double_tap_charges
        new.burst_charges = self.burst_charges
        new.pending_damage_bonus = self.pending_damage_bonus
        new.pending_damage_multiplier = self.pending_damage_multiplier
        new.cards_played_this_turn = self.cards_played_this_turn
        new.cards_discarded_this_turn = self.cards_discarded_this_turn
        new.turn_should_end_early = self.turn_should_end_early
        new.next_turn_double_damage = self.next_turn_double_damage
        new.skip_monster_turn = self.skip_monster_turn
        return new

    # --- queries ---
    @property
    def living_monsters(self) -> List[Monster]:
        return [m for m in self.monsters if not m.is_dead]

    def result(self) -> Result:
        if self.player.is_dead:
            return Result.LOSS
        if not self.living_monsters:
            return Result.WIN
        return Result.ONGOING

    def _effective_cost(self, card: Card) -> int:
        """Cost after Corruption's "Skills cost 0" override, or Blood for
        Blood's "costs 1 less for each time you lose HP this combat". X-cost
        cards aren't gated by this (they spend whatever energy remains)."""
        if card.card_type == CardType.SKILL and self.player.has_power("Corruption"):
            return 0
        if card.name in ("Blood for Blood", "Blood for Blood+"):
            return max(0, card.cost - self.player.hp_loss_count_this_combat)
        if card.name in ("Masterful Stab", "Masterful Stab+"):
            return card.cost + self.player.hp_loss_count_this_combat
        return card.cost

    def legal_actions(self) -> List[tuple]:
        """Actions available right now, as ('play', card, target) or ('end',).

        This is the action set the AI selects from.
        """
        actions: List[tuple] = []
        if self.turn_should_end_early:
            # Time Eater's Time Warp fired on the last card played -- no
            # further card can legally go down, only ending the turn.
            return [("end",)]
        for card in self.hand:
            if not card.playable:
                continue  # Status cards (Wound, Dazed, ...)
            if not card.is_x_cost and self._effective_cost(card) > self.player.energy:
                continue
            if card.extra_legal_check is not None and not card.extra_legal_check(self):
                continue
            if card.targeted:
                for m in self.living_monsters:
                    actions.append(("play", card, m))
            else:
                actions.append(("play", card, None))
        actions.append(("end",))
        return actions

    # --- primitives used by cards / monsters ---
    def deal_attack_damage(self, attacker, target, base: int, strength_multiplier: int = 1) -> int:
        dmg = attacker.calc_attack_damage(base, strength_multiplier)
        if attacker is self.player:
            dmg = int(dmg * self.pending_damage_multiplier) + self.pending_damage_bonus
            if self.next_turn_double_damage:
                dmg *= 2
        hp_loss = target.take_damage(dmg)
        if target.is_dead and target.has_power("Corpse Explosion"):
            for m in self.living_monsters:
                m.take_damage(target.max_hp)
        if target is self.player and hp_loss > 0:
            self._fire_hp_loss_relics(attacker, hp_loss)
        if target is self.player and attacker is not self.player:
            for power in list(self.player.powers.values()):
                power.on_player_attacked(attacker, self)
        if target.has_power("Talk to the Hand"):
            self.player.gain_block(target.get_power_amount("Talk to the Hand"))
        if attacker is self.player and hp_loss > 0:
            envenom = self.player.get_power_amount("Envenom")
            if envenom > 0:
                target.add_power(Poison(envenom))
        if self.verbose:
            self.log(f"    {attacker.name} hits {target.name} for {dmg} "
                     f"(-{hp_loss} HP)")
        return hp_loss

    def _fire_hp_loss_relics(self, attacker, amount: int) -> None:
        for relic in self.player.relics:
            relic.on_player_hp_loss(self.player, self, attacker, amount)

    def draw_cards(self, n: int) -> None:
        for _ in range(n):
            if not self.draw_pile:
                if not self.discard_pile:
                    return  # nothing left to draw
                self.draw_pile = self.discard_pile
                self.discard_pile = []
                self.rng.shuffle(self.draw_pile)
            card = self.draw_pile.pop()
            self.hand.append(card)
            if card.card_type == CardType.STATUS:
                for power in list(self.player.powers.values()):
                    power.on_draw_status(self.player, self)

    def scry(self, amount: int) -> None:
        """Look at the top ``amount`` cards of the draw pile, discard them all
        (pragmatic no-UI approximation of "player chooses which to discard"),
        and leave the rest of the draw pile untouched in its original order.
        Standard StS rule: if the draw pile is empty, shuffle the discard pile
        into it first."""
        if amount <= 0:
            return
        if not self.draw_pile and self.discard_pile:
            self.draw_pile = self.discard_pile
            self.discard_pile = []
            self.rng.shuffle(self.draw_pile)
        taken = 0
        while taken < amount and self.draw_pile:
            self.discard_pile.append(self.draw_pile.pop())
            taken += 1

    def add_card_to_hand(self, card: Card) -> None:
        """Add a freshly-created card straight to hand (Blade Dance's Shivs,
        Power Through's Wounds, Infernal Blade's random Attack) -- distinct
        from draw_cards() since this doesn't come from the draw pile and
        doesn't trigger on-draw-status effects."""
        self.hand.append(card)

    def player_loses_hp_from_card(self, amount: int) -> None:
        """HP loss that's a direct card effect (Bloodletting, Hemokinesis,
        Offering) rather than combat damage -- fires Rupture and any
        any-HP-loss relics (Runic Cube, Centennial Puzzle, Torii)."""
        self.player.lose_hp(amount)
        for power in list(self.player.powers.values()):
            power.on_hp_loss_from_card(self.player, self)
        if amount > 0:
            self._fire_hp_loss_relics(None, amount)

    def channel_orb(self, orb) -> None:
        """Add an orb to the queue; if it overflows orb_slots, the oldest
        orb is evicted and evoked (Defect)."""
        self.player.orbs.append(orb)
        if len(self.player.orbs) > self.player.orb_slots:
            evoked = self.player.orbs.pop(0)
            evoked.evoke(self.player, self)

    def apply_end_of_turn_orbs(self) -> None:
        for orb in self.player.orbs:
            orb.passive(self.player, self)

    def _exhaust(self, card: Card) -> None:
        # Sentinel: "if this card is Exhausted, gain 2(3) energy" -- a
        # per-card reaction, not a player-wide power, so checked by name
        # here rather than via the powers loop below.
        if card.name in ("Sentinel", "Sentinel+"):
            self.player.energy += 3 if card.upgraded else 2
        self.exhaust_pile.append(card)
        for power in list(self.player.powers.values()):
            power.on_exhaust(self.player, self)

    def on_discard(self, card: Card) -> None:
        self.cards_discarded_this_turn += 1
        if card.name == "Reflex":
            self.draw_cards(2)
        elif card.name == "Reflex+":
            self.draw_cards(3)
        elif card.name == "Tactician":
            self.player.energy += 1
        elif card.name == "Tactician+":
            self.player.energy += 2

    # --- player actions ---
    def play_card(self, card: Card, target: Optional[Monster]) -> None:
        if card not in self.hand:
            raise ValueError(f"{card} not in hand")
        if not card.playable:
            raise ValueError(f"{card} is unplayable")
        if not card.is_x_cost and self._effective_cost(card) > self.player.energy:
            raise ValueError(f"not enough energy for {card}")
        if card.extra_legal_check is not None and not card.extra_legal_check(self):
            raise ValueError(f"{card} is not legally playable right now")
        if card.targeted and (target is None or target.is_dead):
            raise ValueError(f"{card} needs a living target")
        if (self.cards_played_this_turn >= 3
                and any(c.name == "Normality" for c in self.hand)):
            raise ValueError("Normality: cannot play more than 3 cards this turn")

        if card.is_x_cost:
            self._x_value = self.player.energy
            self.player.energy = 0
        else:
            self.player.energy -= self._effective_cost(card)
        self.cards_played_this_turn += 1

        # Pain: "while in hand, lose 1 HP whenever you play a card" -- fires
        # for any OTHER card being played, checked before the play itself so
        # a lethal self-inflicted loss can't be dodged by ordering.
        if any(c.name == "Pain" and c is not card for c in self.hand):
            self.player.lose_hp(1)

        self.hand.remove(card)
        if self.verbose:
            self.log(f"  play {card.name}"
                     + (f" -> {target.name}" if target else ""))

        if card.card_type == CardType.ATTACK:
            for relic in self.player.relics:
                relic.before_attack_card(self.player, self)

        block_before_play = self.player.block
        card.play(self, target)

        if card.card_type == CardType.ATTACK:
            for power in list(self.player.powers.values()):
                power.on_play_attack(self.player, self)
            for relic in self.player.relics:
                relic.on_attack_played(self.player, self)
            if self.double_tap_charges > 0:
                self.double_tap_charges -= 1
                card.play(self, target)
                for power in list(self.player.powers.values()):
                    power.on_play_attack(self.player, self)
                for relic in self.player.relics:
                    relic.on_attack_played(self.player, self)
            self.pending_damage_bonus = 0
            self.pending_damage_multiplier = 1.0

        else:
            # Chosen's Hex move ("whenever you play a non-Attack card,
            # shuffle a Dazed into your draw pile") is the only thing that
            # needs a hook into non-attack card plays specifically -- no
            # generic "monster reacts to card type" mechanism existed before
            # this, so it's a plain flag check here rather than a new
            # abstraction for what's currently a single monster's ability.
            for m in self.monsters:
                if not m.is_dead and getattr(m, "hex_active", False):
                    self.draw_pile.append(make_dazed())
                    self.rng.shuffle(self.draw_pile)
                    break

            if card.card_type == CardType.SKILL and self.burst_charges > 0:
                self.burst_charges -= 1
                card.play(self, target)

        # Panache: "every time you play 5 cards in a single turn, deal
        # damage to ALL enemies" -- reuses cards_played_this_turn (added
        # for Normality/Pain above); checked here rather than as a Power
        # hook since there's no generic "on any card played" hook, just the
        # attack-specific on_play_attack one, and adding a whole new hook
        # for one card felt like more machinery than a single `%` check.
        if self.player.has_power("Panache") and self.cards_played_this_turn % 5 == 0:
            for m in self.living_monsters:
                m.take_damage(self.player.get_power_amount("Panache"))

        # Time Eater's Time Warp: "whenever you play a card [any type],
        # ends your turn and gains 2 Strength" after the 12th such card --
        # the counter carries over between turns (confirmed real-game
        # behavior, not guessed), so it lives on the monster itself rather
        # than reusing cards_played_this_turn, which resets every turn.
        for m in self.monsters:
            if not m.is_dead and hasattr(m, "time_warp_counter"):
                m.time_warp_counter += 1
                if m.time_warp_counter >= 12:
                    m.time_warp_counter = 0
                    m.add_power(Strength(2))
                    self.turn_should_end_early = True
                break

        if self.player.block > block_before_play and self.player.has_power("Juggernaut"):
            living = self.living_monsters
            if living:
                target_monster = self.rng.choice(living)
                target_monster.take_damage(self.player.get_power_amount("Juggernaut"))

        # on_any_card_played: fires for every card played (Thousand Cuts, After Image)
        for power in list(self.player.powers.values()):
            power.on_any_card_played(self.player, self, card)

        if card.card_type == CardType.POWER:
            for power in list(self.player.powers.values()):
                power.on_power_played(self.player, self)

        # Power cards always leave the deck permanently once played (their
        # effect is now a standing buff; the physical card would just be a
        # dead future draw otherwise) -- same end state as exhausting.
        force_exhaust = (
            card.card_type == CardType.POWER
            or (card.card_type == CardType.SKILL and self.player.has_power("Corruption"))
        )
        if card.exhausts or force_exhaust:
            self._exhaust(card)
        else:
            self.discard_pile.append(card)

    # --- turn flow ---
    def start_player_turn(self) -> None:
        self.turn += 1
        self.cards_played_this_turn = 0
        self.cards_discarded_this_turn = 0
        self.turn_should_end_early = False
        self.next_turn_double_damage = False
        if self.player.no_block_from_cards_turns > 0:
            self.player.no_block_from_cards_turns -= 1
        if not self.player.has_power("Barricade") and not self.player.has_power("Blur"):
            self.player.block = 0
        self.player.energy = self.player.max_energy
        self.player.start_turn(self)
        for relic in self.player.relics:
            relic.on_turn_start(self.player, self)
        self.draw_cards(self.HAND_SIZE)
        if self.verbose:
            self.log(f"\n=== Turn {self.turn} ===")
            self.log("  " + self.player.status_str())
            for m in self.living_monsters:
                self.log(f"  {m.name} intends: {m.intent}")

    def end_player_turn(self) -> None:
        self.player.end_turn(self)
        for relic in self.player.relics:
            relic.on_turn_end(self.player, self)
        self.apply_end_of_turn_orbs()
        remaining = self.hand
        self.hand = []

        # Curses that act "while in hand" at end of turn -- applied once per
        # instance present (so two copies both trigger), before they're
        # cleared out below. Deliberately real per-turn effects, not treated
        # as inert filler like Wound/Dazed/Burn: these four are exactly the
        # curses whose whole threat *is* this recurring effect.
        for card in remaining:
            if card.name == "Doubt":
                self.player.add_power(Weak(1))
            elif card.name == "Shame":
                self.player.add_power(Frail(1))
            elif card.name == "Regret":
                self.player.lose_hp(len(remaining))
            elif card.name == "Decay":
                self.player.lose_hp(2)

        # Well-Laid Plans: retain up to amount random cards that don't
        # already have the retain flag set on the card itself.
        wlp_amount = self.player.get_power_amount("Well Laid Plans")
        wlp_retain = []
        if wlp_amount > 0:
            non_retain = [c for c in remaining if not c.retain]
            self.rng.shuffle(non_retain)
            wlp_retain = non_retain[:wlp_amount]

        for card in remaining:
            if card.retain or card in wlp_retain:
                if card.retain_callback is not None:
                    card.retain_callback(card, self)
                self.hand.append(card)
            elif card.ethereal:
                self._exhaust(card)
            else:
                self.discard_pile.append(card)

    def enemy_turn(self) -> None:
        if self.skip_monster_turn:
            self.skip_monster_turn = False
            return
        for m in self.living_monsters:
            m.block = 0
            m.start_turn(self)
            m.take_turn(self)
            m.end_turn(self)
            if not m.is_dead:
                m.roll_intent(self.rng)
            if self.player.is_dead:
                return
