"""Cards.

A Card carries its cost and a ``play`` function describing its effect. Effects
are written against the CombatState so a card can deal damage, gain block, apply
powers, draw, etc. Adding a card is a few lines here — no engine changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from .powers import (
    Strength, Dexterity, Vulnerable, Weak, Poison, Frail,
    Combust, DarkEmbrace, FeelNoPain, Evolve, Metallicize, Rage,
    FireBreathing, Rupture, Barricade, DemonForm, BerserkEnergy, Brutality,
    Corruption, Focus, Juggernaut, FlameBarrier,
    Artifact, Panache, TheBomb, Magnetism, Mayhem, SadisticNature,
    Loop, StaticDischarge, Storm, Heatsinks, CreativeAI, HelloWorld,
    MachineLearning, BiasedCognition, Buffer, Amplify, EchoForm,
    NoxiousFumes, InfiniteBlades, Caltrops, ThousandCuts, AfterImage,
    Envenom, ToolsOfTheTrade, WraithForm,
    NextTurnBlock, NextTurnEnergy, NextTurnDraw,
    AccuracyPower, BlurPower, ChokePower, PhantasmalKillerPower,
    DoppelgangerPower, WellLaidPlansPower, BurstPower, NightmarePower,
    NextTurnZeroCostPower, CorpseExplosionPower,
    LikeWater, Rushdown, MentalFortress, BattleHymn, Fasting, Devotion,
    Shackled, Intangible, TalkToTheHand, SelfRepairPower,
)
from .orbs import make_lightning_orb, make_frost_orb, make_dark_orb, make_plasma_orb


class CardType(Enum):
    ATTACK = "Attack"
    SKILL = "Skill"
    POWER = "Power"
    STATUS = "Status"  # Wound, Dazed, Burn -- unplayable deck clutter
    CURSE = "Curse"  # Ascender's Bane, etc. -- distinct from STATUS because
    # combat.py's draw_cards() fires on_draw_status (Evolve, Fire Breathing)
    # specifically for CardType.STATUS; curses don't trigger that in the
    # real game, so mislabeling one as STATUS would be a real behavior bug,
    # not just a wrong name.


@dataclass
class Card:
    name: str
    cost: int
    card_type: CardType
    # play(combat, target) -> None. target is a Monster for attacks, else None.
    play: Callable
    exhausts: bool = False
    # Attacks need a target; skills/powers usually don't.
    targeted: bool = False
    # Status cards (Wound, Dazed, ...): never appear in legal_actions.
    playable: bool = True
    # Ethereal: if still in hand at end of turn, exhausts instead of
    # discarding (Ghostly Armor, Reaper).
    ethereal: bool = False
    # X-cost cards (Whirlwind): spend all remaining energy; the amount spent
    # is passed to `play` via combat._x_value rather than a fixed cost.
    is_x_cost: bool = False
    # Extra playability predicate beyond cost/target (Clash: "only playable
    # if every card in hand is an Attack"). Signature: (combat) -> bool.
    extra_legal_check: Optional[Callable] = None
    # Retain: card stays in hand at end of turn instead of being discarded.
    # retain_callback is called when the card is retained, for scaling
    # mechanics (Perseverance, Sands of Time, Windmill Strike).
    retain: bool = False
    retain_callback: Optional[Callable] = None  # (card, combat) -> None
    # Whether this is the upgraded ("+") version. Set by passing
    # upgraded=True to a make_*() factory that supports it -- see the note
    # above the upgrade-supporting factories below. Deliberately part of the
    # Card's identity (folded into its name, e.g. "Strike+") rather than a
    # separate flag some other code has to remember to check: the
    # transposition table already hashes cards by name (see search.py's
    # _pile_key), so an upgraded card naturally gets treated as distinct
    # from its base version without any extra cache-key plumbing.
    upgraded: bool = False

    def __repr__(self) -> str:
        return f"{self.name}(cost {self.cost})"


# --- Status cards: inert deck clutter that some cards shuffle in.
# Simplified vs. the real game: Burn normally deals damage while sitting in
# hand at end of turn and Dazed is Ethereal (auto-exhausts if not played);
# here all three are just unplayable filler that dilutes future draws,
# which is their main strategic cost anyway.

def _unplayable(combat, target):
    raise AssertionError("status cards should never be playable")


def make_wound() -> Card:
    return Card("Wound", 0, CardType.STATUS, _unplayable, playable=False)


def make_dazed() -> Card:
    return Card("Dazed", 0, CardType.STATUS, _unplayable, playable=False)


def make_burn() -> Card:
    return Card("Burn", 0, CardType.STATUS, _unplayable, playable=False)


def make_slimed() -> Card:
    """Added by Slime Boss's Goop Spray. Ethereal (auto-exhausts if drawn
    and not dealt with), same treatment as Ascender's Bane above -- no other
    mechanical effect (real Burn's own "take 2 damage if in hand at end of
    turn" isn't modeled in this engine either, so this stays consistent
    with that existing simplification level)."""
    return Card("Slimed", 1, CardType.STATUS, _unplayable, playable=False, ethereal=True)


def make_ascenders_bane() -> Card:
    """Auto-added to the starting deck at Ascension 10+ (confirmed via
    direct sts_lightspeed construction: GameContext's own deck includes it
    starting exactly at A10, ground truth not guessed). Unlike Wound/Dazed/
    Burn, this one's real-game Ethereal flag actually matters here rather
    than being simplified away -- with `lightspeed/`'s training now
    defaulting to A20, every single training episode has this in the deck
    from turn 1, and without ethereal=True it would just sit inert in hand/
    discard forever like a Wound; with it, it auto-exhausts if drawn and
    not dealt with, matching the real card. No other mechanical effect --
    unlike Regret/Pain/Doubt/Shame/Decay below, which do have active
    hand-size/HP-loss/debuff effects (now modeled, see below)."""
    return Card("Ascender's Bane", 0, CardType.CURSE, _unplayable,
                playable=False, ethereal=True)


def make_injury() -> Card:
    """No mechanical effect beyond being unplayable deck clutter -- the
    curse equivalent of Wound."""
    return Card("Injury", 0, CardType.CURSE, _unplayable, playable=False)


def make_clumsy() -> Card:
    """Unplayable + Ethereal, no other effect -- the curse equivalent of
    Dazed."""
    return Card("Clumsy", 0, CardType.CURSE, _unplayable, playable=False, ethereal=True)


def make_parasite() -> Card:
    """Real effect ("lose 3 Max HP if transformed or removed from your
    deck") only fires on deck-editing events (a card removal service, a
    transform relic/potion) -- this project's CombatState models a single
    fight, not the meta-game around a full run's deck, so there's nothing
    for that hook to attach to here. No in-combat effect, same as
    Ascender's Bane's own "no other mechanical effect" bucket."""
    return Card("Parasite", 0, CardType.CURSE, _unplayable, playable=False)


def make_writhe() -> Card:
    """Real card is Innate (always in the opening hand of every combat) --
    not modeled: no card in this project currently forces itself into the
    opening hand, and adding that mechanic for one curse's minor
    guarantee (rather than a real gameplay-shaping effect) is out of
    proportion to the payoff. Falls back to ordinary shuffle-into-deck
    placement, same treatment as every other curse here without an
    Innate keyword."""
    return Card("Writhe", 0, CardType.CURSE, _unplayable, playable=False)


def make_doubt() -> Card:
    """'While in hand, at the end of your turn, gain 1 Weak' -- the actual
    effect lives in CombatState.end_player_turn (checked by name before the
    hand is cleared), not here; this factory just needs to exist so the
    card can be drawn/discarded/shuffled like any other."""
    return Card("Doubt", 0, CardType.CURSE, _unplayable, playable=False)


def make_shame() -> Card:
    """'While in hand, at the end of your turn, gain 1 Frail' -- effect in
    CombatState.end_player_turn, see Doubt above."""
    return Card("Shame", 0, CardType.CURSE, _unplayable, playable=False)


def make_regret() -> Card:
    """'While in hand, at the end of your turn, lose HP equal to hand size'
    -- effect in CombatState.end_player_turn, see Doubt above."""
    return Card("Regret", 0, CardType.CURSE, _unplayable, playable=False)


def make_decay() -> Card:
    """'While in hand, at the end of your turn, take 2 damage' -- effect in
    CombatState.end_player_turn, see Doubt above."""
    return Card("Decay", 0, CardType.CURSE, _unplayable, playable=False)


def make_pain() -> Card:
    """'While in hand, lose 1 HP whenever you play a card' -- effect in
    CombatState.play_card (checked by name on every other card played),
    not here."""
    return Card("Pain", 0, CardType.CURSE, _unplayable, playable=False)


def make_normality() -> Card:
    """'While in hand, you cannot play more than 3 cards this turn' --
    effect (a legality gate) in CombatState.play_card, not here."""
    return Card("Normality", 0, CardType.CURSE, _unplayable, playable=False)


def _miracle(combat, target):
    combat.player.energy += 1


def make_miracle() -> Card:
    """Not a real card in anyone's deck -- added to hand by Pure Water
    (Watcher's starting relic) at the start of combat."""
    return Card("Miracle", 0, CardType.SKILL, _miracle, targeted=False, exhausts=True)


# --- Ironclad: starter deck ---
#
# These, plus a further ~20-card subset below (see the "Upgrade support"
# section), implement real upgraded ("+") numbers -- pass upgraded=True to
# the factory. Deltas are plausible, hand-entered approximations of the real
# game's values (not verified against datamined game files), consistent with
# every other numeric simplification already noted in this file. Factories
# outside that subset don't accept `upgraded` at all -- calling e.g.
# make_bludgeon(upgraded=True) raises a TypeError, which is deliberate:
# an explicit crash beats a silent no-op that pretends to upgrade a card
# and doesn't.

def _strike_effect(combat, target, dmg):
    combat.deal_attack_damage(combat.player, target, dmg)


def _defend_effect(combat, target, block):
    combat.player.gain_block(block)


def _bash_effect(combat, target, dmg, vuln):
    combat.deal_attack_damage(combat.player, target, dmg)
    target.add_power(Vulnerable(vuln))


def _pommel_strike_effect(combat, target, dmg, draw):
    combat.deal_attack_damage(combat.player, target, dmg)
    combat.draw_cards(draw)


def _iron_wave_effect(combat, target, dmg, block):
    combat.deal_attack_damage(combat.player, target, dmg)
    combat.player.gain_block(block)


def _twin_strike_effect(combat, target, dmg):
    combat.deal_attack_damage(combat.player, target, dmg)
    combat.deal_attack_damage(combat.player, target, dmg)


def _cleave_effect(combat, target, dmg):
    for m in combat.living_monsters:
        combat.deal_attack_damage(combat.player, m, dmg)


def _thunderclap_effect(combat, target, dmg):
    for m in combat.living_monsters:
        combat.deal_attack_damage(combat.player, m, dmg)
        m.add_power(Vulnerable(1))


def _clothesline_effect(combat, target, dmg, weak):
    combat.deal_attack_damage(combat.player, target, dmg)
    target.add_power(Weak(weak))


def _shrug_it_off_effect(combat, target, block):
    combat.player.gain_block(block)
    combat.draw_cards(1)


def _body_slam(combat, target):
    combat.deal_attack_damage(combat.player, target, combat.player.block)


def _flex_effect(combat, target, amount):
    combat.player.add_temp_strength(amount)


def _inflame_effect(combat, target, amount):
    combat.player.add_power(Strength(amount))


def make_strike(upgraded: bool = False) -> Card:
    dmg = 9 if upgraded else 6
    return Card("Strike+" if upgraded else "Strike", 1, CardType.ATTACK,
                lambda combat, target: _strike_effect(combat, target, dmg),
                targeted=True, upgraded=upgraded)


def make_defend(upgraded: bool = False) -> Card:
    block = 8 if upgraded else 5
    return Card("Defend+" if upgraded else "Defend", 1, CardType.SKILL,
                lambda combat, target: _defend_effect(combat, target, block),
                targeted=False, upgraded=upgraded)


def make_bash(upgraded: bool = False) -> Card:
    dmg, vuln = (10, 3) if upgraded else (8, 2)
    return Card("Bash+" if upgraded else "Bash", 2, CardType.ATTACK,
                lambda combat, target: _bash_effect(combat, target, dmg, vuln),
                targeted=True, upgraded=upgraded)


def make_pommel_strike(upgraded: bool = False) -> Card:
    dmg, draw = (10, 2) if upgraded else (9, 1)
    return Card("Pommel Strike+" if upgraded else "Pommel Strike", 1, CardType.ATTACK,
                lambda combat, target: _pommel_strike_effect(combat, target, dmg, draw),
                targeted=True, upgraded=upgraded)


def make_iron_wave(upgraded: bool = False) -> Card:
    dmg, block = (7, 7) if upgraded else (5, 5)
    return Card("Iron Wave+" if upgraded else "Iron Wave", 1, CardType.ATTACK,
                lambda combat, target: _iron_wave_effect(combat, target, dmg, block),
                targeted=True, upgraded=upgraded)


def make_twin_strike(upgraded: bool = False) -> Card:
    dmg = 7 if upgraded else 5
    return Card("Twin Strike+" if upgraded else "Twin Strike", 1, CardType.ATTACK,
                lambda combat, target: _twin_strike_effect(combat, target, dmg),
                targeted=True, upgraded=upgraded)


def make_cleave(upgraded: bool = False) -> Card:
    """AoE attack: hits every living enemy, no single target needed."""
    dmg = 11 if upgraded else 8
    return Card("Cleave+" if upgraded else "Cleave", 1, CardType.ATTACK,
                lambda combat, target: _cleave_effect(combat, target, dmg),
                targeted=False, upgraded=upgraded)


def make_thunderclap(upgraded: bool = False) -> Card:
    """AoE attack + Vulnerable to every living enemy."""
    dmg = 7 if upgraded else 4
    return Card("Thunderclap+" if upgraded else "Thunderclap", 1, CardType.ATTACK,
                lambda combat, target: _thunderclap_effect(combat, target, dmg),
                targeted=False, upgraded=upgraded)


def make_clothesline(upgraded: bool = False) -> Card:
    dmg, weak = (14, 3) if upgraded else (12, 2)
    return Card("Clothesline+" if upgraded else "Clothesline", 2, CardType.ATTACK,
                lambda combat, target: _clothesline_effect(combat, target, dmg, weak),
                targeted=True, upgraded=upgraded)


def make_shrug_it_off(upgraded: bool = False) -> Card:
    block = 11 if upgraded else 8
    return Card("Shrug It Off+" if upgraded else "Shrug It Off", 1, CardType.SKILL,
                lambda combat, target: _shrug_it_off_effect(combat, target, block),
                targeted=False, upgraded=upgraded)


def make_body_slam(upgraded: bool = False) -> Card:
    """Damage equal to current block -- rewards stacking block first.
    Upgrading drops its cost to 0 rather than changing its effect."""
    return Card("Body Slam+" if upgraded else "Body Slam", 0 if upgraded else 1,
                CardType.ATTACK, _body_slam, targeted=True, upgraded=upgraded)


def make_flex(upgraded: bool = False) -> Card:
    """Strength that's clawed back at end of turn (see Creature.add_temp_strength)."""
    amount = 4 if upgraded else 2
    return Card("Flex+" if upgraded else "Flex", 0, CardType.SKILL,
                lambda combat, target: _flex_effect(combat, target, amount),
                targeted=False, upgraded=upgraded)


def make_inflame(upgraded: bool = False) -> Card:
    """Permanent Strength -- a Power card, unlike Flex's temporary version."""
    amount = 3 if upgraded else 2
    return Card("Inflame+" if upgraded else "Inflame", 1, CardType.POWER,
                lambda combat, target: _inflame_effect(combat, target, amount),
                targeted=False, upgraded=upgraded)


def ironclad_starter_deck() -> list[Card]:
    """The Ironclad's real starting deck: 5 Strike, 4 Defend, 1 Bash."""
    deck = [make_strike() for _ in range(5)]
    deck += [make_defend() for _ in range(4)]
    deck.append(make_bash())
    return deck


# --- Ironclad: the rest of the card pool.
#
# This covers nearly every Ironclad common/uncommon/rare from the real game.
# Deliberately NOT implemented, and why:
#   - Rampage, Blood for Blood, Searing Blow: each needs per-copy mutable
#     state (Rampage's damage grows only for THAT card instance across
#     re-draws). Cards are shared-by-reference across search's clone()s for
#     performance (see CombatState.clone's docstring) -- giving one Card
#     object mutable game state would leak that mutation across every clone
#     holding a reference to it, corrupting the search tree. Not worth
#     special-casing the engine for three cards.
#   - Juggernaut: "whenever you gain Block, deal damage to a random enemy"
#     needs Creature.gain_block() to know about `combat`, which it currently
#     doesn't take as a parameter -- would mean threading combat through
#     every gain_block() call site in the codebase for one card.
#   - No Curse cards yet, so curse-triggered clauses on Fire Breathing etc.
#     are simply inert here (only Status cards trigger them).
#   - Upgrades ("+") ARE modeled now, but only for a ~25-card subset spanning
#     starters across all 4 classes plus commonly-relevant common/uncommon/
#     rare Ironclad cards -- pass upgraded=True to a make_*() factory that
#     supports it (it'll raise TypeError if it doesn't). Armaments' own
#     "upgrade a card in hand" CLAUSE is still omitted (there's no in-hand
#     card-upgrade action to model), and Searing Blow's upgrade-scaling
#     (damage scales with how many times it's been upgraded, which requires
#     tracking upgrade *count* per card, not just a binary flag) is skipped
#     along with Searing Blow itself, for the reasons above.
#   - A few "pick a card" effects (Armaments' upgrade target, Dual Wield's
#     duplicate target, Exhume's retrieval target, Burning Pact's exhaust
#     target) have no player-choice UI, so they're resolved by a fixed rule
#     (first/last/random applicable card) instead -- noted per-card below.

def _headbutt(combat, target, dmg=9):
    combat.deal_attack_damage(combat.player, target, dmg)
    if combat.discard_pile:
        combat.draw_pile.append(combat.discard_pile.pop())


def _wild_strike(combat, target, dmg=12):
    combat.deal_attack_damage(combat.player, target, dmg)
    combat.draw_pile.append(make_wound())
    combat.rng.shuffle(combat.draw_pile)


def _perfected_strike(combat, target, base=6):
    whole_deck = combat.hand + combat.draw_pile + combat.discard_pile + combat.exhaust_pile
    strike_count = sum(1 for c in whole_deck if "Strike" in c.name) + 1  # +1: this card itself, already removed from hand
    combat.deal_attack_damage(combat.player, target, base + 2 * strike_count)


def _sword_boomerang(combat, target, hits=3):
    for _ in range(hits):
        living = combat.living_monsters
        if not living:
            break
        t = combat.rng.choice(living)
        combat.deal_attack_damage(combat.player, t, 3)


def _anger(combat, target, dmg=6):
    combat.deal_attack_damage(combat.player, target, dmg)
    combat.discard_pile.append(make_anger())


def _clash_legal(combat) -> bool:
    return all(c.card_type == CardType.ATTACK for c in combat.hand)


def _clash(combat, target, dmg=14):
    combat.deal_attack_damage(combat.player, target, dmg)


def _armaments(combat, target):
    """Simplified: real Armaments also upgrades a card in hand; upgrades
    aren't modeled, so this is just the Block clause."""
    combat.player.gain_block(5)


def _havoc(combat, target):
    if not combat.draw_pile:
        return
    card = combat.draw_pile.pop()
    if card.playable and card.card_type in (CardType.ATTACK, CardType.SKILL):
        auto_target = min(combat.living_monsters, key=lambda m: m.hp, default=None)
        card.play(combat, auto_target)
    combat._exhaust(card)


def _true_grit(combat, target, block=7):
    combat.player.gain_block(block)
    if combat.hand:
        card = combat.rng.choice(combat.hand)
        combat.hand.remove(card)
        combat._exhaust(card)


def _warcry(combat, target, draw=1):
    combat.draw_cards(draw)
    if combat.hand:
        card = combat.hand.pop(0)
        combat.discard_pile.append(card)
        combat.on_discard(card)

def _uppercut(combat, target, stacks=1):
    combat.deal_attack_damage(combat.player, target, 13)
    target.add_power(Weak(stacks))
    target.add_power(Vulnerable(stacks))


def _whirlwind(combat, target, per_hit=5):
    x = combat._x_value
    for _ in range(x):
        for m in combat.living_monsters:
            combat.deal_attack_damage(combat.player, m, per_hit)


def _dropkick(combat, target, dmg=8):
    was_vulnerable = target.has_power("Vulnerable")
    combat.deal_attack_damage(combat.player, target, dmg)
    if was_vulnerable:
        combat.player.energy += 1
        combat.draw_cards(1)


def _pummel(combat, target, hits=4):
    for _ in range(hits):
        combat.deal_attack_damage(combat.player, target, 2)


def _reckless_charge(combat, target, dmg=7):
    combat.deal_attack_damage(combat.player, target, dmg)
    combat.draw_cards(1)
    combat.draw_pile.append(make_dazed())
    combat.rng.shuffle(combat.draw_pile)


def _sever_soul(combat, target, dmg=16):
    combat.deal_attack_damage(combat.player, target, dmg)
    for card in list(combat.hand):
        if card.card_type != CardType.ATTACK:
            combat.hand.remove(card)
            combat._exhaust(card)


def _hemokinesis(combat, target):
    combat.player_loses_hp_from_card(2)
    combat.deal_attack_damage(combat.player, target, 15)


def _disarm(combat, target, str_loss=2):
    target.add_power(Strength(-str_loss))


def _dual_wield(combat, target):
    """Simplified: duplicates the last Attack/Power in hand (no player pick)."""
    candidates = [c for c in combat.hand if c.card_type in (CardType.ATTACK, CardType.POWER)]
    if candidates:
        combat.add_card_to_hand(candidates[-1])


def _entrench(combat, target):
    combat.player.block *= 2


def _ghostly_armor(combat, target, block=10):
    combat.player.gain_block(block)


def _infernal_blade(combat, target):
    attack_factories = [
        make_strike, make_bash, make_twin_strike, make_pommel_strike,
        make_iron_wave, make_cleave, make_clothesline, make_body_slam,
    ]
    combat.add_card_to_hand(combat.rng.choice(attack_factories)())


def _intimidate(combat, target, weak=1):
    for m in combat.living_monsters:
        m.add_power(Weak(weak))


def _power_through(combat, target, block=15):
    combat.player.gain_block(block)
    combat.add_card_to_hand(make_wound())
    combat.add_card_to_hand(make_wound())


def _second_wind(combat, target):
    count = 0
    for card in list(combat.hand):
        if card.card_type != CardType.ATTACK:
            combat.hand.remove(card)
            combat._exhaust(card)
            count += 1
    combat.player.gain_block(5 * count)


def _seeing_red(combat, target):
    combat.player.energy += 2


def _shockwave(combat, target, stacks=3):
    for m in combat.living_monsters:
        m.add_power(Weak(stacks))
        m.add_power(Vulnerable(stacks))


def _spot_weakness(combat, target, str_gain=3):
    from .enemies import IntentType
    if target.intent and target.intent.type in (IntentType.ATTACK, IntentType.ATTACK_DEFEND):
        combat.player.add_power(Strength(str_gain))


def _battle_trance(combat, target, draw=3):
    combat.draw_cards(draw)


def _bloodletting(combat, target, energy=2):
    combat.player_loses_hp_from_card(3)
    combat.player.energy += energy


def _burning_pact(combat, target, draw=2):
    others = [c for c in combat.hand]
    if others:
        card = combat.rng.choice(others)
        combat.hand.remove(card)
        combat._exhaust(card)
    combat.draw_cards(draw)


def _combust_card(combat, target, dmg=5):
    combat.player.add_power(Combust(dmg))


def _dark_embrace(combat, target):
    combat.player.add_power(DarkEmbrace(1))


def _feel_no_pain(combat, target, block=3):
    combat.player.add_power(FeelNoPain(block))


def _evolve(combat, target, draw=1):
    combat.player.add_power(Evolve(draw))


def _metallicize(combat, target, amount=3):
    combat.player.add_power(Metallicize(amount))


def _rage_card(combat, target):
    combat.player.add_power(Rage(3))


def _fire_breathing(combat, target, dmg=6):
    combat.player.add_power(FireBreathing(dmg))


def _rupture_card(combat, target, str_gain=1):
    combat.player.add_power(Rupture(str_gain))


def _bludgeon(combat, target, dmg=32):
    combat.deal_attack_damage(combat.player, target, dmg)


def _immolate(combat, target, dmg=21):
    for m in combat.living_monsters:
        combat.deal_attack_damage(combat.player, m, dmg)
    combat.discard_pile.append(make_burn())


def _offering(combat, target, draw=3):
    combat.player_loses_hp_from_card(6)
    combat.player.energy += 2
    combat.draw_cards(draw)


def _double_tap(combat, target, charges=1):
    combat.double_tap_charges += charges


def _limit_break(combat, target):
    combat.player.add_power(Strength(combat.player.get_power_amount("Strength")))


def _exhume(combat, target):
    if combat.exhaust_pile:
        card = combat.rng.choice(combat.exhaust_pile)
        combat.exhaust_pile.remove(card)
        combat.add_card_to_hand(card)


def _impervious(combat, target, block=30):
    combat.player.gain_block(block)


def _barricade_card(combat, target):
    combat.player.add_power(Barricade(1))


def _demon_form(combat, target, amount=2):
    combat.player.add_power(DemonForm(amount))


def _berserk(combat, target, vuln_stacks=2):
    combat.player.add_power(Vulnerable(vuln_stacks))
    combat.player.add_power(BerserkEnergy(1))


def _brutality(combat, target):
    combat.player.add_power(Brutality(1))


def _corruption_card(combat, target):
    combat.player.add_power(Corruption(1))


def _feed(combat, target, dmg=10):
    combat.deal_attack_damage(combat.player, target, dmg)
    if target.is_dead:
        combat.player.max_hp += 3
        combat.player.hp = min(combat.player.max_hp, combat.player.hp + 3)


def _fiend_fire(combat, target, dmg_per_card=7):
    count = len(combat.hand)
    for card in list(combat.hand):
        combat.hand.remove(card)
        combat._exhaust(card)
    combat.deal_attack_damage(combat.player, target, dmg_per_card * count)


def _reaper(combat, target, dmg=4):
    healed = 0
    for m in combat.living_monsters:
        healed += combat.deal_attack_damage(combat.player, m, dmg)
    combat.player.hp = min(combat.player.max_hp, combat.player.hp + healed)


def make_headbutt(upgraded: bool = False) -> Card:
    dmg = 12 if upgraded else 9
    return Card("Headbutt+" if upgraded else "Headbutt", 1, CardType.ATTACK,
                lambda combat, target: _headbutt(combat, target, dmg),
                targeted=True, upgraded=upgraded)


def make_wild_strike(upgraded: bool = False) -> Card:
    dmg = 17 if upgraded else 12
    return Card("Wild Strike+" if upgraded else "Wild Strike", 1, CardType.ATTACK,
                lambda combat, target: _wild_strike(combat, target, dmg),
                targeted=True, upgraded=upgraded)


def make_perfected_strike(upgraded: bool = False) -> Card:
    base = 9 if upgraded else 6
    return Card("Perfected Strike+" if upgraded else "Perfected Strike", 2, CardType.ATTACK,
                lambda combat, target: _perfected_strike(combat, target, base),
                targeted=True, upgraded=upgraded)


def make_sword_boomerang(upgraded: bool = False) -> Card:
    hits = 4 if upgraded else 3
    return Card("Sword Boomerang+" if upgraded else "Sword Boomerang", 1, CardType.ATTACK,
                lambda combat, target: _sword_boomerang(combat, target, hits),
                targeted=False, upgraded=upgraded)


def make_anger(upgraded: bool = False) -> Card:
    dmg = 8 if upgraded else 6
    return Card("Anger+" if upgraded else "Anger", 0, CardType.ATTACK,
                lambda combat, target: _anger(combat, target, dmg),
                targeted=True, upgraded=upgraded)


def make_clash(upgraded: bool = False) -> Card:
    dmg = 18 if upgraded else 14
    return Card("Clash+" if upgraded else "Clash", 0, CardType.ATTACK,
                lambda combat, target: _clash(combat, target, dmg),
                targeted=True,
                extra_legal_check=_clash_legal, upgraded=upgraded)


def make_armaments() -> Card:
    return Card("Armaments", 1, CardType.SKILL, _armaments, targeted=False)


def make_havoc(upgraded: bool = False) -> Card:
    return Card("Havoc+" if upgraded else "Havoc", 1, CardType.SKILL, _havoc,
                targeted=False, exhausts=True, upgraded=upgraded)


def make_true_grit(upgraded: bool = False) -> Card:
    block = 9 if upgraded else 7
    return Card("True Grit+" if upgraded else "True Grit", 1, CardType.SKILL,
                lambda combat, target: _true_grit(combat, target, block),
                targeted=False, upgraded=upgraded)


def make_warcry(upgraded: bool = False) -> Card:
    draw = 2 if upgraded else 1
    return Card("Warcry+" if upgraded else "Warcry", 0, CardType.SKILL,
                lambda combat, target: _warcry(combat, target, draw),
                targeted=False, exhausts=True, upgraded=upgraded)


def make_uppercut(upgraded: bool = False) -> Card:
    stacks = 2 if upgraded else 1
    return Card("Uppercut+" if upgraded else "Uppercut", 2, CardType.ATTACK,
                lambda combat, target: _uppercut(combat, target, stacks),
                targeted=True, upgraded=upgraded)


def make_whirlwind(upgraded: bool = False) -> Card:
    per_hit = 8 if upgraded else 5
    return Card("Whirlwind+" if upgraded else "Whirlwind", 0, CardType.ATTACK,
                lambda combat, target: _whirlwind(combat, target, per_hit),
                targeted=False, is_x_cost=True, upgraded=upgraded)


def make_dropkick(upgraded: bool = False) -> Card:
    dmg = 12 if upgraded else 8
    return Card("Dropkick+" if upgraded else "Dropkick", 1, CardType.ATTACK,
                lambda combat, target: _dropkick(combat, target, dmg),
                targeted=True, upgraded=upgraded)


def make_pummel(upgraded: bool = False) -> Card:
    hits = 5 if upgraded else 4
    return Card("Pummel+" if upgraded else "Pummel", 1, CardType.ATTACK,
                lambda combat, target: _pummel(combat, target, hits),
                targeted=True, exhausts=True, upgraded=upgraded)


def make_reckless_charge(upgraded: bool = False) -> Card:
    dmg = 10 if upgraded else 7
    return Card("Reckless Charge+" if upgraded else "Reckless Charge", 0, CardType.ATTACK,
                lambda combat, target: _reckless_charge(combat, target, dmg),
                targeted=True, upgraded=upgraded)


def make_sever_soul(upgraded: bool = False) -> Card:
    dmg = 22 if upgraded else 16
    return Card("Sever Soul+" if upgraded else "Sever Soul", 2, CardType.ATTACK,
                lambda combat, target: _sever_soul(combat, target, dmg),
                targeted=True, upgraded=upgraded)


def make_hemokinesis() -> Card:
    return Card("Hemokinesis", 1, CardType.ATTACK, _hemokinesis, targeted=True)


def make_disarm(upgraded: bool = False) -> Card:
    str_loss = 3 if upgraded else 2
    return Card("Disarm+" if upgraded else "Disarm", 1, CardType.SKILL,
                lambda combat, target: _disarm(combat, target, str_loss),
                targeted=True, exhausts=True, upgraded=upgraded)


def make_dual_wield() -> Card:
    return Card("Dual Wield", 1, CardType.SKILL, _dual_wield, targeted=False)


def make_entrench() -> Card:
    return Card("Entrench", 2, CardType.SKILL, _entrench, targeted=False)


def make_ghostly_armor(upgraded: bool = False) -> Card:
    block = 13 if upgraded else 10
    return Card("Ghostly Armor+" if upgraded else "Ghostly Armor", 1, CardType.SKILL,
                lambda combat, target: _ghostly_armor(combat, target, block),
                targeted=False, ethereal=True, upgraded=upgraded)


def make_infernal_blade() -> Card:
    return Card("Infernal Blade", 0, CardType.SKILL, _infernal_blade, targeted=False, exhausts=True)


def make_intimidate(upgraded: bool = False) -> Card:
    weak = 2 if upgraded else 1
    return Card("Intimidate+" if upgraded else "Intimidate", 0, CardType.SKILL,
                lambda combat, target: _intimidate(combat, target, weak),
                targeted=False, exhausts=True, upgraded=upgraded)


def make_power_through(upgraded: bool = False) -> Card:
    block = 20 if upgraded else 15
    return Card("Power Through+" if upgraded else "Power Through", 1, CardType.SKILL,
                lambda combat, target: _power_through(combat, target, block),
                targeted=False, upgraded=upgraded)


def make_second_wind() -> Card:
    return Card("Second Wind", 1, CardType.SKILL, _second_wind, targeted=False, exhausts=True)


def make_seeing_red() -> Card:
    return Card("Seeing Red", 1, CardType.SKILL, _seeing_red, targeted=False, exhausts=True)


def make_shockwave(upgraded: bool = False) -> Card:
    stacks = 5 if upgraded else 3
    return Card("Shockwave+" if upgraded else "Shockwave", 2, CardType.SKILL,
                lambda combat, target: _shockwave(combat, target, stacks),
                targeted=False, exhausts=True, upgraded=upgraded)


def make_spot_weakness(upgraded: bool = False) -> Card:
    str_gain = 4 if upgraded else 3
    return Card("Spot Weakness+" if upgraded else "Spot Weakness", 1, CardType.SKILL,
                lambda combat, target: _spot_weakness(combat, target, str_gain),
                targeted=True, upgraded=upgraded)


def make_battle_trance(upgraded: bool = False) -> Card:
    draw = 4 if upgraded else 3
    return Card("Battle Trance+" if upgraded else "Battle Trance", 0, CardType.SKILL,
                lambda combat, target: _battle_trance(combat, target, draw),
                targeted=False, upgraded=upgraded)


def make_bloodletting(upgraded: bool = False) -> Card:
    energy = 3 if upgraded else 2
    return Card("Bloodletting+" if upgraded else "Bloodletting", 0, CardType.SKILL,
                lambda combat, target: _bloodletting(combat, target, energy),
                targeted=False, upgraded=upgraded)


def make_burning_pact(upgraded: bool = False) -> Card:
    draw = 3 if upgraded else 2
    return Card("Burning Pact+" if upgraded else "Burning Pact", 1, CardType.SKILL,
                lambda combat, target: _burning_pact(combat, target, draw),
                targeted=False, upgraded=upgraded)


def make_combust(upgraded: bool = False) -> Card:
    dmg = 7 if upgraded else 5
    return Card("Combust+" if upgraded else "Combust", 1, CardType.POWER,
                lambda combat, target: _combust_card(combat, target, dmg),
                targeted=False, upgraded=upgraded)


def make_dark_embrace() -> Card:
    return Card("Dark Embrace", 2, CardType.POWER, _dark_embrace, targeted=False)


def make_feel_no_pain(upgraded: bool = False) -> Card:
    block = 4 if upgraded else 3
    return Card("Feel No Pain+" if upgraded else "Feel No Pain", 1, CardType.POWER,
                lambda combat, target: _feel_no_pain(combat, target, block),
                targeted=False, upgraded=upgraded)


def make_evolve(upgraded: bool = False) -> Card:
    draw = 2 if upgraded else 1
    return Card("Evolve+" if upgraded else "Evolve", 1, CardType.POWER,
                lambda combat, target: _evolve(combat, target, draw),
                targeted=False, upgraded=upgraded)


def make_metallicize(upgraded: bool = False) -> Card:
    amount = 4 if upgraded else 3
    return Card("Metallicize+" if upgraded else "Metallicize", 1, CardType.POWER,
                lambda combat, target: _metallicize(combat, target, amount),
                targeted=False, upgraded=upgraded)


def make_rage() -> Card:
    return Card("Rage", 0, CardType.SKILL, _rage_card, targeted=False)


def make_fire_breathing(upgraded: bool = False) -> Card:
    dmg = 10 if upgraded else 6
    return Card("Fire Breathing+" if upgraded else "Fire Breathing", 1, CardType.POWER,
                lambda combat, target: _fire_breathing(combat, target, dmg),
                targeted=False, upgraded=upgraded)


def make_rupture(upgraded: bool = False) -> Card:
    str_gain = 2 if upgraded else 1
    return Card("Rupture+" if upgraded else "Rupture", 1, CardType.POWER,
                lambda combat, target: _rupture_card(combat, target, str_gain),
                targeted=False, upgraded=upgraded)


def make_bludgeon(upgraded: bool = False) -> Card:
    dmg = 42 if upgraded else 32
    return Card("Bludgeon+" if upgraded else "Bludgeon", 3, CardType.ATTACK,
                lambda combat, target: _bludgeon(combat, target, dmg),
                targeted=True, upgraded=upgraded)


def make_immolate(upgraded: bool = False) -> Card:
    dmg = 28 if upgraded else 21
    return Card("Immolate+" if upgraded else "Immolate", 2, CardType.ATTACK,
                lambda combat, target: _immolate(combat, target, dmg),
                targeted=False, upgraded=upgraded)


def make_offering(upgraded: bool = False) -> Card:
    draw = 5 if upgraded else 3
    return Card("Offering+" if upgraded else "Offering", 0, CardType.SKILL,
                lambda combat, target: _offering(combat, target, draw),
                targeted=False, exhausts=True, upgraded=upgraded)


def make_double_tap(upgraded: bool = False) -> Card:
    charges = 2 if upgraded else 1
    return Card("Double Tap+" if upgraded else "Double Tap", 1, CardType.SKILL,
                lambda combat, target: _double_tap(combat, target, charges),
                targeted=False, upgraded=upgraded)


def make_limit_break() -> Card:
    return Card("Limit Break", 1, CardType.SKILL, _limit_break, targeted=False, exhausts=True)


def make_exhume() -> Card:
    return Card("Exhume", 1, CardType.SKILL, _exhume, targeted=False, exhausts=True)


def make_impervious(upgraded: bool = False) -> Card:
    block = 40 if upgraded else 30
    return Card("Impervious+" if upgraded else "Impervious", 2, CardType.SKILL,
                lambda combat, target: _impervious(combat, target, block),
                targeted=False, exhausts=True, upgraded=upgraded)


def make_barricade() -> Card:
    return Card("Barricade", 3, CardType.POWER, _barricade_card, targeted=False)


def make_demon_form(upgraded: bool = False) -> Card:
    amount = 3 if upgraded else 2
    return Card("Demon Form+" if upgraded else "Demon Form", 3, CardType.POWER,
                lambda combat, target: _demon_form(combat, target, amount),
                targeted=False, upgraded=upgraded)


def make_berserk(upgraded: bool = False) -> Card:
    vuln = 1 if upgraded else 2
    return Card("Berserk+" if upgraded else "Berserk", 0, CardType.POWER,
                lambda combat, target: _berserk(combat, target, vuln),
                targeted=False, upgraded=upgraded)


def make_brutality() -> Card:
    return Card("Brutality", 0, CardType.POWER, _brutality, targeted=False)


def make_corruption() -> Card:
    return Card("Corruption", 3, CardType.POWER, _corruption_card, targeted=False)


def make_feed(upgraded: bool = False) -> Card:
    dmg = 12 if upgraded else 10
    return Card("Feed+" if upgraded else "Feed", 1, CardType.ATTACK,
                lambda combat, target: _feed(combat, target, dmg),
                targeted=True, exhausts=True, upgraded=upgraded)


def make_fiend_fire(upgraded: bool = False) -> Card:
    dmg_per_card = 10 if upgraded else 7
    return Card("Fiend Fire+" if upgraded else "Fiend Fire", 1, CardType.ATTACK,
                lambda combat, target: _fiend_fire(combat, target, dmg_per_card),
                targeted=True, upgraded=upgraded)


def make_reaper(upgraded: bool = False) -> Card:
    dmg = 5 if upgraded else 4
    return Card("Reaper+" if upgraded else "Reaper", 2, CardType.ATTACK,
                lambda combat, target: _reaper(combat, target, dmg),
                targeted=False, ethereal=True, upgraded=upgraded)


def _heavy_blade(combat, target, dmg=14, strength_mult=3):
    combat.deal_attack_damage(combat.player, target, dmg, strength_multiplier=strength_mult)


def make_heavy_blade(upgraded: bool = False) -> Card:
    mult = 5 if upgraded else 3
    return Card("Heavy Blade+" if upgraded else "Heavy Blade", 2, CardType.ATTACK,
                lambda combat, target: _heavy_blade(combat, target, 14, mult),
                targeted=True, upgraded=upgraded)


def _carnage(combat, target, dmg=20):
    combat.deal_attack_damage(combat.player, target, dmg)


def make_carnage(upgraded: bool = False) -> Card:
    dmg = 28 if upgraded else 20
    return Card("Carnage+" if upgraded else "Carnage", 2, CardType.ATTACK,
                lambda combat, target: _carnage(combat, target, dmg),
                targeted=True, ethereal=True, upgraded=upgraded)


def _flame_barrier(combat, target, block=12, dmg_back=4):
    combat.player.gain_block(block)
    combat.player.add_power(FlameBarrier(dmg_back))


def make_flame_barrier(upgraded: bool = False) -> Card:
    block, dmg_back = (16, 6) if upgraded else (12, 4)
    return Card("Flame Barrier+" if upgraded else "Flame Barrier", 1, CardType.SKILL,
                lambda combat, target: _flame_barrier(combat, target, block, dmg_back),
                targeted=False, upgraded=upgraded)


def make_rampage(upgraded: bool = False) -> Card:
    """Deal 8 damage; increase by 5(8) each time *this specific card
    instance* is played, for the rest of combat. Each call to
    make_rampage() creates its own closure over `state`, so the persistent
    bonus naturally lives per-instance (this card object specifically, not
    "Rampage" cards in general) without needing any change to the Card
    class itself -- Python closures already give per-instance mutable state
    for free here."""
    state = {"bonus": 0}
    increment = 8 if upgraded else 5

    def _effect(combat, target):
        combat.deal_attack_damage(combat.player, target, 8 + state["bonus"])
        state["bonus"] += increment

    return Card("Rampage+" if upgraded else "Rampage", 1, CardType.ATTACK,
                _effect, targeted=True, upgraded=upgraded)


def make_searing_blow(upgraded: bool = False) -> Card:
    state = {"upgrade_count": 1 if upgraded else 0}

    def _effect(combat, target):
        n = state["upgrade_count"]
        dmg = 12 + n * (n + 7) // 2
        combat.deal_attack_damage(combat.player, target, dmg)

    return Card("Searing Blow+" if upgraded else "Searing Blow", 1, CardType.ATTACK,
                _effect, targeted=True, upgraded=upgraded)


def _sentinel(combat, target, block=5):
    combat.player.gain_block(block)


def make_sentinel(upgraded: bool = False) -> Card:
    """The energy-on-exhaust part of this card is in
    CombatState._exhaust (checked by name, since it's a per-card reaction
    rather than a player-wide power) -- this factory just needs the block
    and to NOT set exhausts=True itself (Sentinel isn't self-exhausting;
    it only reacts if something *else* exhausts it, e.g. Second Wind,
    Corruption on skills)."""
    block = 8 if upgraded else 5
    return Card("Sentinel+" if upgraded else "Sentinel", 1, CardType.SKILL,
                lambda combat, target: _sentinel(combat, target, block),
                targeted=False, upgraded=upgraded)


def make_blood_for_blood(upgraded: bool = False) -> Card:
    """Dynamic cost ("costs 1 less for each time you lose HP this combat")
    is handled in CombatState._effective_cost by name, reading
    player.hp_loss_count_this_combat -- the `cost` field here is just the
    starting/base cost before that reduction."""
    dmg = 22 if upgraded else 18
    return Card("Blood for Blood+" if upgraded else "Blood for Blood", 2, CardType.ATTACK,
                lambda combat, target: combat.deal_attack_damage(combat.player, target, dmg),
                targeted=True, upgraded=upgraded)


def make_juggernaut(upgraded: bool = False) -> Card:
    amount = 7 if upgraded else 5
    return Card("Juggernaut+" if upgraded else "Juggernaut", 2, CardType.POWER,
                lambda combat, target: combat.player.add_power(Juggernaut(amount)),
                targeted=False, upgraded=upgraded)


def common_card_pool() -> list[Card]:
    """Every non-starter Ironclad card implemented, one copy each (~63
    cards -- every real Ironclad card now, none intentionally omitted)."""
    return [
        make_pommel_strike(), make_iron_wave(), make_twin_strike(),
        make_cleave(), make_thunderclap(), make_clothesline(),
        make_shrug_it_off(), make_body_slam(), make_flex(), make_inflame(),
        make_headbutt(), make_wild_strike(), make_perfected_strike(),
        make_sword_boomerang(), make_anger(), make_clash(),
        make_armaments(), make_havoc(), make_true_grit(), make_warcry(),
        make_uppercut(), make_whirlwind(), make_dropkick(), make_pummel(),
        make_reckless_charge(), make_sever_soul(), make_hemokinesis(),
        make_disarm(), make_dual_wield(), make_entrench(),
        make_ghostly_armor(), make_infernal_blade(), make_intimidate(),
        make_power_through(), make_second_wind(), make_seeing_red(),
        make_shockwave(), make_spot_weakness(), make_battle_trance(),
        make_bloodletting(), make_burning_pact(),
        make_combust(), make_dark_embrace(), make_feel_no_pain(),
        make_evolve(), make_metallicize(), make_rage(), make_fire_breathing(),
        make_rupture(),
        make_bludgeon(), make_immolate(), make_offering(), make_double_tap(),
        make_limit_break(), make_exhume(), make_impervious(),
        make_barricade(), make_demon_form(), make_berserk(),
        make_brutality(), make_corruption(), make_feed(), make_fiend_fire(),
        make_reaper(),
        make_heavy_blade(), make_carnage(), make_flame_barrier(),
        make_rampage(), make_searing_blow(), make_sentinel(),
        make_blood_for_blood(), make_juggernaut(),
    ]


def varied_ironclad_deck() -> list[Card]:
    """A richer 11-card deck mixing starters with common-pool cards, for
    exercising AoE/multi-hit/scaling decisions the starter deck can't."""
    deck = [make_strike() for _ in range(3)]
    deck += [make_defend() for _ in range(3)]
    deck += [
        make_bash(), make_twin_strike(), make_iron_wave(), make_cleave(),
        make_clothesline(), make_shrug_it_off(), make_flex(),
    ]
    return deck


def big_ironclad_deck() -> list[Card]:
    """A realistic 30-card midgame deck: the starter 10 plus 20 hand-picked
    cards spanning every mechanic added above -- multi-hit, AoE, block/
    attack hybrids, Strength scaling (temp and permanent), a self-damage +
    Rupture combo, and an exhaust engine (Dark Embrace draws on exhaust,
    Second Wind exhausts non-Attacks for block -- they synergize directly).
    Exists to stress-test the engine and search at real deck-size scale."""
    deck = ironclad_starter_deck()
    deck += [
        make_twin_strike(), make_iron_wave(), make_cleave(), make_thunderclap(),
        make_body_slam(), make_flex(), make_inflame(), make_pommel_strike(),
        make_bloodletting(), make_perfected_strike(), make_uppercut(),
        make_whirlwind(), make_dropkick(), make_entrench(), make_second_wind(),
        make_dark_embrace(), make_feel_no_pain(), make_metallicize(),
        make_demon_form(), make_rupture(),
    ]
    assert len(deck) == 30, f"expected a 30-card deck, got {len(deck)}"
    return deck


# --- Colorless: available to any class (Neow/shop/certain events/relics).
# A running theme below: several real cards ("Discovery", "Forethought",
# "Secret Technique"/"Secret Weapon", "Thinking Ahead") let the player
# *choose* which card/of-3-offered they get. This project doesn't have a
# player-choice-resolution mechanism for that (no UI/search hook for
# "pick one of these options" outside of what a few already-implemented
# Ironclad cards approximate the same way -- see _true_grit's random
# hand-card pick and _warcry's fixed-position pick above) -- so the same
# approximation is used consistently here: an arbitrary/random pick stands
# in for player choice, flagged in each factory's docstring, not silently
# presented as the real mechanic.

def _random_card_of_type(combat, card_type):
    candidates = [c for c in combat.draw_pile if c.card_type == card_type]
    if not candidates:
        return None
    card = combat.rng.choice(candidates)
    combat.draw_pile.remove(card)
    return card


def make_bandage_up(upgraded: bool = False) -> Card:
    amount = 6 if upgraded else 4
    return Card("Bandage Up+" if upgraded else "Bandage Up", 0, CardType.SKILL,
                lambda combat, target: combat.player.heal(amount),
                targeted=False, exhausts=True, upgraded=upgraded)


def _blind(combat, target, all_enemies):
    if all_enemies:
        for m in combat.living_monsters:
            m.add_power(Weak(1))
    else:
        target.add_power(Weak(1))


def make_blind(upgraded: bool = False) -> Card:
    return Card("Blind+" if upgraded else "Blind", 0, CardType.SKILL,
                lambda combat, target: _blind(combat, target, upgraded),
                targeted=not upgraded, upgraded=upgraded)


def _dark_shackles_effect(combat, target, amount):
    target.add_power(Strength(-amount))
    combat.player.add_power(Shackled(amount, target))


def make_dark_shackles(upgraded: bool = False) -> Card:
    amount = 15 if upgraded else 9
    return Card("Dark Shackles+" if upgraded else "Dark Shackles", 0, CardType.SKILL,
                lambda combat, target: _dark_shackles_effect(combat, target, amount),
                targeted=True, exhausts=True, upgraded=upgraded)


def _deep_breath(combat, target, draw):
    combat.draw_pile.extend(combat.discard_pile)
    combat.discard_pile = []
    combat.rng.shuffle(combat.draw_pile)
    combat.draw_cards(draw)


def make_deep_breath(upgraded: bool = False) -> Card:
    draw = 2 if upgraded else 1
    return Card("Deep Breath+" if upgraded else "Deep Breath", 0, CardType.SKILL,
                lambda combat, target: _deep_breath(combat, target, draw),
                targeted=False, upgraded=upgraded)


def _discovery(combat, target):
    """Real text: "Choose 1 of 3 random cards...". Approximated as 1
    random card from the combined Ironclad+Colorless pool (no choice) --
    see the module-level note above."""
    pool = common_card_pool() + colorless_card_pool()
    card = combat.rng.choice(pool)
    card.cost = 0
    combat.add_card_to_hand(card)


def make_discovery(upgraded: bool = False) -> Card:
    return Card("Discovery+" if upgraded else "Discovery", 1, CardType.SKILL,
                _discovery, targeted=False, exhausts=not upgraded, upgraded=upgraded)


def _dramatic_entrance(combat, target, dmg):
    for m in combat.living_monsters:
        combat.deal_attack_damage(combat.player, m, dmg)


def make_dramatic_entrance(upgraded: bool = False) -> Card:
    """Real card is also Innate -- not modeled, same as every other
    unmodeled-Innate card in this file (Writhe, Mind Blast below)."""
    dmg = 12 if upgraded else 8
    return Card("Dramatic Entrance+" if upgraded else "Dramatic Entrance", 0, CardType.ATTACK,
                lambda combat, target: _dramatic_entrance(combat, target, dmg),
                targeted=False, exhausts=True, upgraded=upgraded)


def _enlightenment(combat, target):
    """Upgraded version is "this combat", not just this turn -- approximated
    as this-turn-only for both, since persisting it across future draws
    would need a per-combat flag threaded into CardManager.draw_cards, not
    built for one card's upgrade delta."""
    for c in combat.hand:
        if c.cost > 1:
            c.cost = 1


def make_enlightenment(upgraded: bool = False) -> Card:
    return Card("Enlightenment+" if upgraded else "Enlightenment", 0, CardType.SKILL,
                _enlightenment, targeted=False, upgraded=upgraded)


def make_finesse(upgraded: bool = False) -> Card:
    block = 4 if upgraded else 2

    def _effect(combat, target):
        combat.player.gain_block(block)
        combat.draw_cards(1)

    return Card("Finesse+" if upgraded else "Finesse", 0, CardType.SKILL,
                _effect, targeted=False, upgraded=upgraded)


def make_flash_of_steel(upgraded: bool = False) -> Card:
    dmg = 6 if upgraded else 3

    def _effect(combat, target):
        combat.deal_attack_damage(combat.player, target, dmg)
        combat.draw_cards(1)

    return Card("Flash of Steel+" if upgraded else "Flash of Steel", 0, CardType.ATTACK,
                _effect, targeted=True, upgraded=upgraded)


def _forethought(combat, self_card):
    """Real card lets you choose the card (and, upgraded, any number of
    them); approximated as a single random non-Forethought hand card --
    see the module-level note above. Takes the Forethought card instance
    itself (not the usual monster `target`, since this card is untargeted)
    so it can be excluded from "other cards in hand" -- see
    make_forethought for how that gets threaded in."""
    others = [c for c in combat.hand if c is not self_card]
    if not others:
        return
    card = combat.rng.choice(others)
    combat.hand.remove(card)
    card.cost = 0
    combat.draw_pile.insert(0, card)


def make_forethought(upgraded: bool = False) -> Card:
    # Built in two steps: `play` needs a reference to this exact Card
    # instance (to exclude itself from "other cards in hand"), which
    # doesn't exist until after construction -- so it starts as a no-op
    # and gets a real closure over `c` bound afterward.
    c = Card("Forethought+" if upgraded else "Forethought", 0, CardType.SKILL,
             lambda combat, target: None, targeted=False, upgraded=upgraded)
    c.play = lambda combat, target: _forethought(combat, c)
    return c


def make_good_instincts(upgraded: bool = False) -> Card:
    block = 9 if upgraded else 6
    return Card("Good Instincts+" if upgraded else "Good Instincts", 0, CardType.SKILL,
                lambda combat, target: combat.player.gain_block(block),
                targeted=False, upgraded=upgraded)


def _impatience(combat, target, draw):
    if not any(c.card_type == CardType.ATTACK for c in combat.hand):
        combat.draw_cards(draw)


def make_impatience(upgraded: bool = False) -> Card:
    draw = 3 if upgraded else 2
    return Card("Impatience+" if upgraded else "Impatience", 0, CardType.SKILL,
                lambda combat, target: _impatience(combat, target, draw),
                targeted=False, upgraded=upgraded)


def _jack_of_all_trades(combat, target, count):
    for _ in range(count):
        pool = colorless_card_pool()
        if pool:
            combat.add_card_to_hand(combat.rng.choice(pool))


def make_jack_of_all_trades(upgraded: bool = False) -> Card:
    count = 2 if upgraded else 1
    return Card("Jack of All Trades+" if upgraded else "Jack of All Trades", 0, CardType.SKILL,
                lambda combat, target: _jack_of_all_trades(combat, target, count),
                targeted=False, exhausts=True, upgraded=upgraded)


def _madness(combat, target):
    """Real card picks a RANDOM hand card; approximated the same way (this
    one genuinely is randomness in the real card too, not a simplified
    choice)."""
    if combat.hand:
        card = combat.rng.choice(combat.hand)
        card.cost = 0


def make_madness(upgraded: bool = False) -> Card:
    return Card("Madness+" if upgraded else "Madness", 0 if upgraded else 1, CardType.SKILL,
                _madness, targeted=False, exhausts=True, upgraded=upgraded)


def make_mind_blast(upgraded: bool = False) -> Card:
    """Real card is also Innate -- not modeled, see Dramatic Entrance above."""
    return Card("Mind Blast+" if upgraded else "Mind Blast", 1 if upgraded else 2, CardType.ATTACK,
                lambda combat, target: combat.deal_attack_damage(
                    combat.player, target, len(combat.draw_pile)),
                targeted=True, upgraded=upgraded)


def make_panacea(upgraded: bool = False) -> Card:
    amount = 2 if upgraded else 1
    return Card("Panacea+" if upgraded else "Panacea", 0, CardType.SKILL,
                lambda combat, target: combat.player.add_power(Artifact(amount)),
                targeted=False, exhausts=True, upgraded=upgraded)


def _panic_button(combat, target, block):
    combat.player.gain_block(block)
    combat.player.no_block_from_cards_turns = 2


def make_panic_button(upgraded: bool = False) -> Card:
    block = 40 if upgraded else 30
    return Card("Panic Button+" if upgraded else "Panic Button", 0, CardType.SKILL,
                lambda combat, target: _panic_button(combat, target, block),
                targeted=False, exhausts=True, upgraded=upgraded)


def _purity(combat, self_card, count):
    """Real card lets you choose up to `count` cards; approximated as
    exhausting the first `count` cards in hand (excluding itself) -- see
    the module-level note above."""
    others = [c for c in combat.hand if c is not self_card][:count]
    for c in others:
        combat.hand.remove(c)
        combat._exhaust(c)


def make_purity(upgraded: bool = False) -> Card:
    count = 5 if upgraded else 3
    c = Card("Purity+" if upgraded else "Purity", 0, CardType.SKILL,
             lambda combat, target: None, targeted=False, exhausts=True, upgraded=upgraded)
    c.play = lambda combat, target: _purity(combat, c, count)
    return c


def make_swift_strike(upgraded: bool = False) -> Card:
    dmg = 10 if upgraded else 7
    return Card("Swift Strike+" if upgraded else "Swift Strike", 0, CardType.ATTACK,
                lambda combat, target: combat.deal_attack_damage(combat.player, target, dmg),
                targeted=True, upgraded=upgraded)


def _trip(combat, target, all_enemies):
    if all_enemies:
        for m in combat.living_monsters:
            m.add_power(Vulnerable(2))
    else:
        target.add_power(Vulnerable(2))


def make_trip(upgraded: bool = False) -> Card:
    return Card("Trip+" if upgraded else "Trip", 0, CardType.SKILL,
                lambda combat, target: _trip(combat, target, upgraded),
                targeted=not upgraded, upgraded=upgraded)


def _apotheosis(combat, target):
    for pile in (combat.hand, combat.draw_pile, combat.discard_pile, combat.exhaust_pile):
        for card in pile:
            if card.upgraded:
                continue
            factory_name = "make_" + card.name.lower().replace(" ", "_").replace("-", "_")
            factory = globals().get(factory_name)
            if factory is None or not callable(factory):
                continue
            try:
                upgraded_card = factory(upgraded=True)
            except TypeError:
                continue
            card.name = upgraded_card.name
            card.cost = upgraded_card.cost
            card.play = upgraded_card.play
            card.upgraded = True


def make_apotheosis(upgraded: bool = False) -> Card:
    return Card("Apotheosis+" if upgraded else "Apotheosis", 1 if upgraded else 2, CardType.SKILL,
                _apotheosis, targeted=False, exhausts=True, upgraded=upgraded)


def _chrysalis(combat, target, count):
    for _ in range(count):
        pool = [c for c in common_card_pool() if c.card_type == CardType.SKILL]
        if pool:
            card = combat.rng.choice(pool)
            card.cost = 0
            combat.draw_pile.append(card)
    combat.rng.shuffle(combat.draw_pile)


def make_chrysalis(upgraded: bool = False) -> Card:
    count = 5 if upgraded else 3
    return Card("Chrysalis+" if upgraded else "Chrysalis", 2, CardType.SKILL,
                lambda combat, target: _chrysalis(combat, target, count),
                targeted=False, exhausts=True, upgraded=upgraded)


def _hand_of_greed(combat, target, dmg):
    was_alive = not target.is_dead
    combat.deal_attack_damage(combat.player, target, dmg)
    # Gold isn't modeled anywhere in this combat-only engine (no economy
    # layer exists between fights), so the "if Fatal, gain Gold" half of
    # this card has nothing to attach to -- damage-only, flagged rather
    # than silently dropped without comment.
    _ = was_alive


def make_hand_of_greed(upgraded: bool = False) -> Card:
    dmg = 25 if upgraded else 20
    return Card("Hand of Greed+" if upgraded else "Hand of Greed", 2, CardType.ATTACK,
                lambda combat, target: _hand_of_greed(combat, target, dmg),
                targeted=True, upgraded=upgraded)


def make_magnetism(upgraded: bool = False) -> Card:
    return Card("Magnetism+" if upgraded else "Magnetism", 1 if upgraded else 2, CardType.POWER,
                lambda combat, target: combat.player.add_power(Magnetism()),
                targeted=False, upgraded=upgraded)


def make_master_of_strategy(upgraded: bool = False) -> Card:
    draw = 4 if upgraded else 3
    return Card("Master of Strategy+" if upgraded else "Master of Strategy", 0, CardType.SKILL,
                lambda combat, target: combat.draw_cards(draw),
                targeted=False, exhausts=True, upgraded=upgraded)


def make_mayhem(upgraded: bool = False) -> Card:
    return Card("Mayhem+" if upgraded else "Mayhem", 1 if upgraded else 2, CardType.POWER,
                lambda combat, target: combat.player.add_power(Mayhem()),
                targeted=False, upgraded=upgraded)


def _metamorphosis(combat, target, count):
    for _ in range(count):
        pool = [c for c in common_card_pool() if c.card_type == CardType.ATTACK]
        if pool:
            card = combat.rng.choice(pool)
            card.cost = 0
            combat.draw_pile.append(card)
    combat.rng.shuffle(combat.draw_pile)


def make_metamorphosis(upgraded: bool = False) -> Card:
    count = 5 if upgraded else 3
    return Card("Metamorphosis+" if upgraded else "Metamorphosis", 2, CardType.SKILL,
                lambda combat, target: _metamorphosis(combat, target, count),
                targeted=False, exhausts=True, upgraded=upgraded)


def make_panache(upgraded: bool = False) -> Card:
    amount = 14 if upgraded else 10
    return Card("Panache+" if upgraded else "Panache", 0, CardType.POWER,
                lambda combat, target: combat.player.add_power(Panache(amount)),
                targeted=False, upgraded=upgraded)


def make_sadistic_nature(upgraded: bool = False) -> Card:
    """Real effect ("whenever you apply a debuff to an enemy, they take
    damage") isn't modeled -- debuffs are applied via dozens of separate
    `target.add_power(...)` call sites scattered across every card in this
    file, with no single hook point analogous to Creature.add_power's
    Artifact check above (Artifact intercepts the DEFENDER's own
    add_power; this needs to know who's attacking, which add_power itself
    doesn't). Granting the power is still real; the reactive damage is
    not, flagged here rather than silently dropped."""
    amount = 7 if upgraded else 5
    return Card("Sadistic Nature+" if upgraded else "Sadistic Nature", 0, CardType.POWER,
                lambda combat, target: combat.player.add_power(SadisticNature(amount)),
                targeted=False, upgraded=upgraded)


def make_secret_technique(upgraded: bool = False) -> Card:
    def _effect(combat, target):
        card = _random_card_of_type(combat, CardType.SKILL)
        if card is not None:
            combat.add_card_to_hand(card)

    return Card("Secret Technique+" if upgraded else "Secret Technique", 0, CardType.SKILL,
                _effect, targeted=False, exhausts=not upgraded, upgraded=upgraded)


def make_secret_weapon(upgraded: bool = False) -> Card:
    def _effect(combat, target):
        card = _random_card_of_type(combat, CardType.ATTACK)
        if card is not None:
            combat.add_card_to_hand(card)

    return Card("Secret Weapon+" if upgraded else "Secret Weapon", 0, CardType.SKILL,
                _effect, targeted=False, exhausts=not upgraded, upgraded=upgraded)


def make_the_bomb(upgraded: bool = False) -> Card:
    amount = 50 if upgraded else 40
    return Card("The Bomb+" if upgraded else "The Bomb", 2, CardType.SKILL,
                lambda combat, target: combat.player.add_power(TheBomb(amount, 3)),
                targeted=False, upgraded=upgraded)


def _thinking_ahead(combat, self_card):
    combat.draw_cards(2)
    others = [c for c in combat.hand if c is not self_card]
    if others:
        card = combat.rng.choice(others)
        combat.hand.remove(card)
        combat.draw_pile.insert(0, card)


def make_thinking_ahead(upgraded: bool = False) -> Card:
    c = Card("Thinking Ahead+" if upgraded else "Thinking Ahead", 0, CardType.SKILL,
             lambda combat, target: None, targeted=False, exhausts=not upgraded, upgraded=upgraded)
    c.play = lambda combat, target: _thinking_ahead(combat, c)
    return c


def _transmutation(combat, target):
    x = combat._x_value
    for _ in range(x):
        pool = colorless_card_pool()
        if pool:
            card = combat.rng.choice(pool)
            card.cost = 0
            combat.add_card_to_hand(card)


def make_transmutation() -> Card:
    """No upgraded=True support -- real upgrade makes the generated cards
    themselves upgraded, which runs into the same "no generic
    already-created-instance upgrade lookup" limitation as Apotheosis
    above. Base printing only, same as every other factory in this file
    that doesn't accept `upgraded` at all (see the module header note)."""
    return Card("Transmutation", 0, CardType.SKILL,
                _transmutation, targeted=False, is_x_cost=True, exhausts=True)


def _violence(combat, target, count):
    for _ in range(count):
        card = _random_card_of_type(combat, CardType.ATTACK)
        if card is not None:
            combat.add_card_to_hand(card)


def make_violence(upgraded: bool = False) -> Card:
    count = 4 if upgraded else 3
    return Card("Violence+" if upgraded else "Violence", 0, CardType.SKILL,
                lambda combat, target: _violence(combat, target, count),
                targeted=False, exhausts=True, upgraded=upgraded)


def _bite(combat, target, dmg, heal):
    combat.deal_attack_damage(combat.player, target, dmg)
    combat.player.heal(heal)


def make_bite(upgraded: bool = False) -> Card:
    dmg, heal = (8, 3) if upgraded else (7, 2)
    return Card("Bite+" if upgraded else "Bite", 1, CardType.ATTACK,
                lambda combat, target: _bite(combat, target, dmg, heal),
                targeted=True, upgraded=upgraded)


def _apparition(combat, target):
    combat.player.add_power(Intangible(1))


def make_apparition(upgraded: bool = False) -> Card:
    return Card("Apparition+" if upgraded else "Apparition", 1, CardType.SKILL,
                _apparition, targeted=False, ethereal=True, exhausts=True, upgraded=upgraded)


def _jax(combat, target, str_gain):
    combat.player_loses_hp_from_card(3)
    combat.player.add_power(Strength(str_gain))


def make_jax(upgraded: bool = False) -> Card:
    str_gain = 3 if upgraded else 2
    return Card("J.A.X.+" if upgraded else "J.A.X.", 0, CardType.SKILL,
                lambda combat, target: _jax(combat, target, str_gain),
                targeted=False, upgraded=upgraded)


def make_ritual_dagger(upgraded: bool = False) -> Card:
    state = {"base_dmg": 15}
    increment = 5 if upgraded else 3

    def _effect(combat, target):
        combat.deal_attack_damage(combat.player, target, state["base_dmg"])
        if target.is_dead:
            state["base_dmg"] += increment

    return Card("Ritual Dagger+" if upgraded else "Ritual Dagger", 1, CardType.ATTACK,
                _effect, targeted=True, exhausts=True, upgraded=upgraded)


def colorless_card_pool() -> list[Card]:
    """Every real colorless card implemented, one copy each -- ground-truth
    list cross-checked against sts_lightspeed's own
    ColorlessRarityCardPool::colorlessCardBlob (the C++ engine's verified
    card-pool table), not guessed."""
    return [
        make_bandage_up(), make_blind(), make_dark_shackles(), make_deep_breath(),
        make_discovery(), make_dramatic_entrance(), make_enlightenment(),
        make_finesse(), make_flash_of_steel(), make_forethought(),
        make_good_instincts(), make_impatience(), make_jack_of_all_trades(),
        make_madness(), make_mind_blast(), make_panacea(), make_panic_button(),
        make_purity(), make_swift_strike(), make_trip(),
        make_apotheosis(), make_chrysalis(), make_hand_of_greed(),
        make_magnetism(), make_master_of_strategy(), make_mayhem(),
        make_metamorphosis(), make_panache(), make_sadistic_nature(),
        make_secret_technique(), make_secret_weapon(), make_the_bomb(),
        make_thinking_ahead(), make_transmutation(), make_violence(),
        make_bite(), make_apparition(), make_jax(), make_ritual_dagger(),
    ]


# --- Silent: Poison + Shivs + discard synergy.
# Strike/Defend are shared across all four classes in the real game (same
# 6 dmg / 5 block numbers), so they're reused here too, not redefined.

def _neutralize(combat, target, dmg=3, weak=1):
    combat.deal_attack_damage(combat.player, target, dmg)
    target.add_power(Weak(weak))


def _survivor(combat, target, block=8):
    """Simplified: real Survivor lets you choose which card to discard;
    here it's just the first card left in hand (no choice modeled)."""
    combat.player.gain_block(block)
    if combat.hand:
        card = combat.hand.pop(0)
        combat.discard_pile.append(card)


def _deadly_poison(combat, target, poison):
    target.add_power(Poison(poison))


def _poisoned_stab(combat, target, dmg, poison):
    combat.deal_attack_damage(combat.player, target, dmg)
    target.add_power(Poison(poison))


def _footwork(combat, target, amount):
    combat.player.add_power(Dexterity(amount))


def _sucker_punch(combat, target, dmg, weak):
    combat.deal_attack_damage(combat.player, target, dmg)
    target.add_power(Weak(weak))


def make_shiv() -> Card:
    return Card("Shiv", 0, CardType.ATTACK,
                lambda combat, target: combat.deal_attack_damage(
                    combat.player, target, 4 + combat.player.get_power_amount("Accuracy")),
                targeted=True, exhausts=True)


def _blade_dance(combat, target, count):
    for _ in range(count):
        combat.add_card_to_hand(make_shiv())


def make_neutralize(upgraded: bool = False) -> Card:
    dmg, weak = (4, 2) if upgraded else (3, 1)
    return Card("Neutralize+" if upgraded else "Neutralize", 0, CardType.ATTACK,
                lambda combat, target: _neutralize(combat, target, dmg, weak),
                targeted=True, upgraded=upgraded)


def make_survivor(upgraded: bool = False) -> Card:
    block = 11 if upgraded else 8
    return Card("Survivor+" if upgraded else "Survivor", 1, CardType.SKILL,
                lambda combat, target: _survivor(combat, target, block),
                targeted=False, upgraded=upgraded)


def make_deadly_poison(upgraded: bool = False) -> Card:
    poison = 7 if upgraded else 5
    return Card("Deadly Poison+" if upgraded else "Deadly Poison", 1, CardType.SKILL,
                lambda combat, target: _deadly_poison(combat, target, poison),
                targeted=True, upgraded=upgraded)


def make_poisoned_stab(upgraded: bool = False) -> Card:
    dmg, poison = (8, 4) if upgraded else (6, 3)
    return Card("Poisoned Stab+" if upgraded else "Poisoned Stab", 1, CardType.ATTACK,
                lambda combat, target: _poisoned_stab(combat, target, dmg, poison),
                targeted=True, upgraded=upgraded)


def make_footwork(upgraded: bool = False) -> Card:
    amount = 3 if upgraded else 2
    return Card("Footwork+" if upgraded else "Footwork", 1, CardType.SKILL,
                lambda combat, target: _footwork(combat, target, amount),
                targeted=False, upgraded=upgraded)


def make_sucker_punch(upgraded: bool = False) -> Card:
    dmg, weak = (9, 2) if upgraded else (7, 1)
    return Card("Sucker Punch+" if upgraded else "Sucker Punch", 1, CardType.ATTACK,
                lambda combat, target: _sucker_punch(combat, target, dmg, weak),
                targeted=True, upgraded=upgraded)


def make_blade_dance(upgraded: bool = False) -> Card:
    count = 4 if upgraded else 3
    return Card("Blade Dance+" if upgraded else "Blade Dance", 1, CardType.SKILL,
                lambda combat, target: _blade_dance(combat, target, count),
                targeted=False, upgraded=upgraded)


def silent_starter_deck() -> list[Card]:
    """The Silent's real starting deck: 5 Strike, 5 Defend, Neutralize, Survivor."""
    deck = [make_strike() for _ in range(5)]
    deck += [make_defend() for _ in range(5)]
    deck += [make_neutralize(), make_survivor()]
    return deck


def silent_card_pool() -> list[Card]:
    return [
        make_deadly_poison(), make_poisoned_stab(), make_footwork(),
        make_sucker_punch(), make_blade_dance(),
        make_cloak_and_dagger(), make_sneaky_strike(), make_dagger_spray(),
        make_bane(), make_deflect(), make_dagger_throw(), make_acrobatics(),
        make_quick_slash(), make_slice(), make_backflip(), make_outmaneuver(),
        make_prepared(), make_piercing_wail(), make_dodge_and_roll(),
        make_flying_knee(),
        make_predator(), make_all_out_attack(), make_distraction(),
        make_accuracy(), make_masterful_stab(), make_flechettes(),
        make_concentrate(), make_bouncing_flask(), make_backstab(),
        make_dash(), make_eviscerate(), make_reflex(), make_heel_hook(),
        make_terror(), make_well_laid_plans(), make_finisher(),
        make_escape_plan(), make_calculated_gamble(), make_skewer(),
        make_riddle_with_holes(), make_endless_agony(), make_setup(),
        make_blur(), make_caltrops(), make_choke(), make_expertise(),
        make_tactician(), make_catalyst(), make_leg_sweep(),
        make_crippling_cloud(), make_alchemize(), make_corpse_explosion(),
        make_malaise(), make_phantasmal_killer(), make_die_die_die(),
        make_adrenaline(), make_envenom(), make_doppelganger(), make_burst(),
        make_wraith_form(), make_nightmare(), make_unload(),
        make_after_image(), make_bullet_time(), make_storm_of_steel(),
        make_glass_knife(), make_thousand_cuts(), make_grand_finale(),
        make_noxious_fumes(), make_infinite_blades(),
    ]


# --- Silent card effect functions ---

def _cloak_and_dagger(combat, target, block, shiv_count):
    combat.player.gain_block(block)
    for _ in range(shiv_count):
        combat.add_card_to_hand(make_shiv())


def make_cloak_and_dagger(upgraded: bool = False) -> Card:
    block = 6
    shiv_count = 2 if upgraded else 1
    return Card("Cloak and Dagger+" if upgraded else "Cloak and Dagger", 1, CardType.SKILL,
                lambda combat, target: _cloak_and_dagger(combat, target, block, shiv_count),
                targeted=False, upgraded=upgraded)


def _sneaky_strike(combat, target, dmg):
    combat.deal_attack_damage(combat.player, target, dmg)
    if combat.cards_discarded_this_turn > 0:
        combat.player.energy += 2


def make_sneaky_strike(upgraded: bool = False) -> Card:
    dmg = 16 if upgraded else 12
    return Card("Sneaky Strike+" if upgraded else "Sneaky Strike", 2, CardType.ATTACK,
                lambda combat, target: _sneaky_strike(combat, target, dmg),
                targeted=True, upgraded=upgraded)


def _dagger_spray(combat, target, dmg):
    for m in combat.living_monsters:
        combat.deal_attack_damage(combat.player, m, dmg)
        combat.deal_attack_damage(combat.player, m, dmg)


def make_dagger_spray(upgraded: bool = False) -> Card:
    dmg = 6 if upgraded else 4
    return Card("Dagger Spray+" if upgraded else "Dagger Spray", 1, CardType.ATTACK,
                lambda combat, target: _dagger_spray(combat, target, dmg),
                targeted=False, upgraded=upgraded)


def _bane(combat, target, dmg):
    combat.deal_attack_damage(combat.player, target, dmg)
    if target.has_power("Poison"):
        combat.deal_attack_damage(combat.player, target, dmg)


def make_bane(upgraded: bool = False) -> Card:
    dmg = 10 if upgraded else 7
    return Card("Bane+" if upgraded else "Bane", 1, CardType.ATTACK,
                lambda combat, target: _bane(combat, target, dmg),
                targeted=True, upgraded=upgraded)


def _deflect(combat, target, block):
    combat.player.gain_block(block)


def make_deflect(upgraded: bool = False) -> Card:
    block = 7 if upgraded else 4
    return Card("Deflect+" if upgraded else "Deflect", 0, CardType.SKILL,
                lambda combat, target: _deflect(combat, target, block),
                targeted=False, upgraded=upgraded)


def _dagger_throw(combat, target, dmg):
    combat.deal_attack_damage(combat.player, target, dmg)
    combat.draw_cards(1)
    if combat.hand:
        card = combat.hand.pop(0)
        combat.discard_pile.append(card)
        combat.on_discard(card)


def make_dagger_throw(upgraded: bool = False) -> Card:
    dmg = 12 if upgraded else 9
    return Card("Dagger Throw+" if upgraded else "Dagger Throw", 1, CardType.ATTACK,
                lambda combat, target: _dagger_throw(combat, target, dmg),
                targeted=True, upgraded=upgraded)


def _acrobatics(combat, target, draw, discard):
    combat.draw_cards(draw)
    for _ in range(discard):
        if combat.hand:
            card = combat.hand.pop(0)
            combat.discard_pile.append(card)
            combat.on_discard(card)


def make_acrobatics(upgraded: bool = False) -> Card:
    draw, discard = (4, 1) if upgraded else (3, 1)
    return Card("Acrobatics+" if upgraded else "Acrobatics", 1, CardType.SKILL,
                lambda combat, target: _acrobatics(combat, target, draw, discard),
                targeted=False, upgraded=upgraded)


def _quick_slash(combat, target, dmg):
    combat.deal_attack_damage(combat.player, target, dmg)
    combat.draw_cards(1)


def make_quick_slash(upgraded: bool = False) -> Card:
    dmg = 12 if upgraded else 8
    return Card("Quick Slash+" if upgraded else "Quick Slash", 1, CardType.ATTACK,
                lambda combat, target: _quick_slash(combat, target, dmg),
                targeted=True, upgraded=upgraded)


def make_slice(upgraded: bool = False) -> Card:
    dmg = 9 if upgraded else 6
    return Card("Slice+" if upgraded else "Slice", 0, CardType.ATTACK,
                lambda combat, target: combat.deal_attack_damage(combat.player, target, dmg),
                targeted=True, upgraded=upgraded)


def _backflip(combat, target, block):
    combat.player.gain_block(block)
    combat.draw_cards(2)


def make_backflip(upgraded: bool = False) -> Card:
    block = 8 if upgraded else 5
    return Card("Backflip+" if upgraded else "Backflip", 1, CardType.SKILL,
                lambda combat, target: _backflip(combat, target, block),
                targeted=False, upgraded=upgraded)


def make_outmaneuver(upgraded: bool = False) -> Card:
    energy = 3 if upgraded else 2
    return Card("Outmaneuver+" if upgraded else "Outmaneuver", 1, CardType.SKILL,
                lambda combat, target: combat.player.add_power(NextTurnEnergy(energy)),
                targeted=False, upgraded=upgraded)


def _prepared(combat, target, draw, discard):
    combat.draw_cards(draw)
    for _ in range(discard):
        if combat.hand:
            card = combat.hand.pop(0)
            combat.discard_pile.append(card)
            combat.on_discard(card)


def make_prepared(upgraded: bool = False) -> Card:
    cost = 0 if upgraded else 1
    draw, discard = (2, 2) if upgraded else (1, 1)
    return Card("Prepared+" if upgraded else "Prepared", cost, CardType.SKILL,
                lambda combat, target: _prepared(combat, target, draw, discard),
                targeted=False, upgraded=upgraded)


def _piercing_wail(combat, target, str_loss):
    for m in combat.living_monsters:
        m.add_power(Strength(-str_loss))
    combat.player.add_power(Shackled(str_loss))


def make_piercing_wail(upgraded: bool = False) -> Card:
    str_loss = 8 if upgraded else 6
    return Card("Piercing Wail+" if upgraded else "Piercing Wail", 1, CardType.SKILL,
                lambda combat, target: _piercing_wail(combat, target, str_loss),
                targeted=False, exhausts=True, upgraded=upgraded)


def _dodge_and_roll(combat, target, block):
    combat.player.gain_block(block)
    combat.player.add_power(NextTurnBlock(block))


def make_dodge_and_roll(upgraded: bool = False) -> Card:
    block = 6 if upgraded else 4
    return Card("Dodge and Roll+" if upgraded else "Dodge and Roll", 1, CardType.SKILL,
                lambda combat, target: _dodge_and_roll(combat, target, block),
                targeted=False, upgraded=upgraded)


def _flying_knee(combat, target, dmg):
    combat.deal_attack_damage(combat.player, target, dmg)
    combat.player.add_power(NextTurnEnergy(1))


def make_flying_knee(upgraded: bool = False) -> Card:
    dmg = 11 if upgraded else 8
    return Card("Flying Knee+" if upgraded else "Flying Knee", 1, CardType.ATTACK,
                lambda combat, target: _flying_knee(combat, target, dmg),
                targeted=True, upgraded=upgraded)


def _predator(combat, target, dmg):
    combat.deal_attack_damage(combat.player, target, dmg)
    combat.player.add_power(NextTurnDraw(2))


def make_predator(upgraded: bool = False) -> Card:
    dmg = 20 if upgraded else 15
    return Card("Predator+" if upgraded else "Predator", 2, CardType.ATTACK,
                lambda combat, target: _predator(combat, target, dmg),
                targeted=True, upgraded=upgraded)


def _all_out_attack(combat, target, dmg):
    for m in combat.living_monsters:
        combat.deal_attack_damage(combat.player, m, dmg)
    if combat.hand:
        card = combat.rng.choice(combat.hand)
        combat.hand.remove(card)
        combat.discard_pile.append(card)
        combat.on_discard(card)


def make_all_out_attack(upgraded: bool = False) -> Card:
    dmg = 14 if upgraded else 10
    return Card("All-Out Attack+" if upgraded else "All-Out Attack", 1, CardType.ATTACK,
                lambda combat, target: _all_out_attack(combat, target, dmg),
                targeted=False, upgraded=upgraded)


def _distraction(combat, target):
    skills = [c for c in silent_card_pool() if c.card_type == CardType.SKILL]
    if skills:
        combat.add_card_to_hand(combat.rng.choice(skills))


def make_distraction(upgraded: bool = False) -> Card:
    return Card("Distraction+" if upgraded else "Distraction", 0 if upgraded else 1, CardType.SKILL,
                _distraction, targeted=False, exhausts=True, upgraded=upgraded)


def make_accuracy(upgraded: bool = False) -> Card:
    amount = 6 if upgraded else 4
    return Card("Accuracy+" if upgraded else "Accuracy", 1, CardType.POWER,
                lambda combat, target: combat.player.add_power(AccuracyPower(amount)),
                targeted=False, upgraded=upgraded)


def make_masterful_stab(upgraded: bool = False) -> Card:
    dmg = 16 if upgraded else 12
    return Card("Masterful Stab+" if upgraded else "Masterful Stab", 0, CardType.ATTACK,
                lambda combat, target: combat.deal_attack_damage(combat.player, target, dmg),
                targeted=True, upgraded=upgraded)


def _flechettes(combat, target, dmg):
    skill_count = sum(1 for c in combat.hand if c.card_type == CardType.SKILL)
    for _ in range(skill_count):
        combat.deal_attack_damage(combat.player, target, dmg)


def make_flechettes(upgraded: bool = False) -> Card:
    dmg = 6 if upgraded else 4
    return Card("Flechettes+" if upgraded else "Flechettes", 1, CardType.ATTACK,
                lambda combat, target: _flechettes(combat, target, dmg),
                targeted=True, upgraded=upgraded)


def _concentrate(combat, target, discards):
    discard_count = 0
    for _ in range(discards):
        if combat.hand:
            card = combat.hand.pop(0)
            combat.discard_pile.append(card)
            combat.on_discard(card)
            discard_count += 1
    if discard_count >= discards:
        combat.player.energy += 2


def make_concentrate(upgraded: bool = False) -> Card:
    discards = 2 if upgraded else 3
    return Card("Concentrate+" if upgraded else "Concentrate", 0, CardType.SKILL,
                lambda combat, target: _concentrate(combat, target, discards),
                targeted=False, upgraded=upgraded)


def _bouncing_flask(combat, target, poison, bounces):
    living = combat.living_monsters
    if not living:
        return
    for _ in range(bounces):
        t = combat.rng.choice(living)
        t.add_power(Poison(poison))


def make_bouncing_flask(upgraded: bool = False) -> Card:
    poison, bounces = (4, 4) if upgraded else (3, 4)
    return Card("Bouncing Flask+" if upgraded else "Bouncing Flask", 2, CardType.SKILL,
                lambda combat, target: _bouncing_flask(combat, target, poison, bounces),
                targeted=False, upgraded=upgraded)


def make_backstab(upgraded: bool = False) -> Card:
    dmg = 15 if upgraded else 11
    return Card("Backstab+" if upgraded else "Backstab", 0, CardType.ATTACK,
                lambda combat, target: combat.deal_attack_damage(combat.player, target, dmg),
                targeted=True, exhausts=True, upgraded=upgraded)


def _dash(combat, target, dmg, block):
    combat.deal_attack_damage(combat.player, target, dmg)
    combat.player.gain_block(block)


def make_dash(upgraded: bool = False) -> Card:
    dmg, block = (13, 13) if upgraded else (10, 10)
    return Card("Dash+" if upgraded else "Dash", 2, CardType.ATTACK,
                lambda combat, target: _dash(combat, target, dmg, block),
                targeted=True, upgraded=upgraded)


def make_eviscerate(upgraded: bool = False) -> Card:
    dmg = 9 if upgraded else 7
    def _effect(combat, target):
        for _ in range(3):
            combat.deal_attack_damage(combat.player, target, dmg)
    return Card("Eviscerate+" if upgraded else "Eviscerate", 3, CardType.ATTACK,
                _effect, targeted=True, upgraded=upgraded)


def make_reflex(upgraded: bool = False) -> Card:
    return Card("Reflex+" if upgraded else "Reflex", 0, CardType.SKILL,
                _unplayable, targeted=False, playable=False, upgraded=upgraded)


def _heel_hook(combat, target, dmg):
    had_weak = target.has_power("Weak")
    combat.deal_attack_damage(combat.player, target, dmg)
    if had_weak:
        combat.player.energy += 1
        combat.draw_cards(1)


def make_heel_hook(upgraded: bool = False) -> Card:
    dmg = 8 if upgraded else 5
    return Card("Heel Hook+" if upgraded else "Heel Hook", 1, CardType.ATTACK,
                lambda combat, target: _heel_hook(combat, target, dmg),
                targeted=True, upgraded=upgraded)


def _terror(combat, target):
    target.add_power(Vulnerable(99))


def make_terror(upgraded: bool = False) -> Card:
    return Card("Terror+" if upgraded else "Terror", 0 if upgraded else 1, CardType.SKILL,
                _terror, targeted=True, exhausts=True, upgraded=upgraded)


def make_well_laid_plans(upgraded: bool = False) -> Card:
    retain = 2 if upgraded else 1
    return Card("Well-Laid Plans+" if upgraded else "Well-Laid Plans", 1, CardType.POWER,
                lambda combat, target: combat.player.add_power(WellLaidPlansPower(retain)),
                targeted=False, upgraded=upgraded)


def _finisher(combat, target, dmg):
    attacks = getattr(combat, 'cards_played_this_turn', 0)
    for _ in range(attacks + 1):
        combat.deal_attack_damage(combat.player, target, dmg)


def make_finisher(upgraded: bool = False) -> Card:
    dmg = 8 if upgraded else 6
    return Card("Finisher+" if upgraded else "Finisher", 1, CardType.ATTACK,
                lambda combat, target: _finisher(combat, target, dmg),
                targeted=True, upgraded=upgraded)


def _escape_plan(combat, target, block):
    if combat.draw_pile:
        top = combat.draw_pile[-1]
        combat.draw_cards(1)
        if top.card_type == CardType.SKILL:
            combat.player.gain_block(block)
    else:
        combat.draw_cards(1)


def make_escape_plan(upgraded: bool = False) -> Card:
    block = 5 if upgraded else 3
    return Card("Escape Plan+" if upgraded else "Escape Plan", 0, CardType.SKILL,
                lambda combat, target: _escape_plan(combat, target, block),
                targeted=False, upgraded=upgraded)


def _calculated_gamble(combat, target):
    count = len(combat.hand)
    for card in list(combat.hand):
        combat.hand.remove(card)
        combat.discard_pile.append(card)
        combat.on_discard(card)
    combat.draw_cards(count)


def make_calculated_gamble(upgraded: bool = False) -> Card:
    return Card("Calculated Gamble+" if upgraded else "Calculated Gamble", 0, CardType.SKILL,
                _calculated_gamble, targeted=False, exhausts=not upgraded, upgraded=upgraded)


def make_skewer(upgraded: bool = False) -> Card:
    per_hit = 10 if upgraded else 7
    def _effect(combat, target):
        x = combat._x_value
        for _ in range(x):
            combat.deal_attack_damage(combat.player, target, per_hit)
    return Card("Skewer+" if upgraded else "Skewer", 0, CardType.ATTACK,
                _effect, targeted=True, is_x_cost=True, upgraded=upgraded)


def _riddle_with_holes(combat, target, dmg):
    for _ in range(5):
        combat.deal_attack_damage(combat.player, target, dmg)


def make_riddle_with_holes(upgraded: bool = False) -> Card:
    dmg = 4 if upgraded else 3
    return Card("Riddle with Holes+" if upgraded else "Riddle with Holes", 2, CardType.ATTACK,
                lambda combat, target: _riddle_with_holes(combat, target, dmg),
                targeted=True, upgraded=upgraded)


def make_endless_agony(upgraded: bool = False) -> Card:
    dmg = 6 if upgraded else 4
    return Card("Endless Agony+" if upgraded else "Endless Agony", 0, CardType.ATTACK,
                lambda combat, target: combat.deal_attack_damage(combat.player, target, dmg),
                targeted=True, exhausts=True, upgraded=upgraded)


def make_setup(upgraded: bool = False) -> Card:
    cost = 0 if upgraded else 1
    def _effect(combat, target):
        others = [c for c in combat.hand]
        if others:
            chosen = combat.rng.choice(others)
            combat.player.add_power(NextTurnZeroCostPower(chosen))
    return Card("Setup+" if upgraded else "Setup", cost, CardType.SKILL,
                _effect, targeted=False, upgraded=upgraded)


def make_blur(upgraded: bool = False) -> Card:
    block = 8 if upgraded else 5
    def _effect(combat, target):
        combat.player.gain_block(block)
        combat.player.add_power(BlurPower(1))
    return Card("Blur+" if upgraded else "Blur", 1, CardType.SKILL,
                _effect, targeted=False, upgraded=upgraded)


def make_caltrops(upgraded: bool = False) -> Card:
    amount = 5 if upgraded else 3
    return Card("Caltrops+" if upgraded else "Caltrops", 1, CardType.POWER,
                lambda combat, target: combat.player.add_power(Caltrops(amount)),
                targeted=False, upgraded=upgraded)


def _choke(combat, target, dmg, per_card):
    combat.deal_attack_damage(combat.player, target, dmg)
    combat.player.add_power(ChokePower(per_card, target))


def make_choke(upgraded: bool = False) -> Card:
    dmg, per_card = (15, 5) if upgraded else (12, 3)
    return Card("Choke+" if upgraded else "Choke", 2, CardType.ATTACK,
                lambda combat, target: _choke(combat, target, dmg, per_card),
                targeted=True, upgraded=upgraded)


def _expertise(combat, target, target_hand_size):
    while len(combat.hand) < target_hand_size:
        if not combat.draw_pile and not combat.discard_pile:
            break
        combat.draw_cards(1)


def make_expertise(upgraded: bool = False) -> Card:
    target_size = 7 if upgraded else 6
    return Card("Expertise+" if upgraded else "Expertise", 1, CardType.SKILL,
                lambda combat, target: _expertise(combat, target, target_size),
                targeted=False, upgraded=upgraded)


def make_tactician(upgraded: bool = False) -> Card:
    return Card("Tactician+" if upgraded else "Tactician", 0, CardType.SKILL,
                _unplayable, targeted=False, playable=False, upgraded=upgraded)


def _catalyst(combat, target, mult):
    poison = target.powers.get("Poison")
    if poison is not None:
        poison.amount *= mult


def make_catalyst(upgraded: bool = False) -> Card:
    mult = 3 if upgraded else 2
    return Card("Catalyst+" if upgraded else "Catalyst", 1, CardType.SKILL,
                lambda combat, target: _catalyst(combat, target, mult),
                targeted=True, exhausts=True, upgraded=upgraded)


def _leg_sweep(combat, target, weak, block):
    target.add_power(Weak(weak))
    combat.player.gain_block(block)


def make_leg_sweep(upgraded: bool = False) -> Card:
    weak, block = (3, 14) if upgraded else (2, 11)
    return Card("Leg Sweep+" if upgraded else "Leg Sweep", 2, CardType.SKILL,
                lambda combat, target: _leg_sweep(combat, target, weak, block),
                targeted=True, upgraded=upgraded)


def _crippling_cloud(combat, target, poison, weak):
    for m in combat.living_monsters:
        m.add_power(Poison(poison))
        m.add_power(Weak(weak))


def make_crippling_cloud(upgraded: bool = False) -> Card:
    poison, weak = (7, 2) if upgraded else (4, 2)
    return Card("Crippling Cloud+" if upgraded else "Crippling Cloud", 2, CardType.SKILL,
                lambda combat, target: _crippling_cloud(combat, target, poison, weak),
                targeted=False, exhausts=True, upgraded=upgraded)


def make_alchemize(upgraded: bool = False) -> Card:
    return Card("Alchemize+" if upgraded else "Alchemize", 0 if upgraded else 1, CardType.SKILL,
                lambda combat, target: None, targeted=False, exhausts=True, upgraded=upgraded)


def _corpse_explosion(combat, target, poison):
    target.add_power(Poison(poison))
    target.add_power(CorpseExplosionPower())


def make_corpse_explosion(upgraded: bool = False) -> Card:
    poison = 9 if upgraded else 6
    return Card("Corpse Explosion+" if upgraded else "Corpse Explosion", 2, CardType.SKILL,
                lambda combat, target: _corpse_explosion(combat, target, poison),
                targeted=True, upgraded=upgraded)


def make_malaise(upgraded: bool = False) -> Card:
    def _effect(combat, target):
        x = combat._x_value
        if upgraded:
            str_loss = int(x * 1.5)
            weak_stacks = int(x * 1.5)
        else:
            str_loss = x
            weak_stacks = x
        target.add_power(Strength(-str_loss))
        target.add_power(Weak(weak_stacks))
    return Card("Malaise+" if upgraded else "Malaise", 0, CardType.SKILL,
                _effect, targeted=True, is_x_cost=True, exhausts=True, upgraded=upgraded)


def make_phantasmal_killer(upgraded: bool = False) -> Card:
    return Card("Phantasmal Killer+" if upgraded else "Phantasmal Killer", 0 if upgraded else 1,
                CardType.SKILL,
                lambda combat, target: combat.player.add_power(PhantasmalKillerPower(1)),
                targeted=False, upgraded=upgraded)


def _die_die_die(combat, target, dmg):
    for m in combat.living_monsters:
        combat.deal_attack_damage(combat.player, m, dmg)


def make_die_die_die(upgraded: bool = False) -> Card:
    dmg = 17 if upgraded else 13
    return Card("Die Die Die+" if upgraded else "Die Die Die", 1, CardType.ATTACK,
                lambda combat, target: _die_die_die(combat, target, dmg),
                targeted=False, exhausts=True, upgraded=upgraded)


def _adrenaline(combat, target, energy, draw):
    combat.player.energy += energy
    combat.draw_cards(draw)


def make_adrenaline(upgraded: bool = False) -> Card:
    energy, draw = (2, 2) if upgraded else (1, 2)
    return Card("Adrenaline+" if upgraded else "Adrenaline", 0, CardType.SKILL,
                lambda combat, target: _adrenaline(combat, target, energy, draw),
                targeted=False, exhausts=True, upgraded=upgraded)


def make_envenom(upgraded: bool = False) -> Card:
    return Card("Envenom+" if upgraded else "Envenom", 1 if upgraded else 2, CardType.POWER,
                lambda combat, target: combat.player.add_power(Envenom(1)),
                targeted=False, upgraded=upgraded)


def make_doppelganger(upgraded: bool = False) -> Card:
    def _effect(combat, target):
        x = combat._x_value
        bonus = 1 if upgraded else 0
        combat.player.add_power(DoppelgangerPower(x + bonus))
    return Card("Doppelganger+" if upgraded else "Doppelganger", 0, CardType.SKILL,
                _effect, targeted=False, is_x_cost=True, exhausts=True, upgraded=upgraded)


def make_burst(upgraded: bool = False) -> Card:
    charges = 2 if upgraded else 1
    return Card("Burst+" if upgraded else "Burst", 1, CardType.SKILL,
                lambda combat, target: setattr(combat, 'burst_charges', combat.burst_charges + charges),
                targeted=False, upgraded=upgraded)


def make_wraith_form(upgraded: bool = False) -> Card:
    intangible = 3 if upgraded else 2
    return Card("Wraith Form+" if upgraded else "Wraith Form", 3, CardType.POWER,
                lambda combat, target: combat.player.add_power(WraithForm(intangible)),
                targeted=False, upgraded=upgraded)


def make_nightmare(upgraded: bool = False) -> Card:
    cost = 2 if upgraded else 3
    def _effect(combat, target):
        if combat.hand:
            chosen = combat.rng.choice(combat.hand)
            combat.player.add_power(NightmarePower(chosen))
    return Card("Nightmare+" if upgraded else "Nightmare", cost, CardType.SKILL,
                _effect, targeted=False, exhausts=True, upgraded=upgraded)


def _unload(combat, target, dmg):
    combat.deal_attack_damage(combat.player, target, dmg)
    for card in list(combat.hand):
        if card.card_type != CardType.ATTACK:
            combat.hand.remove(card)
            combat.discard_pile.append(card)
            combat.on_discard(card)


def make_unload(upgraded: bool = False) -> Card:
    dmg = 18 if upgraded else 14
    return Card("Unload+" if upgraded else "Unload", 1, CardType.ATTACK,
                lambda combat, target: _unload(combat, target, dmg),
                targeted=True, upgraded=upgraded)


def make_after_image(upgraded: bool = False) -> Card:
    return Card("After Image+" if upgraded else "After Image", 1, CardType.POWER,
                lambda combat, target: combat.player.add_power(AfterImage(1)),
                targeted=False, upgraded=upgraded)


def make_bullet_time(upgraded: bool = False) -> Card:
    def _effect(combat, target):
        for c in combat.hand:
            c.cost = 0
    return Card("Bullet Time+" if upgraded else "Bullet Time", 2 if upgraded else 3,
                CardType.SKILL, _effect, targeted=False, upgraded=upgraded)


def make_storm_of_steel(upgraded: bool = False) -> Card:
    def _effect(combat, target):
        for card in list(combat.hand):
            combat.hand.remove(card)
            combat.discard_pile.append(card)
            combat.on_discard(card)
            shiv = make_shiv()
            if upgraded:
                shiv.play = lambda combat, target: combat.deal_attack_damage(combat.player, target, 6)
            combat.add_card_to_hand(shiv)
    return Card("Storm of Steel+" if upgraded else "Storm of Steel", 1, CardType.SKILL,
                _effect, targeted=False, upgraded=upgraded)


def make_glass_knife(upgraded: bool = False) -> Card:
    state = {"dmg": 12 if upgraded else 8}
    def _effect(combat, target):
        dmg = state["dmg"]
        combat.deal_attack_damage(combat.player, target, dmg)
        combat.deal_attack_damage(combat.player, target, dmg)
        state["dmg"] = max(2, dmg - 2)
    return Card("Glass Knife+" if upgraded else "Glass Knife", 1, CardType.ATTACK,
                _effect, targeted=True, upgraded=upgraded)


def make_thousand_cuts(upgraded: bool = False) -> Card:
    amount = 2 if upgraded else 1
    return Card("A Thousand Cuts+" if upgraded else "A Thousand Cuts", 2, CardType.POWER,
                lambda combat, target: combat.player.add_power(ThousandCuts(amount)),
                targeted=False, upgraded=upgraded)


def _grand_finale_legal(combat) -> bool:
    return len(combat.draw_pile) == 0 and len(combat.hand) == 1


def _grand_finale(combat, target, dmg):
    for m in combat.living_monsters:
        combat.deal_attack_damage(combat.player, m, dmg)


def make_grand_finale(upgraded: bool = False) -> Card:
    dmg = 60 if upgraded else 50
    return Card("Grand Finale+" if upgraded else "Grand Finale", 0, CardType.ATTACK,
                lambda combat, target: _grand_finale(combat, target, dmg),
                targeted=False, extra_legal_check=_grand_finale_legal, upgraded=upgraded)


def make_noxious_fumes(upgraded: bool = False) -> Card:
    amount = 3 if upgraded else 2
    return Card("Noxious Fumes+" if upgraded else "Noxious Fumes", 1, CardType.POWER,
                lambda combat, target: combat.player.add_power(NoxiousFumes(amount)),
                targeted=False, upgraded=upgraded)


def make_infinite_blades(upgraded: bool = False) -> Card:
    return Card("Infinite Blades+" if upgraded else "Infinite Blades", 1, CardType.POWER,
                lambda combat, target: combat.player.add_power(InfiniteBlades(1)),
                targeted=False, upgraded=upgraded)


# --- Defect: Orbs + Focus. See sts/orbs.py for the Lightning/Frost mechanics
# (Dark and Plasma orbs are not implemented).

def _zap(combat, target):
    combat.channel_orb(make_lightning_orb())


def _dualcast(combat, target, extra_triggers=1):
    """Simplified from the real "evoke your next orb twice": immediately
    re-triggers the passive of the most recently channeled orb.
    Upgrading (extra_triggers=2) re-triggers it twice instead of once."""
    if combat.player.orbs:
        for _ in range(extra_triggers):
            combat.player.orbs[-1].passive(combat.player, combat)


def _ball_lightning(combat, target):
    combat.deal_attack_damage(combat.player, target, 7)
    combat.channel_orb(make_lightning_orb())


def _cold_snap(combat, target):
    combat.deal_attack_damage(combat.player, target, 6)
    combat.channel_orb(make_frost_orb())


def _coolheaded(combat, target):
    combat.channel_orb(make_frost_orb())
    combat.draw_cards(1)


def _focus_card(combat, target):
    combat.player.add_power(Focus(2))


def make_zap() -> Card:
    return Card("Zap", 1, CardType.SKILL, _zap, targeted=False)


def make_dualcast(upgraded: bool = False) -> Card:
    extra = 2 if upgraded else 1
    return Card("Dualcast+" if upgraded else "Dualcast", 1, CardType.SKILL,
                lambda combat, target: _dualcast(combat, target, extra),
                targeted=False, upgraded=upgraded)


def make_ball_lightning() -> Card:
    return Card("Ball Lightning", 1, CardType.ATTACK, _ball_lightning, targeted=True)


def make_cold_snap() -> Card:
    return Card("Cold Snap", 1, CardType.ATTACK, _cold_snap, targeted=True)


def make_coolheaded() -> Card:
    return Card("Coolheaded", 1, CardType.SKILL, _coolheaded, targeted=False)


def make_focus_card() -> Card:
    return Card("Focus", 1, CardType.POWER, _focus_card, targeted=False)


def defect_starter_deck() -> list[Card]:
    """The Defect's real starting deck: 4 Strike, 4 Defend, Zap, Dualcast."""
    deck = [make_strike() for _ in range(4)]
    deck += [make_defend() for _ in range(4)]
    deck += [make_zap(), make_dualcast()]
    return deck



# --- Void status card (added by Turbo) ---

def _void_noop(combat, target):
    pass


def make_void() -> Card:
    return Card("Void", 0, CardType.STATUS, _void_noop, playable=False, ethereal=True)


# --- Defect: Common cards ---

def _go_for_the_eyes(combat, target, dmg, weak_amt):
    combat.deal_attack_damage(combat.player, target, dmg)
    from .enemies import IntentType
    if target.intent and target.intent.type == IntentType.ATTACK:
        target.add_power(Weak(weak_amt))


def make_go_for_the_eyes(upgraded: bool = False) -> Card:
    dmg, weak = (4, 2) if upgraded else (3, 1)
    return Card("Go for the Eyes+" if upgraded else "Go for the Eyes", 0, CardType.ATTACK,
                lambda combat, target: _go_for_the_eyes(combat, target, dmg, weak),
                targeted=True, upgraded=upgraded)


def make_streamline(upgraded: bool = False) -> Card:
    base_dmg = 20 if upgraded else 15
    c = Card("Streamline+" if upgraded else "Streamline", 2, CardType.ATTACK,
             lambda combat, target: None, targeted=True, upgraded=upgraded)

    def _effect(combat, target):
        combat.deal_attack_damage(combat.player, target, base_dmg)
        c.cost = max(0, c.cost - 1)

    c.play = _effect
    return c


def _recursion(combat, target, evoke_count):
    if not combat.player.orbs:
        return
    for _ in range(evoke_count):
        if not combat.player.orbs:
            break
        orb = combat.player.orbs[0]
        orb.evoke(combat.player, combat)
        combat.player.orbs.pop(0)
        if orb.name == "Lightning":
            combat.channel_orb(make_lightning_orb())
        elif orb.name == "Frost":
            combat.channel_orb(make_frost_orb())
        elif orb.name == "Dark":
            combat.channel_orb(make_dark_orb())
        elif orb.name == "Plasma":
            combat.channel_orb(make_plasma_orb())


def make_recursion(upgraded: bool = False) -> Card:
    evoke_count = 2 if upgraded else 1
    return Card("Recursion+" if upgraded else "Recursion", 1, CardType.SKILL,
                lambda combat, target: _recursion(combat, target, evoke_count),
                targeted=False, upgraded=upgraded)


def _compile_driver(combat, target, dmg):
    combat.deal_attack_damage(combat.player, target, dmg)
    unique_types = set(orb.name for orb in combat.player.orbs)
    combat.draw_cards(len(unique_types))


def make_compile_driver(upgraded: bool = False) -> Card:
    dmg = 10 if upgraded else 7
    return Card("Compile Driver+" if upgraded else "Compile Driver", 1, CardType.ATTACK,
                lambda combat, target: _compile_driver(combat, target, dmg),
                targeted=True, upgraded=upgraded)


def _barrage(combat, target, per_orb):
    hits = combat.player.orb_channeled_count
    for _ in range(hits):
        combat.deal_attack_damage(combat.player, target, per_orb)


def make_barrage(upgraded: bool = False) -> Card:
    per_orb = 6 if upgraded else 4
    return Card("Barrage+" if upgraded else "Barrage", 1, CardType.ATTACK,
                lambda combat, target: _barrage(combat, target, per_orb),
                targeted=True, upgraded=upgraded)


def _stack(combat, target, base):
    combat.player.gain_block(len(combat.discard_pile) + base)


def make_stack(upgraded: bool = False) -> Card:
    base = 5 if upgraded else 3
    return Card("Stack+" if upgraded else "Stack", 1, CardType.SKILL,
                lambda combat, target: _stack(combat, target, base),
                targeted=False, upgraded=upgraded)


def _rebound(combat, target, dmg):
    combat.deal_attack_damage(combat.player, target, dmg)


def make_rebound(upgraded: bool = False) -> Card:
    dmg = 12 if upgraded else 9
    return Card("Rebound+" if upgraded else "Rebound", 1, CardType.ATTACK,
                lambda combat, target: _rebound(combat, target, dmg),
                targeted=True, upgraded=upgraded)


def make_claw(upgraded: bool = False) -> Card:
    dmg = 5 if upgraded else 3

    bonus = [0]

    def _effect(combat, target):
        combat.deal_attack_damage(combat.player, target, dmg + bonus[0])
        bonus[0] += 2

    return Card("Claw+" if upgraded else "Claw", 0, CardType.ATTACK,
                _effect, targeted=True, upgraded=upgraded)


def _turbo(combat, target, energy):
    combat.player.energy += energy
    combat.discard_pile.append(make_void())


def make_turbo(upgraded: bool = False) -> Card:
    energy = 3 if upgraded else 2
    return Card("Turbo+" if upgraded else "Turbo", 0, CardType.SKILL,
                lambda combat, target: _turbo(combat, target, energy),
                targeted=False, upgraded=upgraded)


def _sweeping_beam(combat, target, dmg):
    for m in combat.living_monsters:
        combat.deal_attack_damage(combat.player, m, dmg)
    combat.draw_cards(1)


def make_sweeping_beam(upgraded: bool = False) -> Card:
    dmg = 9 if upgraded else 6
    return Card("Sweeping Beam+" if upgraded else "Sweeping Beam", 1, CardType.ATTACK,
                lambda combat, target: _sweeping_beam(combat, target, dmg),
                targeted=False, upgraded=upgraded)


def _charge_battery(combat, target, block):
    combat.player.gain_block(block)
    combat.player.add_power(NextTurnEnergy(1))


def make_charge_battery(upgraded: bool = False) -> Card:
    block = 10 if upgraded else 7
    return Card("Charge Battery+" if upgraded else "Charge Battery", 1, CardType.SKILL,
                lambda combat, target: _charge_battery(combat, target, block),
                targeted=False, upgraded=upgraded)


def make_hologram(upgraded: bool = False) -> Card:
    block = 5 if upgraded else 3
    exhausts = not upgraded
    c = Card("Hologram+" if upgraded else "Hologram", 1, CardType.SKILL,
             lambda combat, target: None, targeted=False, exhausts=exhausts, upgraded=upgraded)

    def _effect(combat, target):
        combat.player.gain_block(block)
        others = [card for card in combat.discard_pile if card is not c]
        if others:
            card = combat.rng.choice(others)
            combat.discard_pile.remove(card)
            combat.add_card_to_hand(card)

    c.play = _effect
    return c


def _beam_cell(combat, target, dmg, vuln):
    combat.deal_attack_damage(combat.player, target, dmg)
    target.add_power(Vulnerable(vuln))


def make_beam_cell(upgraded: bool = False) -> Card:
    dmg, vuln = (4, 2) if upgraded else (3, 1)
    return Card("Beam Cell+" if upgraded else "Beam Cell", 0, CardType.ATTACK,
                lambda combat, target: _beam_cell(combat, target, dmg, vuln),
                targeted=True, upgraded=upgraded)


def _leap(combat, target, block):
    combat.player.gain_block(block)


def make_leap(upgraded: bool = False) -> Card:
    block = 12 if upgraded else 9
    return Card("Leap+" if upgraded else "Leap", 1, CardType.SKILL,
                lambda combat, target: _leap(combat, target, block),
                targeted=False, upgraded=upgraded)


def make_steam_barrier(upgraded: bool = False) -> Card:
    initial_block = 8 if upgraded else 6
    state = {"block_amount": initial_block}

    def _effect(combat, target):
        combat.player.gain_block(state["block_amount"])
        state["block_amount"] = max(0, state["block_amount"] - 1)

    return Card("Steam Barrier+" if upgraded else "Steam Barrier", 0, CardType.SKILL,
                _effect, targeted=False, upgraded=upgraded)


# --- Defect: Uncommon cards ---

def _doom_and_gloom(combat, target, dmg):
    for m in combat.living_monsters:
        combat.deal_attack_damage(combat.player, m, dmg)
    combat.channel_orb(make_dark_orb())
    combat.player.orb_channeled_count += 1


def make_doom_and_gloom(upgraded: bool = False) -> Card:
    dmg = 14 if upgraded else 10
    return Card("Doom and Gloom+" if upgraded else "Doom and Gloom", 2, CardType.ATTACK,
                lambda combat, target: _doom_and_gloom(combat, target, dmg),
                targeted=False, upgraded=upgraded)


def _defragment(combat, target, amount):
    combat.player.add_power(Focus(amount))


def make_defragment(upgraded: bool = False) -> Card:
    amount = 2 if upgraded else 1
    return Card("Defragment+" if upgraded else "Defragment", 1, CardType.POWER,
                lambda combat, target: _defragment(combat, target, amount),
                targeted=False, upgraded=upgraded)


def _capacitor(combat, target, slots):
    combat.player.orb_slots += slots


def make_capacitor(upgraded: bool = False) -> Card:
    cost = 0 if upgraded else 1
    slots = 3 if upgraded else 2
    return Card("Capacitor+" if upgraded else "Capacitor", cost, CardType.POWER,
                lambda combat, target: _capacitor(combat, target, slots),
                targeted=False, upgraded=upgraded)


def _white_noise(combat, target):
    from .cards import common_card_pool, silent_card_pool, defect_card_pool, watcher_card_pool
    all_power_pools = [common_card_pool(), silent_card_pool(),
                       defect_card_pool(), watcher_card_pool()]
    all_cards = [c for pool in all_power_pools for c in pool if c.card_type == CardType.POWER]
    if all_cards:
        card = combat.rng.choice(all_cards)
        card.cost = 0
        combat.add_card_to_hand(card)


def make_white_noise(upgraded: bool = False) -> Card:
    return Card("White Noise+" if upgraded else "White Noise", 0 if upgraded else 1, CardType.SKILL,
                _white_noise, targeted=False, exhausts=True, upgraded=upgraded)


def _skim(combat, target, draw):
    combat.draw_cards(draw)


def make_skim(upgraded: bool = False) -> Card:
    draw = 4 if upgraded else 3
    return Card("Skim+" if upgraded else "Skim", 1, CardType.SKILL,
                lambda combat, target: _skim(combat, target, draw),
                targeted=False, upgraded=upgraded)


def make_recycle(upgraded: bool = False) -> Card:
    cost = 0
    c = Card("Recycle+" if upgraded else "Recycle", cost, CardType.SKILL,
             lambda combat, target: None, targeted=False, exhausts=True, upgraded=upgraded)

    def _effect(combat, target):
        others = [card for card in combat.hand if card is not c]
        if not others:
            return
        card = combat.rng.choice(others)
        combat.hand.remove(card)
        combat._exhaust(card)
        combat.player.energy += card.cost

    c.play = _effect
    return c


def make_scrape(upgraded: bool = False) -> Card:
    dmg = 10 if upgraded else 7
    draw = 5 if upgraded else 4

    def _effect(combat, target):
        combat.deal_attack_damage(combat.player, target, dmg)
        combat.draw_cards(draw)
        to_discard = [c for c in combat.hand if c.cost != 0]
        for c in to_discard[:draw]:
            combat.hand.remove(c)
            combat.discard_pile.append(c)
            combat.on_discard(c)

    return Card("Scrape+" if upgraded else "Scrape", 0, CardType.ATTACK,
                _effect, targeted=True, upgraded=upgraded)


def _bullseye(combat, target, dmg):
    combat.deal_attack_damage(combat.player, target, dmg)
    if not target.has_power("Lock-On"):
        target.add_power(Strength(0))


def make_bullseye(upgraded: bool = False) -> Card:
    dmg = 11 if upgraded else 8
    return Card("Bullseye+" if upgraded else "Bullseye", 1, CardType.ATTACK,
                lambda combat, target: _bullseye(combat, target, dmg),
                targeted=True, upgraded=upgraded)


def _reprogram(combat, target, amount):
    combat.player.add_power(Strength(amount))
    combat.player.add_power(Dexterity(amount))
    combat.player.add_power(Focus(-1))


def make_reprogram(upgraded: bool = False) -> Card:
    amount = 2 if upgraded else 1
    return Card("Reprogram+" if upgraded else "Reprogram", 1, CardType.SKILL,
                lambda combat, target: _reprogram(combat, target, amount),
                targeted=False, upgraded=upgraded)


def _auto_shields(combat, target, block):
    if combat.player.block == 0:
        combat.player.gain_block(block)


def make_auto_shields(upgraded: bool = False) -> Card:
    block = 15 if upgraded else 11
    return Card("Auto-Shields+" if upgraded else "Auto-Shields", 1, CardType.SKILL,
                lambda combat, target: _auto_shields(combat, target, block),
                targeted=False, upgraded=upgraded)


def _reinforced_body(combat, target, per_block):
    x = combat._x_value
    for _ in range(x):
        combat.player.gain_block(per_block)


def make_reinforced_body(upgraded: bool = False) -> Card:
    per_block = 9 if upgraded else 7
    return Card("Reinforced Body+" if upgraded else "Reinforced Body", 0, CardType.SKILL,
                lambda combat, target: _reinforced_body(combat, target, per_block),
                targeted=False, is_x_cost=True, upgraded=upgraded)


def _double_energy(combat, target):
    combat.player.energy *= 2


def make_double_energy(upgraded: bool = False) -> Card:
    return Card("Double Energy+" if upgraded else "Double Energy", 0 if upgraded else 1, CardType.SKILL,
                _double_energy, targeted=False, exhausts=True, upgraded=upgraded)


def _darkness(combat, target):
    orb = make_dark_orb()
    combat.channel_orb(orb)
    combat.player.orb_channeled_count += 1
    orb.passive(combat.player, combat)


def make_darkness(upgraded: bool = False) -> Card:
    return Card("Darkness+" if upgraded else "Darkness", 1, CardType.SKILL,
                _darkness, targeted=False, upgraded=upgraded)


def _rip_and_tear(combat, target, dmg):
    for _ in range(2):
        living = combat.living_monsters
        if not living:
            break
        t = combat.rng.choice(living)
        combat.deal_attack_damage(combat.player, t, dmg)


def make_rip_and_tear(upgraded: bool = False) -> Card:
    dmg = 9 if upgraded else 7
    return Card("Rip and Tear+" if upgraded else "Rip and Tear", 1, CardType.ATTACK,
                lambda combat, target: _rip_and_tear(combat, target, dmg),
                targeted=False, upgraded=upgraded)


def _ftl(combat, target, dmg, threshold):
    combat.deal_attack_damage(combat.player, target, dmg)
    if combat.cards_played_this_turn < threshold:
        combat.draw_cards(1)


def make_ftl(upgraded: bool = False) -> Card:
    dmg, threshold = (6, 4) if upgraded else (5, 3)
    return Card("FTL+" if upgraded else "FTL", 0, CardType.ATTACK,
                lambda combat, target: _ftl(combat, target, dmg, threshold),
                targeted=True, upgraded=upgraded)


def _force_field(combat, target, block):
    combat.player.gain_block(block)


def make_force_field(upgraded: bool = False) -> Card:
    base_cost = 3 if upgraded else 4
    block = 16 if upgraded else 12
    return Card("Force Field+" if upgraded else "Force Field", base_cost, CardType.SKILL,
                lambda combat, target: _force_field(combat, target, block),
                targeted=False, upgraded=upgraded)


def _equilibrium(combat, target, block):
    combat.player.gain_block(block)


def make_equilibrium(upgraded: bool = False) -> Card:
    block = 16 if upgraded else 13
    return Card("Equilibrium+" if upgraded else "Equilibrium", 2, CardType.SKILL,
                lambda combat, target: _equilibrium(combat, target, block),
                targeted=False, upgraded=upgraded)


def _tempest(combat, target, extra):
    x = combat._x_value
    for _ in range(x + extra):
        combat.channel_orb(make_lightning_orb())
        combat.player.orb_channeled_count += 1
        combat.player.lightning_channeled += 1


def make_tempest(upgraded: bool = False) -> Card:
    extra = 1 if upgraded else 0
    return Card("Tempest+" if upgraded else "Tempest", 0, CardType.SKILL,
                lambda combat, target: _tempest(combat, target, extra),
                targeted=False, is_x_cost=True, exhausts=True, upgraded=upgraded)


def _boot_sequence(combat, target, block):
    combat.player.gain_block(block)


def make_boot_sequence(upgraded: bool = False) -> Card:
    block = 13 if upgraded else 10
    return Card("Boot Sequence+" if upgraded else "Boot Sequence", 0, CardType.SKILL,
                lambda combat, target: _boot_sequence(combat, target, block),
                targeted=False, exhausts=True, upgraded=upgraded)


def _chill(combat, target):
    count = len(combat.living_monsters)
    for _ in range(count):
        combat.channel_orb(make_frost_orb())
        combat.player.orb_channeled_count += 1
        combat.player.frost_channeled += 1


def make_chill(upgraded: bool = False) -> Card:
    return Card("Chill+" if upgraded else "Chill", 0, CardType.SKILL,
                _chill, targeted=False, exhausts=True, upgraded=upgraded)


# --- Defect: Rare cards ---

def _loop_card(combat, target, amount):
    combat.player.add_power(Loop(amount))


def make_loop(upgraded: bool = False) -> Card:
    amount = 2 if upgraded else 1
    return Card("Loop+" if upgraded else "Loop", 1, CardType.POWER,
                lambda combat, target: _loop_card(combat, target, amount),
                targeted=False, upgraded=upgraded)


def _self_repair(combat, target, amount):
    combat.player.add_power(SelfRepairPower(amount))


def make_self_repair(upgraded: bool = False) -> Card:
    heal = 10 if upgraded else 7
    return Card("Self Repair+" if upgraded else "Self Repair", 1, CardType.POWER,
                lambda combat, target: _self_repair(combat, target, heal),
                targeted=False, upgraded=upgraded)


def _melter(combat, target, dmg):
    target.block = 0
    combat.deal_attack_damage(combat.player, target, dmg)


def make_melter(upgraded: bool = False) -> Card:
    dmg = 14 if upgraded else 10
    return Card("Melter+" if upgraded else "Melter", 1, CardType.ATTACK,
                lambda combat, target: _melter(combat, target, dmg),
                targeted=True, upgraded=upgraded)


def _chaos(combat, target, count):
    orb_types = [make_lightning_orb, make_frost_orb, make_dark_orb, make_plasma_orb]
    for _ in range(count):
        factory = combat.rng.choice(orb_types)
        orb = factory()
        combat.channel_orb(orb)
        combat.player.orb_channeled_count += 1
        if orb.name == "Frost":
            combat.player.frost_channeled += 1
        elif orb.name == "Lightning":
            combat.player.lightning_channeled += 1


def make_chaos(upgraded: bool = False) -> Card:
    count = 2 if upgraded else 1
    return Card("Chaos+" if upgraded else "Chaos", 1, CardType.SKILL,
                lambda combat, target: _chaos(combat, target, count),
                targeted=False, upgraded=upgraded)


def _blizzard(combat, target, per_frost):
    hits = combat.player.frost_channeled
    for _ in range(hits):
        for m in combat.living_monsters:
            combat.deal_attack_damage(combat.player, m, per_frost)


def make_blizzard(upgraded: bool = False) -> Card:
    per_frost = 3 if upgraded else 2
    return Card("Blizzard+" if upgraded else "Blizzard", 1, CardType.ATTACK,
                lambda combat, target: _blizzard(combat, target, per_frost),
                targeted=False, upgraded=upgraded)


def _aggregate(combat, target, divisor):
    gained = len(combat.draw_pile) // divisor
    combat.player.energy += gained


def make_aggregate(upgraded: bool = False) -> Card:
    divisor = 3 if upgraded else 4
    return Card("Aggregate+" if upgraded else "Aggregate", 1, CardType.SKILL,
                lambda combat, target: _aggregate(combat, target, divisor),
                targeted=False, upgraded=upgraded)


def _fusion(combat, target):
    combat.channel_orb(make_plasma_orb())
    combat.player.orb_channeled_count += 1


def make_fusion(upgraded: bool = False) -> Card:
    return Card("Fusion+" if upgraded else "Fusion", 1 if upgraded else 2, CardType.SKILL,
                _fusion, targeted=False, upgraded=upgraded)


def _consume(combat, target, focus_gain):
    combat.player.orb_slots = max(0, combat.player.orb_slots - 1)
    while len(combat.player.orbs) > combat.player.orb_slots:
        evoked = combat.player.orbs.pop(0)
        evoked.evoke(combat.player, combat)
    combat.player.add_power(Focus(focus_gain))


def make_consume(upgraded: bool = False) -> Card:
    focus_gain = 3 if upgraded else 2
    return Card("Consume+" if upgraded else "Consume", 2, CardType.SKILL,
                lambda combat, target: _consume(combat, target, focus_gain),
                targeted=False, upgraded=upgraded)


def _glacier(combat, target, block):
    combat.player.gain_block(block)
    for _ in range(2):
        combat.channel_orb(make_frost_orb())
        combat.player.orb_channeled_count += 1
        combat.player.frost_channeled += 1


def make_glacier(upgraded: bool = False) -> Card:
    block = 10 if upgraded else 7
    return Card("Glacier+" if upgraded else "Glacier", 2, CardType.SKILL,
                lambda combat, target: _glacier(combat, target, block),
                targeted=False, upgraded=upgraded)


def _sunder(combat, target, dmg):
    combat.deal_attack_damage(combat.player, target, dmg)
    if target.is_dead:
        combat.player.energy += 3


def make_sunder(upgraded: bool = False) -> Card:
    dmg = 32 if upgraded else 24
    return Card("Sunder+" if upgraded else "Sunder", 2, CardType.ATTACK,
                lambda combat, target: _sunder(combat, target, dmg),
                targeted=True, upgraded=upgraded)


def _overclock(combat, target, draw):
    combat.draw_cards(draw)
    combat.discard_pile.append(make_burn())


def make_overclock(upgraded: bool = False) -> Card:
    draw = 3 if upgraded else 2
    return Card("Overclock+" if upgraded else "Overclock", 0, CardType.SKILL,
                lambda combat, target: _overclock(combat, target, draw),
                targeted=False, upgraded=upgraded)


def make_genetic_algorithm(upgraded: bool = False) -> Card:
    increment = 3 if upgraded else 2
    state = {"block": 1}

    def _effect(combat, target):
        combat.player.gain_block(state["block"])
        state["block"] += increment

    return Card("Genetic Algorithm+" if upgraded else "Genetic Algorithm", 1, CardType.SKILL,
                _effect, targeted=False, exhausts=True, upgraded=upgraded)


def _multi_cast(combat, target, extra):
    x = combat._x_value
    if not combat.player.orbs:
        return
    for _ in range(x + extra):
        if not combat.player.orbs:
            break
        orb = combat.player.orbs[0]
        orb.evoke(combat.player, combat)
        combat.player.orbs.pop(0)


def make_multi_cast(upgraded: bool = False) -> Card:
    extra = 1 if upgraded else 0
    return Card("Multi-Cast+" if upgraded else "Multi-Cast", 0, CardType.SKILL,
                lambda combat, target: _multi_cast(combat, target, extra),
                targeted=False, is_x_cost=True, upgraded=upgraded)


def _hyperbeam(combat, target, dmg):
    for m in combat.living_monsters:
        combat.deal_attack_damage(combat.player, m, dmg)
    focus = combat.player.powers.get("Focus")
    if focus is not None:
        focus.amount = max(-999, focus.amount - 3)
    else:
        combat.player.add_power(Focus(-3))


def make_hyperbeam(upgraded: bool = False) -> Card:
    dmg = 34 if upgraded else 26
    return Card("Hyperbeam+" if upgraded else "Hyperbeam", 2, CardType.ATTACK,
                lambda combat, target: _hyperbeam(combat, target, dmg),
                targeted=False, upgraded=upgraded)


def _thunder_strike(combat, target, per_lightning):
    hits = combat.player.lightning_channeled
    for _ in range(hits):
        living = combat.living_monsters
        if not living:
            break
        t = combat.rng.choice(living)
        combat.deal_attack_damage(combat.player, t, per_lightning)


def make_thunder_strike(upgraded: bool = False) -> Card:
    per_lightning = 9 if upgraded else 7
    return Card("Thunder Strike+" if upgraded else "Thunder Strike", 3, CardType.ATTACK,
                lambda combat, target: _thunder_strike(combat, target, per_lightning),
                targeted=False, upgraded=upgraded)


def _biased_cognition_card(combat, target, amount):
    combat.player.add_power(BiasedCognition(amount))


def make_biased_cognition(upgraded: bool = False) -> Card:
    amount = 5 if upgraded else 4
    return Card("Biased Cognition+" if upgraded else "Biased Cognition", 1, CardType.POWER,
                lambda combat, target: _biased_cognition_card(combat, target, amount),
                targeted=False, upgraded=upgraded)


def _machine_learning_card(combat, target, amount):
    combat.player.add_power(MachineLearning(amount))


def make_machine_learning(upgraded: bool = False) -> Card:
    return Card("Machine Learning+" if upgraded else "Machine Learning", 1, CardType.POWER,
                lambda combat, target: _machine_learning_card(combat, target, 1),
                targeted=False, upgraded=upgraded)


def _electrodynamics(combat, target, lightning_count):
    for _ in range(lightning_count):
        combat.channel_orb(make_lightning_orb())
        combat.player.orb_channeled_count += 1
        combat.player.lightning_channeled += 1


def make_electrodynamics(upgraded: bool = False) -> Card:
    lightning_count = 3 if upgraded else 2
    return Card("Electrodynamics+" if upgraded else "Electrodynamics", 2, CardType.POWER,
                lambda combat, target: _electrodynamics(combat, target, lightning_count),
                targeted=False, upgraded=upgraded)


def _rainbow(combat, target):
    for factory in [make_lightning_orb, make_frost_orb, make_dark_orb, make_plasma_orb]:
        orb = factory()
        combat.channel_orb(orb)
        combat.player.orb_channeled_count += 1
        if orb.name == "Frost":
            combat.player.frost_channeled += 1
        elif orb.name == "Lightning":
            combat.player.lightning_channeled += 1


def make_rainbow(upgraded: bool = False) -> Card:
    return Card("Rainbow+" if upgraded else "Rainbow", 2, CardType.SKILL,
                _rainbow, targeted=False, exhausts=True, upgraded=upgraded)


def make_seek(upgraded: bool = False) -> Card:
    count = 2 if upgraded else 1

    def _effect(combat, target):
        for _ in range(count):
            if not combat.draw_pile:
                break
            card = combat.rng.choice(combat.draw_pile)
            combat.draw_pile.remove(card)
            combat.add_card_to_hand(card)

    return Card("Seek+" if upgraded else "Seek", 0, CardType.SKILL,
                _effect, targeted=False, exhausts=True, upgraded=upgraded)


def _creative_ai_card(combat, target):
    combat.player.add_power(CreativeAI())


def make_creative_ai(upgraded: bool = False) -> Card:
    return Card("Creative AI+" if upgraded else "Creative AI", 2 if upgraded else 3, CardType.POWER,
                _creative_ai_card, targeted=False, upgraded=upgraded)


def _all_for_one(combat, target, dmg):
    combat.deal_attack_damage(combat.player, target, dmg)
    for card in list(combat.discard_pile):
        if card.cost == 0:
            combat.discard_pile.remove(card)
            combat.add_card_to_hand(card)


def make_all_for_one(upgraded: bool = False) -> Card:
    dmg = 14 if upgraded else 10
    return Card("All For One+" if upgraded else "All For One", 2, CardType.ATTACK,
                lambda combat, target: _all_for_one(combat, target, dmg),
                targeted=True, upgraded=upgraded)


def _reboot(combat, target, draw_count):
    combat.draw_pile.extend(combat.discard_pile)
    combat.discard_pile = []
    combat.draw_pile.extend(combat.hand)
    combat.hand.clear()
    combat.rng.shuffle(combat.draw_pile)
    combat.draw_cards(draw_count)


def make_reboot(upgraded: bool = False) -> Card:
    draw_count = 6 if upgraded else 4
    return Card("Reboot+" if upgraded else "Reboot", 0, CardType.SKILL,
                lambda combat, target: _reboot(combat, target, draw_count),
                targeted=False, exhausts=True, upgraded=upgraded)


def _meteor_strike(combat, target, dmg):
    combat.deal_attack_damage(combat.player, target, dmg)
    for _ in range(3):
        combat.channel_orb(make_plasma_orb())
        combat.player.orb_channeled_count += 1


def make_meteor_strike(upgraded: bool = False) -> Card:
    dmg = 30 if upgraded else 24
    return Card("Meteor Strike+" if upgraded else "Meteor Strike", 5, CardType.ATTACK,
                lambda combat, target: _meteor_strike(combat, target, dmg),
                targeted=True, upgraded=upgraded)


def _fission(combat, target, upgrade):
    orb_count = len(combat.player.orbs)
    if upgrade:
        for _ in range(orb_count):
            if combat.player.orbs:
                orb = combat.player.orbs.pop(0)
                orb.evoke(combat.player, combat)
    combat.player.energy += orb_count
    combat.draw_cards(orb_count)
    combat.player.orbs.clear()


def make_fission(upgraded: bool = False) -> Card:
    return Card("Fission+" if upgraded else "Fission", 0, CardType.SKILL,
                lambda combat, target: _fission(combat, target, upgraded),
                targeted=False, exhausts=True, upgraded=upgraded)


def _core_surge(combat, target, dmg):
    combat.deal_attack_damage(combat.player, target, dmg)
    combat.player.add_power(Artifact(1))


def make_core_surge(upgraded: bool = False) -> Card:
    dmg = 15 if upgraded else 11
    return Card("Core Surge+" if upgraded else "Core Surge", 1, CardType.ATTACK,
                lambda combat, target: _core_surge(combat, target, dmg),
                targeted=True, exhausts=True, upgraded=upgraded)


def _static_discharge_card(combat, target, amount):
    combat.player.add_power(StaticDischarge(amount))


def make_static_discharge(upgraded: bool = False) -> Card:
    amount = 2 if upgraded else 1
    return Card("Static Discharge+" if upgraded else "Static Discharge", 1, CardType.POWER,
                lambda combat, target: _static_discharge_card(combat, target, amount),
                targeted=False, upgraded=upgraded)


def _storm_card(combat, target, amount):
    combat.player.add_power(Storm(amount))


def make_storm(upgraded: bool = False) -> Card:
    amount = 2 if upgraded else 1
    return Card("Storm+" if upgraded else "Storm", 1, CardType.POWER,
                lambda combat, target: _storm_card(combat, target, amount),
                targeted=False, upgraded=upgraded)


def _heatsinks_card(combat, target, amount):
    combat.player.add_power(Heatsinks(amount))


def make_heatsinks(upgraded: bool = False) -> Card:
    amount = 2 if upgraded else 1
    return Card("Heatsinks+" if upgraded else "Heatsinks", 1, CardType.POWER,
                lambda combat, target: _heatsinks_card(combat, target, amount),
                targeted=False, upgraded=upgraded)


def _hello_world_card(combat, target):
    combat.player.add_power(HelloWorld())


def make_hello_world(upgraded: bool = False) -> Card:
    return Card("Hello World+" if upgraded else "Hello World", 1, CardType.POWER,
                _hello_world_card, targeted=False, upgraded=upgraded)


def _buffer_card(combat, target, amount):
    combat.player.add_power(Buffer(amount))


def make_buffer(upgraded: bool = False) -> Card:
    amount = 2 if upgraded else 1
    return Card("Buffer+" if upgraded else "Buffer", 2, CardType.POWER,
                lambda combat, target: _buffer_card(combat, target, amount),
                targeted=False, upgraded=upgraded)


def _amplify_card(combat, target):
    combat.player.add_power(Amplify(1))


def make_amplify(upgraded: bool = False) -> Card:
    return Card("Amplify+" if upgraded else "Amplify", 1, CardType.SKILL,
                _amplify_card, targeted=False, upgraded=upgraded)


def _echo_form_card(combat, target):
    combat.player.add_power(EchoForm())


def make_echo_form(upgraded: bool = False) -> Card:
    return Card("Echo Form+" if upgraded else "Echo Form", 2 if upgraded else 3, CardType.POWER,
                _echo_form_card, targeted=False, upgraded=upgraded)


# --- Updated defect card pool ---

def defect_card_pool() -> list[Card]:
    return [
        make_ball_lightning(), make_cold_snap(), make_coolheaded(), make_focus_card(),
        make_go_for_the_eyes(), make_streamline(), make_recursion(),
        make_compile_driver(), make_barrage(), make_stack(), make_rebound(),
        make_claw(), make_turbo(), make_sweeping_beam(), make_charge_battery(),
        make_hologram(), make_beam_cell(), make_leap(), make_steam_barrier(),
        make_doom_and_gloom(), make_defragment(), make_capacitor(),
        make_white_noise(), make_skim(), make_recycle(), make_scrape(),
        make_bullseye(), make_reprogram(), make_auto_shields(),
        make_reinforced_body(), make_double_energy(), make_darkness(),
        make_rip_and_tear(), make_ftl(), make_force_field(), make_equilibrium(),
        make_tempest(), make_boot_sequence(), make_chill(),
        make_loop(), make_self_repair(), make_melter(), make_chaos(),
        make_blizzard(), make_aggregate(), make_fusion(), make_consume(),
        make_glacier(), make_sunder(), make_overclock(), make_genetic_algorithm(),
        make_multi_cast(), make_hyperbeam(), make_thunder_strike(),
        make_biased_cognition(), make_machine_learning(), make_electrodynamics(),
        make_rainbow(), make_seek(), make_creative_ai(), make_all_for_one(),
        make_reboot(), make_meteor_strike(), make_fission(), make_core_surge(),
        make_static_discharge(), make_storm(), make_heatsinks(),
        make_hello_world(), make_buffer(), make_amplify(), make_echo_form(),
    ]
# (Watcher's third stance and the resource that unlocks it) -- see
# Creature.stance in creatures.py for how Wrath's damage doubling is applied.

def _set_stance(combat, new_stance):
    player = combat.player
    old_stance = player.stance
    if player.stance == "Calm" and new_stance != "Calm":
        player.energy += 2  # Exit Calm: the real payoff for entering it
    player.stance = new_stance
    # Fire on_stance_change for powers that care (Rushdown, Mental Fortress)
    for power in list(player.powers.values()):
        power.on_stance_change(player, combat, new_stance, old_stance)
    # Flurry of Blows: on stance change, return from discard pile to hand
    for card in list(combat.discard_pile):
        if "Flurry of Blows" in card.name:
            combat.discard_pile.remove(card)
            combat.add_card_to_hand(card)


def _eruption(combat, target):
    combat.deal_attack_damage(combat.player, target, 9)
    _set_stance(combat, "Wrath")


def _vigilance(combat, target):
    combat.player.gain_block(8)
    _set_stance(combat, "Calm")


def _crescendo(combat, target):
    _set_stance(combat, "Wrath")


def _tranquility(combat, target):
    _set_stance(combat, "Calm")


def _empty_body(combat, target):
    combat.player.gain_block(7)
    _set_stance(combat, None)


def _empty_fist(combat, target):
    combat.deal_attack_damage(combat.player, target, 9)
    _set_stance(combat, None)


def make_eruption(upgraded: bool = False) -> Card:
    """Upgrading drops its cost 2->1 rather than changing its effect."""
    return Card("Eruption+" if upgraded else "Eruption", 1 if upgraded else 2,
                CardType.ATTACK, _eruption, targeted=True, upgraded=upgraded)


def make_vigilance(upgraded: bool = False) -> Card:
    """Upgrading drops its cost 2->1 rather than changing its effect."""
    return Card("Vigilance+" if upgraded else "Vigilance", 1 if upgraded else 2,
                CardType.SKILL, _vigilance, targeted=False, upgraded=upgraded)


def make_crescendo() -> Card:
    return Card("Crescendo", 1, CardType.SKILL, _crescendo, targeted=False, exhausts=True)


def make_tranquility() -> Card:
    return Card("Tranquility", 1, CardType.SKILL, _tranquility, targeted=False, exhausts=True)


def make_empty_body() -> Card:
    return Card("Empty Body", 1, CardType.SKILL, _empty_body, targeted=False)


def make_empty_fist() -> Card:
    return Card("Empty Fist", 1, CardType.ATTACK, _empty_fist, targeted=True)


def watcher_starter_deck() -> list[Card]:
    """The Watcher's real starting deck: 4 Strike, 4 Defend, Eruption, Vigilance."""
    deck = [make_strike() for _ in range(4)]
    deck += [make_defend() for _ in range(4)]
    deck += [make_eruption(), make_vigilance()]
    return deck


def watcher_card_pool() -> list[Card]:
    return [
        make_crescendo(), make_tranquility(), make_empty_body(), make_empty_fist(),
        make_consecrate(), make_bowling_bash(), make_flying_sleeves(),
        make_halt(), make_just_lucky(), make_flurry_of_blows(),
        make_protect(), make_third_eye(), make_sash_whip(),
        make_cut_through_fate(), make_follow_up(), make_pressure_points(),
        make_crush_joints(), make_evaluate(), make_prostrate(),
        make_pray(), make_signature_move(), make_weave(),
        make_empty_mind_card(), make_nirvana(), make_tantrum(),
        make_conclude(), make_worship(), make_swivel(),
        make_perseverance(), make_meditate(), make_study(),
        make_wave_of_the_hand(), make_sands_of_time(), make_fear_no_evil(),
        make_reach_heaven(), make_deceive_reality(), make_inner_peace(),
        make_collect(), make_wreath_of_flame(), make_wallop(),
        make_carve_reality(), make_like_water(), make_fasting_card(),
        make_foreign_influence(), make_windmill_strike(),
        make_indignation(), make_talk_to_the_hand(), make_sanctity(),
        make_foresight(), make_simmering_fury(), make_wheel_kick(),
        make_judgment(), make_conjure_blade(), make_master_reality(),
        make_brilliance(), make_devotion_card(), make_blasphemy(),
        make_ragnarok(), make_lesson_learned(), make_scrawl(),
        make_vault(), make_alpha(), make_wish(),
        make_omniscience(), make_establishment(), make_spirit_shield(),
        make_deva_form(), make_deus_ex_machina(),
    ]


# --- Watcher card effect functions ---

class _StudyPower:
    name = "Study"
    is_duration = False
    def __init__(self):
        self.amount = 1
    def on_end_turn(self, owner, combat):
        combat.draw_pile.append(make_insight())
        combat.rng.shuffle(combat.draw_pile)


class _DevaFormPower:
    name = "Deva Form"
    is_duration = False
    def __init__(self):
        self.amount = 1
    def on_start_turn(self, owner, combat):
        owner.energy += self.amount
        self.amount += 1


class _ForesightPower:
    name = "Foresight"
    is_duration = False
    def __init__(self, amount):
        self.amount = amount
    def on_start_turn(self, owner, combat):
        combat.scry(self.amount)


class _MasterRealityPower:
    name = "Master Reality"
    is_duration = False
    def __init__(self):
        self.amount = 1


class _EstablishmentPower:
    name = "Establishment"
    is_duration = False
    def __init__(self):
        self.amount = 1


class _OmegaPower:
    name = "Omega"
    is_duration = False
    def __init__(self, amount):
        self.amount = amount
    def on_end_turn(self, owner, combat):
        for m in combat.living_monsters:
            m.take_damage(self.amount)


class _NirvanaPower:
    name = "Nirvana"
    is_duration = False
    def __init__(self, amount):
        self.amount = amount


def make_smite(upgraded: bool = False) -> Card:
    dmg = 16 if upgraded else 12
    return Card("Smite+" if upgraded else "Smite", 1, CardType.ATTACK,
                lambda combat, target: combat.deal_attack_damage(combat.player, target, dmg),
                targeted=True, exhausts=True, upgraded=upgraded)


def make_insight(upgraded: bool = False) -> Card:
    draw = 3 if upgraded else 2
    return Card("Insight+" if upgraded else "Insight", 0, CardType.SKILL,
                lambda combat, target: combat.draw_cards(draw),
                targeted=False, exhausts=True, upgraded=upgraded)


def make_safety(upgraded: bool = False) -> Card:
    block = 16 if upgraded else 12
    return Card("Safety+" if upgraded else "Safety", 1, CardType.SKILL,
                lambda combat, target: combat.player.gain_block(block),
                targeted=False, exhausts=True, upgraded=upgraded)


def make_through_violence(upgraded: bool = False) -> Card:
    dmg = 30 if upgraded else 20
    return Card("Through Violence+" if upgraded else "Through Violence", 0, CardType.ATTACK,
                lambda combat, target: combat.deal_attack_damage(combat.player, target, dmg),
                targeted=True, exhausts=True, upgraded=upgraded)


def _beta_effect(combat, upgraded):
    combat.draw_pile.append(make_omega(upgraded))
    combat.rng.shuffle(combat.draw_pile)


def make_beta(upgraded: bool = False) -> Card:
    return Card("Beta+" if upgraded else "Beta", 2, CardType.SKILL,
                lambda combat, target: _beta_effect(combat, upgraded),
                targeted=False, exhausts=True, upgraded=upgraded)


def make_omega(upgraded: bool = False) -> Card:
    dmg = 60 if upgraded else 50
    return Card("Omega+" if upgraded else "Omega", 3, CardType.POWER,
                lambda combat, target: combat.player.add_power(_OmegaPower(dmg)),
                targeted=False, upgraded=upgraded)


def make_expunger(x: int, upgraded: bool = False) -> Card:
    dmg_per = 15 if upgraded else 9
    total = x * dmg_per
    return Card("Expunger", x, CardType.ATTACK,
                lambda combat, target: combat.deal_attack_damage(combat.player, target, total),
                targeted=True)


def _consecrate(combat, dmg):
    for m in combat.living_monsters:
        combat.deal_attack_damage(combat.player, m, dmg)


def make_consecrate(upgraded: bool = False) -> Card:
    dmg = 8 if upgraded else 5
    return Card("Consecrate+" if upgraded else "Consecrate", 0, CardType.ATTACK,
                lambda combat, target: _consecrate(combat, dmg),
                targeted=False, upgraded=upgraded)


def _bowling_bash(combat, target, per):
    count = len(combat.living_monsters)
    for _ in range(count):
        if target.is_dead:
            break
        combat.deal_attack_damage(combat.player, target, per)


def make_bowling_bash(upgraded: bool = False) -> Card:
    per = 10 if upgraded else 7
    return Card("Bowling Bash+" if upgraded else "Bowling Bash", 1, CardType.ATTACK,
                lambda combat, target: _bowling_bash(combat, target, per),
                targeted=True, upgraded=upgraded)


def _flying_sleeves(combat, target, dmg):
    combat.deal_attack_damage(combat.player, target, dmg)
    combat.deal_attack_damage(combat.player, target, dmg)


def make_flying_sleeves(upgraded: bool = False) -> Card:
    dmg = 6 if upgraded else 4
    return Card("Flying Sleeves+" if upgraded else "Flying Sleeves", 1, CardType.ATTACK,
                lambda combat, target: _flying_sleeves(combat, target, dmg),
                targeted=True, upgraded=upgraded)


def _halt(combat, block, bonus):
    combat.player.gain_block(block)
    if combat.player.stance == "Wrath":
        combat.player.gain_block(bonus)


def make_halt(upgraded: bool = False) -> Card:
    block = 4 if upgraded else 3
    bonus = 14 if upgraded else 9
    return Card("Halt+" if upgraded else "Halt", 0, CardType.SKILL,
                lambda combat, target: _halt(combat, block, bonus),
                targeted=False, upgraded=upgraded)


def _just_lucky(combat, target, dmg, scry, block):
    combat.scry(scry)
    combat.deal_attack_damage(combat.player, target, dmg)
    combat.player.gain_block(block)


def make_just_lucky(upgraded: bool = False) -> Card:
    dmg = 4 if upgraded else 3
    scry = 2 if upgraded else 1
    block = 3 if upgraded else 2
    return Card("Just Lucky+" if upgraded else "Just Lucky", 0, CardType.ATTACK,
                lambda combat, target: _just_lucky(combat, target, dmg, scry, block),
                targeted=True, upgraded=upgraded)


def make_flurry_of_blows(upgraded: bool = False) -> Card:
    dmg = 6 if upgraded else 4
    return Card("Flurry of Blows+" if upgraded else "Flurry of Blows", 0, CardType.ATTACK,
                lambda combat, target: combat.deal_attack_damage(combat.player, target, dmg),
                targeted=True, upgraded=upgraded)


def make_protect(upgraded: bool = False) -> Card:
    block = 16 if upgraded else 12
    return Card("Protect+" if upgraded else "Protect", 2, CardType.SKILL,
                lambda combat, target: combat.player.gain_block(block),
                targeted=False, upgraded=upgraded)


def _third_eye(combat, block, scry):
    combat.player.gain_block(block)
    combat.scry(scry)


def make_third_eye(upgraded: bool = False) -> Card:
    block = 9 if upgraded else 7
    scry = 5 if upgraded else 3
    return Card("Third Eye+" if upgraded else "Third Eye", 1, CardType.SKILL,
                lambda combat, target: _third_eye(combat, block, scry),
                targeted=False, upgraded=upgraded)


def _sash_whip(combat, target, dmg, weak):
    combat.deal_attack_damage(combat.player, target, dmg)
    target.add_power(Weak(weak))
    target.add_power(Weak(1))


def make_sash_whip(upgraded: bool = False) -> Card:
    dmg = 10 if upgraded else 8
    weak = 2 if upgraded else 1
    return Card("Sash Whip+" if upgraded else "Sash Whip", 1, CardType.ATTACK,
                lambda combat, target: _sash_whip(combat, target, dmg, weak),
                targeted=True, upgraded=upgraded)


def _cut_through_fate(combat, target, dmg, scry):
    combat.scry(scry)
    combat.draw_cards(1)
    combat.deal_attack_damage(combat.player, target, dmg)


def make_cut_through_fate(upgraded: bool = False) -> Card:
    dmg = 9 if upgraded else 7
    scry = 3 if upgraded else 2
    return Card("Cut Through Fate+" if upgraded else "Cut Through Fate", 1, CardType.ATTACK,
                lambda combat, target: _cut_through_fate(combat, target, dmg, scry),
                targeted=True, upgraded=upgraded)


def _follow_up(combat, target, dmg):
    combat.deal_attack_damage(combat.player, target, dmg)
    combat.player.energy += 1


def make_follow_up(upgraded: bool = False) -> Card:
    dmg = 11 if upgraded else 7
    return Card("Follow Up+" if upgraded else "Follow Up", 1, CardType.ATTACK,
                lambda combat, target: _follow_up(combat, target, dmg),
                targeted=True, upgraded=upgraded)


def _pressure_points(combat, target, mark):
    target.add_power(Mark(mark))
    target.take_damage(target.get_power_amount("Mark"))


def make_pressure_points(upgraded: bool = False) -> Card:
    mark = 11 if upgraded else 8
    return Card("Pressure Points+" if upgraded else "Pressure Points", 1, CardType.SKILL,
                lambda combat, target: _pressure_points(combat, target, mark),
                targeted=True, upgraded=upgraded)


def _crush_joints(combat, target, dmg, vuln):
    combat.deal_attack_damage(combat.player, target, dmg)
    target.add_power(Vulnerable(vuln))
    target.add_power(Vulnerable(1))


def make_crush_joints(upgraded: bool = False) -> Card:
    dmg = 10 if upgraded else 8
    vuln = 2 if upgraded else 1
    return Card("Crush Joints+" if upgraded else "Crush Joints", 1, CardType.ATTACK,
                lambda combat, target: _crush_joints(combat, target, dmg, vuln),
                targeted=True, upgraded=upgraded)


def _evaluate(combat, block):
    combat.player.gain_block(block)
    combat.draw_pile.append(make_insight())
    combat.rng.shuffle(combat.draw_pile)


def make_evaluate(upgraded: bool = False) -> Card:
    block = 10 if upgraded else 6
    return Card("Evaluate+" if upgraded else "Evaluate", 1, CardType.SKILL,
                lambda combat, target: _evaluate(combat, block),
                targeted=False, upgraded=upgraded)


def _prostrate(combat, mantra_gain, block):
    combat.player._mantra = getattr(combat.player, '_mantra', 0) + mantra_gain
    if combat.player._mantra >= 10:
        combat.player._mantra = 0
        combat.player.energy += 3
    combat.player.gain_block(block)


def make_prostrate(upgraded: bool = False) -> Card:
    mantra_gain = 3 if upgraded else 2
    block = 4
    return Card("Prostrate+" if upgraded else "Prostrate", 0, CardType.SKILL,
                lambda combat, target: _prostrate(combat, mantra_gain, block),
                targeted=False, upgraded=upgraded)


def _pray(combat, mantra_gain):
    combat.player._mantra = getattr(combat.player, '_mantra', 0) + mantra_gain
    if combat.player._mantra >= 10:
        combat.player._mantra = 0
        combat.player.energy += 3
    combat.draw_pile.append(make_insight())
    combat.rng.shuffle(combat.draw_pile)


def make_pray(upgraded: bool = False) -> Card:
    mantra_gain = 4 if upgraded else 3
    return Card("Pray+" if upgraded else "Pray", 1, CardType.SKILL,
                lambda combat, target: _pray(combat, mantra_gain),
                targeted=False, upgraded=upgraded)


def _sig_move_legal(combat) -> bool:
    return sum(1 for c in combat.hand if c.card_type == CardType.ATTACK) == 1


def make_signature_move(upgraded: bool = False) -> Card:
    dmg = 40 if upgraded else 30
    return Card("Signature Move+" if upgraded else "Signature Move", 2, CardType.ATTACK,
                lambda combat, target: combat.deal_attack_damage(combat.player, target, dmg),
                targeted=True, extra_legal_check=_sig_move_legal, upgraded=upgraded)


def make_weave(upgraded: bool = False) -> Card:
    dmg = 6 if upgraded else 4
    return Card("Weave+" if upgraded else "Weave", 0, CardType.ATTACK,
                lambda combat, target: combat.deal_attack_damage(combat.player, target, dmg),
                targeted=True, upgraded=upgraded)


def _empty_mind(combat, draw):
    _set_stance(combat, None)
    combat.draw_cards(draw)


def make_empty_mind_card(upgraded: bool = False) -> Card:
    draw = 3 if upgraded else 2
    return Card("Empty Mind+" if upgraded else "Empty Mind", 1, CardType.SKILL,
                lambda combat, target: _empty_mind(combat, draw),
                targeted=False, upgraded=upgraded)


def make_nirvana(upgraded: bool = False) -> Card:
    block = 4 if upgraded else 3
    return Card("Nirvana+" if upgraded else "Nirvana", 1, CardType.POWER,
                lambda combat, target: combat.player.add_power(_NirvanaPower(block)),
                targeted=False, upgraded=upgraded)


def _tantrum(combat, target, dmg):
    for _ in range(3):
        combat.deal_attack_damage(combat.player, target, dmg)
    _set_stance(combat, "Wrath")
    combat.draw_pile.append(make_tantrum())
    combat.rng.shuffle(combat.draw_pile)


def make_tantrum(upgraded: bool = False) -> Card:
    dmg = 4 if upgraded else 3
    return Card("Tantrum+" if upgraded else "Tantrum", 1, CardType.ATTACK,
                lambda combat, target: _tantrum(combat, target, dmg),
                targeted=True, upgraded=upgraded)


def _conclude(combat, dmg):
    for m in combat.living_monsters:
        combat.deal_attack_damage(combat.player, m, dmg)
    combat.turn_should_end_early = True


def make_conclude(upgraded: bool = False) -> Card:
    dmg = 16 if upgraded else 12
    return Card("Conclude+" if upgraded else "Conclude", 1, CardType.ATTACK,
                lambda combat, target: _conclude(combat, dmg),
                targeted=False, upgraded=upgraded)


def _worship(combat):
    combat.player._mantra = getattr(combat.player, '_mantra', 0) + 5
    if combat.player._mantra >= 10:
        combat.player._mantra = 0
        combat.player.energy += 3


def make_worship(upgraded: bool = False) -> Card:
    return Card("Worship+" if upgraded else "Worship", 2, CardType.SKILL,
                lambda combat, target: _worship(combat),
                targeted=False, upgraded=upgraded)


def _swivel(combat, block):
    combat.player.gain_block(block)
    attacks = [c for c in combat.hand if c.card_type == CardType.ATTACK]
    if attacks:
        combat.rng.choice(attacks).cost = 0


def make_swivel(upgraded: bool = False) -> Card:
    block = 11 if upgraded else 8
    return Card("Swivel+" if upgraded else "Swivel", 2, CardType.SKILL,
                lambda combat, target: _swivel(combat, block),
                targeted=False, upgraded=upgraded)


def make_perseverance(upgraded: bool = False) -> Card:
    state = {"base": 7 if upgraded else 5, "bonus": 0}
    increment = 3 if upgraded else 2

    def _effect(combat, target):
        combat.player.gain_block(state["base"] + state["bonus"])

    def _retain(card, combat):
        state["bonus"] += increment

    return Card("Perseverance+" if upgraded else "Perseverance", 1, CardType.SKILL,
                _effect, targeted=False, retain=True, retain_callback=_retain, upgraded=upgraded)


def _meditate(combat, retain_count):
    _set_stance(combat, "Calm")
    combat.turn_should_end_early = True


def make_meditate(upgraded: bool = False) -> Card:
    return Card("Meditate+" if upgraded else "Meditate", 1, CardType.SKILL,
                lambda combat, target: _meditate(combat, 3 if upgraded else 2),
                targeted=False, upgraded=upgraded)


def make_study(upgraded: bool = False) -> Card:
    return Card("Study+" if upgraded else "Study", 1 if upgraded else 2, CardType.POWER,
                lambda combat, target: combat.player.add_power(_StudyPower()),
                targeted=False, upgraded=upgraded)


def _wave_of_the_hand(combat, weak):
    for m in combat.living_monsters:
        m.add_power(Weak(weak))


def make_wave_of_the_hand(upgraded: bool = False) -> Card:
    weak = 2 if upgraded else 1
    return Card("Wave of the Hand+" if upgraded else "Wave of the Hand", 1, CardType.SKILL,
                lambda combat, target: _wave_of_the_hand(combat, weak),
                targeted=False, upgraded=upgraded)


def make_sands_of_time(upgraded: bool = False) -> Card:
    dmg = 26 if upgraded else 20

    def _effect(combat, target):
        combat.deal_attack_damage(combat.player, target, dmg)

    def _retain(card, combat):
        card.cost = max(0, card.cost - 1)

    return Card("Sands of Time+" if upgraded else "Sands of Time", 4, CardType.ATTACK,
                _effect, targeted=True, retain=True, retain_callback=_retain, upgraded=upgraded)


def _fear_no_evil(combat, target, dmg):
    from .enemies import IntentType
    combat.deal_attack_damage(combat.player, target, dmg)
    if target.intent and target.intent.type in (IntentType.ATTACK, IntentType.ATTACK_DEFEND):
        _set_stance(combat, "Calm")


def make_fear_no_evil(upgraded: bool = False) -> Card:
    dmg = 11 if upgraded else 8
    return Card("Fear No Evil+" if upgraded else "Fear No Evil", 1, CardType.ATTACK,
                lambda combat, target: _fear_no_evil(combat, target, dmg),
                targeted=True, upgraded=upgraded)


def _reach_heaven(combat, target, dmg, upgraded):
    combat.deal_attack_damage(combat.player, target, dmg)
    combat.draw_pile.append(make_through_violence(upgraded))
    combat.rng.shuffle(combat.draw_pile)


def make_reach_heaven(upgraded: bool = False) -> Card:
    dmg = 15 if upgraded else 10
    return Card("Reach Heaven+" if upgraded else "Reach Heaven", 2, CardType.ATTACK,
                lambda combat, target: _reach_heaven(combat, target, dmg, upgraded),
                targeted=True, upgraded=upgraded)


def _deceive_reality(combat, block, upgraded):
    combat.player.gain_block(block)
    combat.add_card_to_hand(make_safety(upgraded))


def make_deceive_reality(upgraded: bool = False) -> Card:
    block = 7 if upgraded else 4
    return Card("Deceive Reality+" if upgraded else "Deceive Reality", 1, CardType.SKILL,
                lambda combat, target: _deceive_reality(combat, block, upgraded),
                targeted=False, upgraded=upgraded)


def _inner_peace(combat, draw):
    if combat.player.stance == "Calm":
        _set_stance(combat, None)
        combat.draw_cards(draw)
    else:
        _set_stance(combat, "Calm")


def make_inner_peace(upgraded: bool = False) -> Card:
    draw = 4 if upgraded else 3
    return Card("Inner Peace+" if upgraded else "Inner Peace", 1, CardType.SKILL,
                lambda combat, target: _inner_peace(combat, draw),
                targeted=False, upgraded=upgraded)


def _collect(combat, upgraded):
    x = combat._x_value
    combat.player.energy += x + (1 if upgraded else 0)


def make_collect(upgraded: bool = False) -> Card:
    return Card("Collect+" if upgraded else "Collect", 0, CardType.SKILL,
                lambda combat, target: _collect(combat, upgraded),
                targeted=False, is_x_cost=True, exhausts=True, upgraded=upgraded)


def _wreath_of_flame(combat, bonus):
    combat.pending_damage_bonus += bonus


def make_wreath_of_flame(upgraded: bool = False) -> Card:
    bonus = 8 if upgraded else 5
    return Card("Wreath of Flame+" if upgraded else "Wreath of Flame", 1, CardType.SKILL,
                lambda combat, target: _wreath_of_flame(combat, bonus),
                targeted=False, upgraded=upgraded)


def _wallop(combat, target, dmg):
    actual_dmg = combat.deal_attack_damage(combat.player, target, dmg)
    combat.player.gain_block(actual_dmg)


def make_wallop(upgraded: bool = False) -> Card:
    dmg = 12 if upgraded else 9
    return Card("Wallop+" if upgraded else "Wallop", 2, CardType.ATTACK,
                lambda combat, target: _wallop(combat, target, dmg),
                targeted=True, upgraded=upgraded)


def _carve_reality(combat, target, dmg, upgraded):
    combat.deal_attack_damage(combat.player, target, dmg)
    combat.add_card_to_hand(make_smite(upgraded))


def make_carve_reality(upgraded: bool = False) -> Card:
    dmg = 10 if upgraded else 6
    return Card("Carve Reality+" if upgraded else "Carve Reality", 1, CardType.ATTACK,
                lambda combat, target: _carve_reality(combat, target, dmg, upgraded),
                targeted=True, upgraded=upgraded)


def make_foreign_influence(upgraded: bool = False) -> Card:
    def _effect(combat, target):
        pool = [c for c in common_card_pool() if c.card_type == CardType.ATTACK]
        if pool:
            combat.add_card_to_hand(combat.rng.choice(pool))
    return Card("Foreign Influence+" if upgraded else "Foreign Influence", 0, CardType.SKILL,
                _effect, targeted=False, exhausts=True, upgraded=upgraded)


def make_windmill_strike(upgraded: bool = False) -> Card:
    state = {"dmg": 14 if upgraded else 10, "bonus": 0}
    increment = 5 if upgraded else 4

    def _effect(combat, target):
        combat.deal_attack_damage(combat.player, target, state["dmg"] + state["bonus"])

    def _retain(card, combat):
        state["bonus"] += increment

    return Card("Windmill Strike+" if upgraded else "Windmill Strike", 2, CardType.ATTACK,
                _effect, targeted=True, retain=True, retain_callback=_retain, upgraded=upgraded)


def _indignation(combat, vuln):
    if combat.player.stance == "Wrath":
        for m in combat.living_monsters:
            m.add_power(Vulnerable(vuln))
    else:
        _set_stance(combat, "Wrath")


def make_indignation(upgraded: bool = False) -> Card:
    vuln = 5 if upgraded else 3
    return Card("Indignation+" if upgraded else "Indignation", 1, CardType.SKILL,
                lambda combat, target: _indignation(combat, vuln),
                targeted=False, upgraded=upgraded)


def _talk_to_the_hand(combat, target, dmg, block):
    combat.deal_attack_damage(combat.player, target, dmg)
    target.add_power(TalkToTheHand(block))


def make_talk_to_the_hand(upgraded: bool = False) -> Card:
    dmg = 7 if upgraded else 5
    block = 3 if upgraded else 2
    return Card("Talk to the Hand+" if upgraded else "Talk to the Hand", 1, CardType.ATTACK,
                lambda combat, target: _talk_to_the_hand(combat, target, dmg, block),
                targeted=True, exhausts=True, upgraded=upgraded)


def _sanctity(combat, block):
    combat.player.gain_block(block)
    combat.draw_cards(2)


def make_sanctity(upgraded: bool = False) -> Card:
    block = 9 if upgraded else 6
    return Card("Sanctity+" if upgraded else "Sanctity", 1, CardType.SKILL,
                lambda combat, target: _sanctity(combat, block),
                targeted=False, upgraded=upgraded)


def make_foresight(upgraded: bool = False) -> Card:
    draw = 4 if upgraded else 3
    return Card("Foresight+" if upgraded else "Foresight", 1, CardType.POWER,
                lambda combat, target: combat.player.add_power(_ForesightPower(draw)),
                targeted=False, upgraded=upgraded)


def _simmering_fury(combat, draw):
    combat.draw_cards(draw)
    _set_stance(combat, "Wrath")


def make_simmering_fury(upgraded: bool = False) -> Card:
    draw = 3 if upgraded else 2
    return Card("Simmering Fury+" if upgraded else "Simmering Fury", 1, CardType.SKILL,
                lambda combat, target: _simmering_fury(combat, draw),
                targeted=False, upgraded=upgraded)


def _wheel_kick(combat, target, dmg):
    combat.deal_attack_damage(combat.player, target, dmg)
    combat.draw_cards(2)


def make_wheel_kick(upgraded: bool = False) -> Card:
    dmg = 20 if upgraded else 15
    return Card("Wheel Kick+" if upgraded else "Wheel Kick", 2, CardType.ATTACK,
                lambda combat, target: _wheel_kick(combat, target, dmg),
                targeted=True, upgraded=upgraded)


def _judgment(target, threshold):
    if target.hp <= threshold:
        target.hp = 0


def make_judgment(upgraded: bool = False) -> Card:
    threshold = 40 if upgraded else 30
    return Card("Judgment+" if upgraded else "Judgment", 1, CardType.SKILL,
                lambda combat, target: _judgment(target, threshold),
                targeted=True, upgraded=upgraded)


def _conjure_blade(combat, upgraded):
    x = combat._x_value
    expunger_x = x + 1 if upgraded else x
    combat.draw_pile.append(make_expunger(expunger_x, upgraded))
    combat.rng.shuffle(combat.draw_pile)


def make_conjure_blade(upgraded: bool = False) -> Card:
    return Card("Conjure Blade+" if upgraded else "Conjure Blade", 0, CardType.SKILL,
                lambda combat, target: _conjure_blade(combat, upgraded),
                targeted=False, is_x_cost=True, exhausts=True, upgraded=upgraded)


def make_master_reality(upgraded: bool = False) -> Card:
    return Card("Master Reality+" if upgraded else "Master Reality", 0 if upgraded else 1,
                CardType.POWER,
                lambda combat, target: combat.player.add_power(_MasterRealityPower()),
                targeted=False, upgraded=upgraded)


def _brilliance(combat, target, base):
    mantra_bonus = getattr(combat.player, '_mantra', 0)
    combat.deal_attack_damage(combat.player, target, base + mantra_bonus)


def make_brilliance(upgraded: bool = False) -> Card:
    base = 16 if upgraded else 12
    return Card("Brilliance+" if upgraded else "Brilliance", 1, CardType.ATTACK,
                lambda combat, target: _brilliance(combat, target, base),
                targeted=True, upgraded=upgraded)


def make_like_water(upgraded: bool = False) -> Card:
    amount = 7 if upgraded else 5
    return Card("Like Water+" if upgraded else "Like Water", 1, CardType.POWER,
                lambda combat, target: combat.player.add_power(LikeWater(amount)),
                targeted=False, upgraded=upgraded)


def _fasting(combat, amount):
    combat.player.add_power(Strength(amount))
    combat.player.add_power(Dexterity(amount))
    combat.player.add_power(Fasting(amount))


def make_fasting_card(upgraded: bool = False) -> Card:
    amount = 4 if upgraded else 3
    return Card("Fasting+" if upgraded else "Fasting", 2, CardType.POWER,
                lambda combat, target: _fasting(combat, amount),
                targeted=False, upgraded=upgraded)


def make_devotion_card(upgraded: bool = False) -> Card:
    amount = 3 if upgraded else 2
    return Card("Devotion+" if upgraded else "Devotion", 1, CardType.POWER,
                lambda combat, target: combat.player.add_power(Devotion(amount)),
                targeted=False, upgraded=upgraded)


def _blasphemy(combat):
    _set_stance(combat, "Divinity")
    # Blasphemer: die at start of next turn. Approximated as entering Divinity
    # plus marking for death -- the real C++ engine uses PS::BLASPHEMER.
    setattr(combat.player, '_blasphemer', True)


def make_blasphemy(upgraded: bool = False) -> Card:
    return Card("Blasphemy+" if upgraded else "Blasphemy", 1, CardType.SKILL,
                lambda combat, target: _blasphemy(combat),
                targeted=False, exhausts=True, upgraded=upgraded)


def _ragnarok(combat, dmg, hits):
    for _ in range(hits):
        living = combat.living_monsters
        if not living:
            break
        t = combat.rng.choice(living)
        combat.deal_attack_damage(combat.player, t, dmg)


def make_ragnarok(upgraded: bool = False) -> Card:
    dmg = 6 if upgraded else 5
    hits = 6 if upgraded else 5
    return Card("Ragnarok+" if upgraded else "Ragnarok", 3, CardType.ATTACK,
                lambda combat, target: _ragnarok(combat, dmg, hits),
                targeted=False, upgraded=upgraded)


def make_lesson_learned(upgraded: bool = False) -> Card:
    dmg = 13 if upgraded else 10
    return Card("Lesson Learned+" if upgraded else "Lesson Learned", 2, CardType.ATTACK,
                lambda combat, target: combat.deal_attack_damage(combat.player, target, dmg),
                targeted=True, exhausts=True, upgraded=upgraded)


def _scrawl(combat):
    max_hand = 10
    while len(combat.hand) < max_hand:
        if not combat.draw_pile and not combat.discard_pile:
            break
        combat.draw_cards(1)


def make_scrawl(upgraded: bool = False) -> Card:
    return Card("Scrawl+" if upgraded else "Scrawl", 0 if upgraded else 1, CardType.SKILL,
                lambda combat, target: _scrawl(combat),
                targeted=False, exhausts=True, upgraded=upgraded)


def _vault(combat):
    combat.skip_monster_turn = True
    combat.turn_should_end_early = True


def make_vault(upgraded: bool = False) -> Card:
    return Card("Vault+" if upgraded else "Vault", 2 if upgraded else 3, CardType.SKILL,
                lambda combat, target: _vault(combat),
                targeted=False, exhausts=True, upgraded=upgraded)


def _alpha(combat, upgraded):
    combat.draw_pile.append(make_beta(upgraded))
    combat.rng.shuffle(combat.draw_pile)


def make_alpha(upgraded: bool = False) -> Card:
    return Card("Alpha+" if upgraded else "Alpha", 1, CardType.SKILL,
                lambda combat, target: _alpha(combat, upgraded),
                targeted=False, exhausts=True, upgraded=upgraded)


def make_wish(upgraded: bool = False) -> Card:
    strength_amt = 4 if upgraded else 3
    return Card("Wish+" if upgraded else "Wish", 3, CardType.SKILL,
                lambda combat, target: combat.player.add_power(Strength(strength_amt)),
                targeted=False, exhausts=True, upgraded=upgraded)


def _omniscience(combat, self_card):
    others = [c for c in combat.hand if c is not self_card]
    if not others:
        return
    card = combat.rng.choice(others)
    auto_target = min(combat.living_monsters, key=lambda m: m.hp, default=None) if card.targeted else None
    card.play(combat, auto_target)
    card.play(combat, auto_target)
    combat.hand.remove(card)
    combat._exhaust(card)


def make_omniscience(upgraded: bool = False) -> Card:
    c = Card("Omniscience+" if upgraded else "Omniscience", 3 if upgraded else 4,
             CardType.SKILL, lambda combat, target: None,
             targeted=False, exhausts=True, upgraded=upgraded)
    c.play = lambda combat, target: _omniscience(combat, c)
    return c


def make_establishment(upgraded: bool = False) -> Card:
    return Card("Establishment+" if upgraded else "Establishment", 1, CardType.POWER,
                lambda combat, target: combat.player.add_power(_EstablishmentPower()),
                targeted=False, upgraded=upgraded)


def make_spirit_shield(upgraded: bool = False) -> Card:
    per_card = 4 if upgraded else 3
    return Card("Spirit Shield+" if upgraded else "Spirit Shield", 2, CardType.SKILL,
                lambda combat, target: combat.player.gain_block(per_card * len(combat.hand)),
                targeted=False, upgraded=upgraded)


def make_deva_form(upgraded: bool = False) -> Card:
    return Card("Deva Form+" if upgraded else "Deva Form", 3, CardType.POWER,
                lambda combat, target: combat.player.add_power(_DevaFormPower()),
                targeted=False, upgraded=upgraded)


def _deus_ex_machina(combat, upgraded):
    for _ in range(3):
        combat.add_card_to_hand(make_miracle())
    if upgraded:
        for _ in range(3):
            combat.discard_pile.append(make_miracle())


def make_deus_ex_machina(upgraded: bool = False) -> Card:
    return Card("Deus Ex Machina+" if upgraded else "Deus Ex Machina", 0 if upgraded else 1,
                CardType.SKILL,
                lambda combat, target: _deus_ex_machina(combat, upgraded),
                targeted=False, exhausts=True, upgraded=upgraded)
