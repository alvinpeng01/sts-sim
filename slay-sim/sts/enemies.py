"""Monsters and their intent AI.

Each monster picks an Intent (telegraphed to the player) at the end of its turn
for the next turn. The player-side AI reads ``monster.intent`` exactly like a
human reads the icon above an enemy's head.

Design note for the search layer: every monster exposes ``intent_options()``,
an exact probability table over its next move given its current AI state
(last move, turn count, ...). ``roll_intent`` just samples from that table.
This split matters because the expectimax solver needs the *exact*
distribution to branch over (a chance node), not a single rng sample --
sampling would let the solver "get lucky" against randomness it should be
taking an expectation over instead.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from .creatures import Creature
from .powers import Strength, Ritual, Weak, Frail, Vulnerable
from .cards import make_dazed, make_burn, make_slimed


class IntentType(Enum):
    ATTACK = "Attack"
    DEFEND = "Defend"
    BUFF = "Buff"
    ATTACK_DEFEND = "Attack+Defend"


@dataclass
class Intent:
    type: IntentType
    # damage shown to the player; None for non-attacks.
    damage: Optional[int] = None
    name: str = ""

    def __str__(self) -> str:
        if self.damage is not None:
            return f"{self.name} (deal {self.damage})"
        return f"{self.name} ({self.type.value})"


def _sample_weighted(options: List[Tuple[float, str]], rng: random.Random) -> str:
    r = rng.random()
    cumulative = 0.0
    for prob, move in options:
        cumulative += prob
        if r < cumulative:
            return move
    return options[-1][1]  # floating-point safety net


class Monster(Creature):
    def __init__(self, name: str, max_hp: int):
        super().__init__(name, max_hp)
        self.intent: Optional[Intent] = None
        self.turn_count = 0

    def intent_options(self) -> List[Tuple[float, str]]:
        """Exact (probability, move_name) distribution for the *next* move,
        given the monster's current AI state. Must sum to 1.0. Override per
        monster."""
        raise NotImplementedError

    def force_intent(self, move: str) -> None:
        """Set self.intent/telegraphed damage for ``move`` without consuming
        any randomness. Override per monster."""
        raise NotImplementedError

    def take_turn(self, combat) -> None:
        """Execute the currently telegraphed intent. Override per monster."""
        raise NotImplementedError

    def roll_intent(self, rng: random.Random) -> None:
        move = _sample_weighted(self.intent_options(), rng)
        self.force_intent(move)


class JawWorm(Monster):
    """Faithful-ish Jaw Worm (StS1 Act 1 elite-lite).

    - Turn 1 is always Chomp.
    - Afterwards: Bellow (buff), Thrash (attack+block), or Chomp (attack).

    ``intent_options`` below is the closed-form distribution of the game's
    original nested-rng anti-repeat logic (base weights 45/30/25 for
    Bellow/Thrash/Chomp, redirected when a guard triggers), derived by hand
    so search can enumerate it exactly instead of sampling it:

      last move Bellow          -> Thrash .525 / Chomp .475
      last move Chomp           -> Bellow .60  / Thrash .40
      last move Thrash (twice)  -> Bellow .63  / Chomp .37
      last move Thrash (once)   -> Bellow .45  / Thrash .30 / Chomp .25 (no guard fires)
    """

    CHOMP = "Chomp"
    THRASH = "Thrash"
    BELLOW = "Bellow"

    # A20 (verified via sts_lightspeed direct construction + trace, replacing
    # this class's earlier un-sourced A0-ish numbers): HP 44->42, Chomp
    # 11->12, Thrash unchanged at 7, Bellow's Strength grant 3->5.
    def __init__(self):
        super().__init__("Jaw Worm", max_hp=42)
        self.last_move: Optional[str] = None
        self.last_move_twice = False

    def _chomp_damage(self) -> int:
        return self.calc_attack_damage(12)

    def _thrash_damage(self) -> int:
        return self.calc_attack_damage(7)

    def intent_options(self) -> List[Tuple[float, str]]:
        if self.turn_count == 0:
            return [(1.0, self.CHOMP)]
        if self.last_move == self.BELLOW:
            return [(0.525, self.THRASH), (0.475, self.CHOMP)]
        if self.last_move == self.CHOMP:
            return [(0.60, self.BELLOW), (0.40, self.THRASH)]
        if self.last_move == self.THRASH and self.last_move_twice:
            return [(0.63, self.BELLOW), (0.37, self.CHOMP)]
        return [(0.45, self.BELLOW), (0.30, self.THRASH), (0.25, self.CHOMP)]

    def force_intent(self, move: str) -> None:
        if move == self.CHOMP:
            self.intent = Intent(IntentType.ATTACK, self._chomp_damage(), self.CHOMP)
        elif move == self.THRASH:
            self.intent = Intent(IntentType.ATTACK_DEFEND, self._thrash_damage(), self.THRASH)
        else:  # BELLOW
            self.intent = Intent(IntentType.BUFF, None, self.BELLOW)
        self._pending_move = move

    def take_turn(self, combat) -> None:
        move = self._pending_move
        if move == self.CHOMP:
            combat.deal_attack_damage(self, combat.player, 12)
        elif move == self.THRASH:
            combat.deal_attack_damage(self, combat.player, 7)
            self.gain_block(5)
        elif move == self.BELLOW:
            self.add_power(Strength(5))
            self.gain_block(6)

        # Track repeats for the anti-repeat guard.
        self.last_move_twice = self.last_move == move
        self.last_move = move
        self.turn_count += 1


class Cultist(Monster):
    """Turn 1: Incantation (Ritual buff, no attack). Every turn after: Dark
    Strike. Fully deterministic AI -- a useful contrast to Jaw Worm/Louse for
    the search layer, since its chance node is trivial (a single move at
    probability 1.0)."""

    INCANTATION = "Incantation"
    DARK_STRIKE = "Dark Strike"

    # A20: HP 50->56, Ritual (Strength/turn) 3->5, Dark Strike unchanged at 6.
    def __init__(self, ritual_amount: int = 5):
        super().__init__("Cultist", max_hp=56)
        self.ritual_amount = ritual_amount

    def intent_options(self) -> List[Tuple[float, str]]:
        if self.turn_count == 0:
            return [(1.0, self.INCANTATION)]
        return [(1.0, self.DARK_STRIKE)]

    def force_intent(self, move: str) -> None:
        if move == self.INCANTATION:
            self.intent = Intent(IntentType.BUFF, None, self.INCANTATION)
        else:
            self.intent = Intent(IntentType.ATTACK,
                                 self.calc_attack_damage(6), self.DARK_STRIKE)
        self._pending_move = move

    def take_turn(self, combat) -> None:
        if self._pending_move == self.INCANTATION:
            self.add_power(Ritual(self.ritual_amount))
        else:
            combat.deal_attack_damage(self, combat.player, 6)
        self.turn_count += 1


class Louse(Monster):
    """Red/Green Louse, simplified: Bite (attack) or Grow (permanent
    Strength), can't Grow twice in a row. HP and bite damage are randomized
    once at creation (matching the real game's per-instance variance), not
    per-turn -- so once a fight starts, search treats them as fixed/observed
    state, same as everything else about a monster."""

    BITE = "Bite"
    GROW = "Grow"

    # A20: HP/bite range shifted up ~1 (10-15/5-7 -> 11-16/6-8), Grow's
    # Strength grant 3->4, both read off a live trace's str-progression delta.
    def __init__(self, rng: Optional[random.Random] = None):
        rng = rng or random.Random()
        super().__init__("Louse", max_hp=rng.randint(11, 16))
        self.bite_damage = rng.randint(6, 8)
        self.last_move: Optional[str] = None

    def intent_options(self) -> List[Tuple[float, str]]:
        if self.last_move == self.GROW:
            return [(1.0, self.BITE)]
        return [(0.75, self.BITE), (0.25, self.GROW)]

    def force_intent(self, move: str) -> None:
        if move == self.BITE:
            self.intent = Intent(IntentType.ATTACK,
                                 self.calc_attack_damage(self.bite_damage), self.BITE)
        else:
            self.intent = Intent(IntentType.BUFF, None, self.GROW)
        self._pending_move = move

    def take_turn(self, combat) -> None:
        if self._pending_move == self.BITE:
            combat.deal_attack_damage(self, combat.player, self.bite_damage)
        else:
            self.add_power(Strength(4))
        self.last_move = self._pending_move


class AcidSlimeM(Monster):
    """Simplified: real Corrosive Spit adds an unplayable 'Slimed' card to
    the discard pile; here it applies Weak instead, since it's the same
    "makes your future turns worse" flavor without adding a fifth Status
    card type just for this one enemy."""

    TACKLE = "Tackle"
    CORRODE = "Corrosive Spit"
    LICK = "Lick"

    def __init__(self):
        super().__init__("Acid Slime (M)", max_hp=30)
        self.last_move: Optional[str] = None

    def intent_options(self) -> List[Tuple[float, str]]:
        if self.last_move == self.LICK:
            return [(0.55, self.TACKLE), (0.45, self.CORRODE)]
        return [(0.4, self.TACKLE), (0.3, self.CORRODE), (0.3, self.LICK)]

    def force_intent(self, move: str) -> None:
        if move == self.TACKLE:
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(12), move)
        elif move == self.CORRODE:
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(8), move)
        else:
            self.intent = Intent(IntentType.DEFEND, None, move)
        self._pending_move = move

    def take_turn(self, combat) -> None:
        move = self._pending_move
        if move == self.TACKLE:
            combat.deal_attack_damage(self, combat.player, 12)
        elif move == self.CORRODE:
            combat.deal_attack_damage(self, combat.player, 8)
            combat.player.add_power(Weak(1))
        else:
            combat.player.add_power(Weak(2))
        self.last_move = move


class SpikeSlimeM(Monster):
    """Frail-themed sibling of Acid Slime (M); same simplification note re:
    the real game's Slimed-card mechanic applies here too."""

    TACKLE = "Tackle"
    FLAME_TACKLE = "Flame Tackle"
    LICK = "Lick"

    def __init__(self):
        super().__init__("Spike Slime (M)", max_hp=30)
        self.last_move: Optional[str] = None

    def intent_options(self) -> List[Tuple[float, str]]:
        if self.last_move == self.LICK:
            return [(0.55, self.TACKLE), (0.45, self.FLAME_TACKLE)]
        return [(0.4, self.TACKLE), (0.3, self.FLAME_TACKLE), (0.3, self.LICK)]

    def force_intent(self, move: str) -> None:
        if move == self.TACKLE:
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(12), move)
        elif move == self.FLAME_TACKLE:
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(8), move)
        else:
            self.intent = Intent(IntentType.DEFEND, None, move)
        self._pending_move = move

    def take_turn(self, combat) -> None:
        move = self._pending_move
        if move == self.TACKLE:
            combat.deal_attack_damage(self, combat.player, 12)
        elif move == self.FLAME_TACKLE:
            combat.deal_attack_damage(self, combat.player, 8)
            combat.player.add_power(Frail(1))
        else:
            combat.player.add_power(Frail(2))
        self.last_move = move


class BlueSlaver(Monster):
    STAB = "Stab"
    RAKE = "Rake"

    def __init__(self):
        super().__init__("Blue Slaver", max_hp=48)

    def intent_options(self) -> List[Tuple[float, str]]:
        return [(0.5, self.STAB), (0.5, self.RAKE)]

    def force_intent(self, move: str) -> None:
        if move == self.STAB:
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(13), move)
        else:
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(8), move)
        self._pending_move = move

    def take_turn(self, combat) -> None:
        if self._pending_move == self.STAB:
            combat.deal_attack_damage(self, combat.player, 13)
        else:
            combat.deal_attack_damage(self, combat.player, 8)
            combat.player.add_power(Weak(1))


class Looter(Monster):
    """Simplified: real Looter steals gold (no gold system here) and can
    flee combat with Smoke Bomb after a few turns (no fleeing modeled --
    Smoke Bomb here is just a block move)."""

    MUG = "Mug"
    LUNGE = "Lunge"
    SMOKE_BOMB = "Smoke Bomb"

    def __init__(self):
        super().__init__("Looter", max_hp=46)

    def intent_options(self) -> List[Tuple[float, str]]:
        if self.turn_count == 0:
            return [(1.0, self.MUG)]
        return [(0.4, self.MUG), (0.4, self.LUNGE), (0.2, self.SMOKE_BOMB)]

    def force_intent(self, move: str) -> None:
        if move == self.MUG:
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(11), move)
        elif move == self.LUNGE:
            self.intent = Intent(IntentType.ATTACK_DEFEND, self.calc_attack_damage(14), move)  # A20: 12->14
        else:
            self.intent = Intent(IntentType.DEFEND, None, move)
        self._pending_move = move

    def take_turn(self, combat) -> None:
        move = self._pending_move
        if move == self.MUG:
            combat.deal_attack_damage(self, combat.player, 11)
        elif move == self.LUNGE:
            combat.deal_attack_damage(self, combat.player, 14)
            self.gain_block(6)
        else:
            self.gain_block(9)
        self.turn_count += 1


class GremlinNob(Monster):
    """Elite. Simplified: real Bellow also grants "Enrage" (gains Strength
    whenever you play a Skill) -- that needs a monster-side on-play-skill
    hook the engine doesn't have yet, so it's omitted; Bellow here is just
    the self-Strength buff."""

    BELLOW = "Bellow"
    RUSH = "Rush"
    SKULL_BASH = "Skull Bash"

    # A20: HP 88->90, Rush 14->16 (verified via trace); Skull Bash's 7 is
    # NOT independently confirmed at A20 (the live trace only ever showed
    # Rush's damage across 12 turns, so this class's exact alternation rule
    # is unverified, not just the number -- left as-is rather than guessed).
    def __init__(self):
        super().__init__("Gremlin Nob", max_hp=90)
        self.last_move: Optional[str] = None

    def intent_options(self) -> List[Tuple[float, str]]:
        if self.turn_count == 0:
            return [(1.0, self.BELLOW)]
        if self.last_move == self.SKULL_BASH:
            return [(1.0, self.RUSH)]
        return [(0.6, self.RUSH), (0.4, self.SKULL_BASH)]

    def force_intent(self, move: str) -> None:
        if move == self.BELLOW:
            self.intent = Intent(IntentType.BUFF, None, move)
        elif move == self.RUSH:
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(16), move)
        else:
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(7), move)
        self._pending_move = move

    def take_turn(self, combat) -> None:
        move = self._pending_move
        if move == self.BELLOW:
            self.add_power(Strength(3))
        elif move == self.RUSH:
            combat.deal_attack_damage(self, combat.player, 16)
        else:
            combat.deal_attack_damage(self, combat.player, 7)
            combat.player.add_power(Vulnerable(2))
        self.last_move = move
        self.turn_count += 1


class Lagavulin(Monster):
    """Elite. Sleeps (no action) for its first 3 turns, or until it takes
    any damage, whichever comes first; simplified: real Siphon Soul (which
    debuffs the player's Strength/Dexterity) is folded into a single
    negative-Strength application rather than modeling Dexterity too."""

    SLEEPING = "Sleeping"
    ATTACK_MOVE = "Attack"
    SIPHON_SOUL = "Siphon Soul"
    WAKE_TURNS = 3

    # A20: HP 112->115, attack move 18->20 (Siphon Soul's -1 Strength and
    # block gain aren't independently confirmed, left as-is).
    def __init__(self):
        super().__init__("Lagavulin", max_hp=115)
        self.asleep = True
        self.last_move: Optional[str] = None

    def take_damage(self, incoming: int) -> int:
        if self.asleep:
            self.asleep = False
        return super().take_damage(incoming)

    def intent_options(self) -> List[Tuple[float, str]]:
        if self.asleep:
            return [(1.0, self.SLEEPING)]
        if self.last_move == self.SIPHON_SOUL:
            return [(1.0, self.ATTACK_MOVE)]
        return [(0.7, self.ATTACK_MOVE), (0.3, self.SIPHON_SOUL)]

    def force_intent(self, move: str) -> None:
        if move == self.SLEEPING:
            self.intent = Intent(IntentType.DEFEND, None, move)
        elif move == self.ATTACK_MOVE:
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(20), move)
        else:  # SIPHON_SOUL: a debuff, but IntentType has no DEBUFF variant
            self.intent = Intent(IntentType.DEFEND, None, move)
        self._pending_move = move

    def take_turn(self, combat) -> None:
        move = self._pending_move
        if move == self.SLEEPING:
            pass  # no action while asleep
        elif move == self.ATTACK_MOVE:
            combat.deal_attack_damage(self, combat.player, 20)
        else:
            combat.player.add_power(Strength(-1))
            self.gain_block(6)
        self.last_move = move
        self.turn_count += 1
        if self.turn_count >= self.WAKE_TURNS:
            self.asleep = False


class Guardian(Monster):
    """Act 1 boss. Alternates Charging Up!/Thrash while offensive; if it
    takes >=30 damage in a single player turn it Mode Shifts into a
    defensive phase (heavy block, Twin Slam/Roll Attack), then shifts back
    to offensive (with a Strength "Enrage") after a few defensive turns.
    Deterministic move order -- the real game's AI is pattern-based too, so
    this isn't a simplification of the AI's *shape*, just its exact
    thresholds/numbers."""

    CHARGING_UP = "Charging Up!"
    THRASH = "Thrash"
    MODE_SHIFT = "Mode Shift"
    TWIN_SLAM = "Twin Slam"
    ROLL_ATTACK = "Roll Attack"
    MODE_SHIFT_THRESHOLD = 30
    DEFENSIVE_TURN_LIMIT = 3

    # A20: HP 240->250, Thrash 32->36. Twin Slam/Roll Attack left unchanged
    # -- the live A20 trace's defensive-phase hit showed a "5x4" pattern that
    # doesn't match either move's existing shape (8x2 / 10x1), suggesting a
    # PRE-EXISTING mechanic mismatch unrelated to ascension scaling; flagged
    # rather than guessed at, since fixing it means re-deriving the
    # defensive-phase moveset, not just updating a number.
    def __init__(self):
        super().__init__("The Guardian", max_hp=250)
        self.mode = "offensive"
        self.dmg_taken_this_turn = 0
        self.defensive_turns = 0

    def take_damage(self, incoming: int) -> int:
        hp_loss = super().take_damage(incoming)
        self.dmg_taken_this_turn += hp_loss
        return hp_loss

    def intent_options(self) -> List[Tuple[float, str]]:
        if self.turn_count == 0:
            return [(1.0, self.CHARGING_UP)]
        if self.mode == "offensive":
            if self.dmg_taken_this_turn >= self.MODE_SHIFT_THRESHOLD:
                return [(1.0, self.MODE_SHIFT)]
            return [(1.0, self.THRASH if self._pending_move == self.CHARGING_UP else self.CHARGING_UP)]
        if self.defensive_turns >= self.DEFENSIVE_TURN_LIMIT:
            return [(1.0, self.MODE_SHIFT)]
        return [(1.0, self.ROLL_ATTACK if self._pending_move == self.TWIN_SLAM else self.TWIN_SLAM)]

    def force_intent(self, move: str) -> None:
        if move == self.CHARGING_UP:
            self.intent = Intent(IntentType.DEFEND, None, move)
        elif move == self.THRASH:
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(36), move)
        elif move == self.MODE_SHIFT:
            self.intent = Intent(IntentType.DEFEND, None, move)
        elif move == self.TWIN_SLAM:
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(8), move)
        else:  # ROLL_ATTACK
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(10), move)
        self._pending_move = move

    def take_turn(self, combat) -> None:
        move = self._pending_move
        if move == self.CHARGING_UP:
            self.gain_block(9)
        elif move == self.THRASH:
            combat.deal_attack_damage(self, combat.player, 36)
        elif move == self.MODE_SHIFT:
            self.gain_block(20)
            if self.mode == "offensive":
                self.mode = "defensive"
                self.defensive_turns = 0
            else:
                self.mode = "offensive"
                self.add_power(Strength(3))
        elif move == self.TWIN_SLAM:
            combat.deal_attack_damage(self, combat.player, 8)
            combat.deal_attack_damage(self, combat.player, 8)
            self.defensive_turns += 1
        else:  # ROLL_ATTACK
            combat.deal_attack_damage(self, combat.player, 10)
            self.defensive_turns += 1

        self.dmg_taken_this_turn = 0
        self.turn_count += 1


class Hexaghost(Monster):
    """Act 1 boss. Opens with Activate (does nothing) then Divider, then
    repeats the 7-move cycle Sear/Tackle/Sear/Inflame/Tackle/Sear/Inferno
    for the rest of combat -- fully deterministic, no chance node.

    Divider's damage depends on the player's HP at the moment it resolves
    ((floor(player_hp/12)+1)*6), which force_intent() can't compute (it only
    has ``self``, not ``combat``) -- so its telegraphed Intent carries
    damage=None (an honest "unknown magnitude" rather than a guessed number)
    and the real damage is computed for real in take_turn() using the
    player's actual current HP. Sear/Inferno's "adds Burn" is inert filler
    here, same simplification this engine already applies to Dazed/Wound --
    it doesn't model Burn's own "take 2 damage if in hand at end of turn"
    effect, so adding real Burns wouldn't do anything different from adding
    nothing; not modeling "upgrades all Burns in your deck" for the same
    reason (nothing here reads an upgraded/plain distinction on Burn)."""

    ACTIVATE = "Activate"
    DIVIDER = "Divider"
    SEAR = "Sear"
    TACKLE = "Tackle"
    INFLAME = "Inflame"
    INFERNO = "Inferno"

    _CYCLE = [SEAR, TACKLE, SEAR, INFLAME, TACKLE, SEAR, INFERNO]

    # A20 (Ascension 9+): HP 250->264. Sear adds 2 Burn (Ascension 19+,
    # 1 base). Inferno hits 3x6 (Ascension 4+, 2x6 base). Highest-tier
    # numbers used throughout, same convention as Guardian above.
    def __init__(self):
        super().__init__("Hexaghost", max_hp=264)

    def intent_options(self) -> List[Tuple[float, str]]:
        if self.turn_count == 0:
            return [(1.0, self.ACTIVATE)]
        if self.turn_count == 1:
            return [(1.0, self.DIVIDER)]
        idx = (self.turn_count - 2) % len(self._CYCLE)
        return [(1.0, self._CYCLE[idx])]

    def force_intent(self, move: str) -> None:
        if move == self.ACTIVATE:
            self.intent = Intent(IntentType.DEFEND, None, move)
        elif move == self.DIVIDER:
            self.intent = Intent(IntentType.ATTACK, None, move)  # magnitude depends on player HP at resolve time -- see class docstring
        elif move == self.SEAR:
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(6), move)
        elif move == self.TACKLE:
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(5), move)  # hits 2x
        elif move == self.INFLAME:
            self.intent = Intent(IntentType.DEFEND, None, move)
        else:  # INFERNO
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(6), move)  # hits 3x
        self._pending_move = move

    def take_turn(self, combat) -> None:
        move = self._pending_move
        if move == self.ACTIVATE:
            pass
        elif move == self.DIVIDER:
            n = combat.player.hp // 12
            combat.deal_attack_damage(self, combat.player, (n + 1) * 6)
        elif move == self.SEAR:
            combat.deal_attack_damage(self, combat.player, 6)
            combat.discard_pile.append(make_burn())
            combat.discard_pile.append(make_burn())
        elif move == self.TACKLE:
            combat.deal_attack_damage(self, combat.player, 5)
            combat.deal_attack_damage(self, combat.player, 5)
        elif move == self.INFLAME:
            self.gain_block(12)
            self.add_power(Strength(2))
        else:  # INFERNO
            for _ in range(3):
                combat.deal_attack_damage(self, combat.player, 6)
            for _ in range(3):
                combat.discard_pile.append(make_burn())
        self.turn_count += 1


class SlimeBoss(Monster):
    """Act 1 boss. Repeats Goop Spray/Preparing/Slam -- except once its HP
    drops to half or below, its *next* move is Split instead (wherever that
    falls in the cycle), which removes it from combat and spawns one Acid
    Slime (M) and one Spike Slime (M), each with Slime Boss's own current HP
    transplanted in (real text: "spawns...with its current HP"). Real Split
    spawns the *Large* size of each slime -- this project's Monster roster
    only has the Medium implementations (AcidSlimeM/SpikeSlimeM), so this
    reuses those for the move sets, just with HP overridden to match. Same
    "closest available ctor, flagged" approximation already used for the
    Large/Small slime id aliases in sts/bridge/state_mapper.py."""

    GOOP_SPRAY = "Goop Spray"
    PREPARING = "Preparing"
    SLAM = "Slam"
    SPLIT = "Split"

    _CYCLE = [GOOP_SPRAY, PREPARING, SLAM]

    # A20 (Ascension 9+): HP 140->150. Slam 35->38 (Ascension 4+). Goop
    # Spray adds 5 Slimed (Ascension 19+, 3 base) -- highest-tier numbers
    # used throughout.
    def __init__(self):
        super().__init__("Slime Boss", max_hp=150)
        self._split_next = False

    def intent_options(self) -> List[Tuple[float, str]]:
        if not self._split_next and self.hp <= self.max_hp // 2:
            self._split_next = True
        if self._split_next:
            return [(1.0, self.SPLIT)]
        idx = self.turn_count % len(self._CYCLE)
        return [(1.0, self._CYCLE[idx])]

    def force_intent(self, move: str) -> None:
        if move == self.GOOP_SPRAY:
            self.intent = Intent(IntentType.DEFEND, None, move)
        elif move == self.PREPARING:
            self.intent = Intent(IntentType.DEFEND, None, move)
        elif move == self.SLAM:
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(38), move)
        else:  # SPLIT
            self.intent = Intent(IntentType.DEFEND, None, move)
        self._pending_move = move

    def take_turn(self, combat) -> None:
        move = self._pending_move
        if move == self.GOOP_SPRAY:
            for _ in range(5):
                combat.discard_pile.append(make_slimed())
        elif move == self.PREPARING:
            pass
        elif move == self.SLAM:
            combat.deal_attack_damage(self, combat.player, 38)
        else:  # SPLIT
            current_hp = self.hp
            acid = AcidSlimeM()
            acid.max_hp = current_hp
            acid.hp = current_hp
            spike = SpikeSlimeM()
            spike.max_hp = current_hp
            spike.hp = current_hp
            combat.monsters.append(acid)
            combat.monsters.append(spike)
            acid.roll_intent(combat.rng)
            spike.roll_intent(combat.rng)
            self.hp = 0  # Slime Boss itself disappears
        self.turn_count += 1


class Sentry(Monster):
    """Act 1 elite (fought as a trio). Deterministically alternates Bolt
    (attack) and Beam (shuffles a Dazed into the draw pile) -- no chance
    node at all, unlike most of the roster, so 3 of these together stress
    the *action-space* axis (many monsters, many single-target choices)
    rather than the chance-branching axis. Simplified from the real
    synchronized-trio pattern: each instance alternates independently from
    its own starting phase rather than the whole trio sharing one clock."""

    BOLT = "Bolt"
    BEAM = "Beam"

    # A20: HP ~40->44 (real trio rolls 43-45, individual variance not
    # modeled here, just the shifted center), Bolt 9->10.
    def __init__(self, starts_with_bolt: bool = True):
        super().__init__("Sentry", max_hp=44)
        self._starts_with_bolt = starts_with_bolt

    def intent_options(self) -> List[Tuple[float, str]]:
        bolt_turn = (self.turn_count % 2 == 0) == self._starts_with_bolt
        return [(1.0, self.BOLT if bolt_turn else self.BEAM)]

    def force_intent(self, move: str) -> None:
        if move == self.BOLT:
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(10), move)
        else:
            self.intent = Intent(IntentType.DEFEND, None, move)
        self._pending_move = move

    def take_turn(self, combat) -> None:
        if self._pending_move == self.BOLT:
            combat.deal_attack_damage(self, combat.player, 10)
        else:
            combat.draw_pile.append(make_dazed())
            combat.rng.shuffle(combat.draw_pile)
        self.turn_count += 1


# NOT independently verified at A20: a live trace of the real GREMLIN_GANG
# encounter shows Gremlin Wizard + Fat Gremlin x2 + Sneaky Gremlin, not this
# class -- a pre-existing composition mismatch discovered during the A20
# pass, unrelated to ascension scaling and out of scope to fix here (see
# encounter_gremlin_gang()). Left at its original numbers.
class MadGremlin(Monster):
    def __init__(self):
        super().__init__("Mad Gremlin", max_hp=21)

    def intent_options(self) -> List[Tuple[float, str]]:
        return [(1.0, "Scratch")]

    def force_intent(self, move: str) -> None:
        self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(7), move)
        self._pending_move = move

    def take_turn(self, combat) -> None:
        combat.deal_attack_damage(self, combat.player, 7)


class SneakyGremlin(Monster):
    def __init__(self):
        super().__init__("Sneaky Gremlin", max_hp=15)

    def intent_options(self) -> List[Tuple[float, str]]:
        return [(1.0, "Puncture")]

    def force_intent(self, move: str) -> None:
        self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(10), move)
        self._pending_move = move

    def take_turn(self, combat) -> None:
        combat.deal_attack_damage(self, combat.player, 10)


class FatGremlin(Monster):
    def __init__(self):
        super().__init__("Fat Gremlin", max_hp=15)

    def intent_options(self) -> List[Tuple[float, str]]:
        return [(1.0, "Smash")]

    def force_intent(self, move: str) -> None:
        self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(5), move)  # A20: 4->5
        self._pending_move = move

    def take_turn(self, combat) -> None:
        combat.deal_attack_damage(self, combat.player, 5)
        combat.player.add_power(Weak(1))


class Byrd(Monster):
    """Act 2 basic. Simplified: real Fly grants literal untargetability;
    here it's a large self-block instead, avoiding a new "can't be
    targeted" mechanic for a single monster's flavor move."""

    PECK = "Peck"
    CAW = "Caw"
    FLY = "Fly"

    def __init__(self):
        super().__init__("Byrd", max_hp=32)
        self.last_move: Optional[str] = None

    def intent_options(self) -> List[Tuple[float, str]]:
        if self.turn_count == 0:
            return [(1.0, self.CAW)]
        if self.last_move == self.FLY:
            return [(1.0, self.PECK)]
        return [(0.5, self.PECK), (0.5, self.FLY)]

    def force_intent(self, move: str) -> None:
        if move == self.PECK:
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(6), move)
        elif move == self.CAW:
            self.intent = Intent(IntentType.BUFF, None, move)
        else:
            self.intent = Intent(IntentType.DEFEND, None, move)
        self._pending_move = move

    def take_turn(self, combat) -> None:
        move = self._pending_move
        if move == self.PECK:
            combat.deal_attack_damage(self, combat.player, 6)
        elif move == self.CAW:
            self.add_power(Strength(1))
        else:
            self.gain_block(12)
        self.last_move = move
        self.turn_count += 1


class Mystic(Monster):
    """Act 2 elite (fought alongside a Centurion). Both moves buff/shield
    itself AND its ally, rather than acting on the player -- the roster's
    first cross-monster interaction. Simplified: real Mystic targets
    whichever ally has the least HP; here it just targets the first other
    living monster found."""

    HEAL = "Heal"
    BUFF = "Buff"

    # A20: HP 48->51. Real Mystic apparently also has a damaging move (a
    # live trace showed occasional 9x1 hits) this buff-only class doesn't
    # model -- a pre-existing mechanic-shape gap, not touched here.
    def __init__(self):
        super().__init__("Mystic", max_hp=51)
        self.last_move: Optional[str] = None

    def intent_options(self) -> List[Tuple[float, str]]:
        if self.last_move == self.BUFF:
            return [(1.0, self.HEAL)]
        return [(0.5, self.HEAL), (0.5, self.BUFF)]

    def force_intent(self, move: str) -> None:
        self.intent = Intent(IntentType.BUFF, None, move)
        self._pending_move = move

    def _ally(self, combat):
        for m in combat.living_monsters:
            if m is not self:
                return m
        return None

    def take_turn(self, combat) -> None:
        move = self._pending_move
        ally = self._ally(combat)
        if move == self.HEAL:
            self.gain_block(12)
            if ally is not None:
                ally.gain_block(12)
        else:
            self.add_power(Strength(2))
            if ally is not None:
                ally.add_power(Strength(2))
        self.last_move = move


class Centurion(Monster):
    """Act 2 elite (fought alongside a Mystic)."""

    SLASH = "Slash"
    FURY = "Fury"

    # HP corrected to real 83 (this class's original 52 was already wrong
    # even at A0 -- not an ascension gap, a pre-existing error caught while
    # sourcing A20 ground truth). Slash 12->14 confirmed at A20; Fury's 6 is
    # unverified (the live trace never showed its distinct 2-hit shape).
    def __init__(self):
        super().__init__("Centurion", max_hp=83)
        self.last_move: Optional[str] = None

    def intent_options(self) -> List[Tuple[float, str]]:
        if self.last_move == self.FURY:
            return [(1.0, self.SLASH)]
        return [(0.6, self.SLASH), (0.4, self.FURY)]

    def force_intent(self, move: str) -> None:
        if move == self.SLASH:
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(14), move)
        else:
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(6), move)  # hits twice
        self._pending_move = move

    def take_turn(self, combat) -> None:
        if self._pending_move == self.SLASH:
            combat.deal_attack_damage(self, combat.player, 14)
        else:
            combat.deal_attack_damage(self, combat.player, 6)
            combat.deal_attack_damage(self, combat.player, 6)
        self.last_move = self._pending_move


class Champ(Monster):
    """Act 2 boss. Simplified from the real multi-mechanic Champ (which also
    has a stacking-Strength taunt move and a temporary self-weaken) down to:
    alternating Face Slap/Gut Check, a one-time Strength surge the first
    time its HP drops to half or below, and an Execute finisher afterward --
    same "threshold flips behavior" shape as Guardian's mode shift, applied
    to a simpler single-trigger case instead of a repeating one."""

    FACE_SLAP = "Face Slap"
    GUT_CHECK = "Gut Check"
    ANGER = "Anger"
    EXECUTE = "Execute"

    # A20: HP 420->440, Face Slap 10->14 confirmed. Gut Check/Execute/Anger
    # left unchanged -- the live trace showed an ambiguous 18x1 hit and
    # unexplained 0-damage turns this class's simplified moveset doesn't
    # have a clean slot for, and Execute/Anger's real HP-enrage never
    # triggered in a 10-turn window (nothing was damaging it), so neither
    # was observable to verify.
    def __init__(self):
        super().__init__("Champ", max_hp=440)
        self.enraged = False
        self.last_move: Optional[str] = None

    def intent_options(self) -> List[Tuple[float, str]]:
        if not self.enraged and self.hp <= self.max_hp // 2:
            return [(1.0, self.ANGER)]
        if self.enraged:
            return [(1.0, self.EXECUTE if self.last_move == self.FACE_SLAP else self.FACE_SLAP)]
        return [(1.0, self.GUT_CHECK if self.last_move == self.FACE_SLAP else self.FACE_SLAP)]

    def force_intent(self, move: str) -> None:
        if move == self.FACE_SLAP:
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(14), move)
        elif move == self.GUT_CHECK:
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(9), move)
        elif move == self.ANGER:
            self.intent = Intent(IntentType.BUFF, None, move)
        else:  # EXECUTE
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(20), move)
        self._pending_move = move

    def take_turn(self, combat) -> None:
        move = self._pending_move
        if move == self.FACE_SLAP:
            combat.deal_attack_damage(self, combat.player, 14)
        elif move == self.GUT_CHECK:
            combat.deal_attack_damage(self, combat.player, 9)
            self.gain_block(15)
        elif move == self.ANGER:
            self.add_power(Strength(5))
            self.enraged = True
        else:  # EXECUTE
            combat.deal_attack_damage(self, combat.player, 20)
        self.last_move = move


# --- mid-fight-spawn bosses/elite: added to test whether search-based play
# (vs. the lightspeed RL policy) correctly prioritizes attacking spawned
# minions rather than face-tanking their damage while focusing the main
# monster. HP totals and spawn timing below are ground truth, pulled
# directly from sts_lightspeed's BattleContext via new_battle() + stepping
# through end-turn-only playouts (not guessed); move damage values and
# cadence (Hyper Beam every 3rd Automaton attack turn, Gremlin Wizard's
# Ultimate Blast every 4th of its own turns, Collector's buff-every-3rd-turn)
# are read off the same traces but the exact underlying AI-state logic that
# produces them is approximated, consistent with every other monster in this
# file -- plausible, not verified against datamined source.
class BronzeOrb(Monster):
    """Spawned in pairs by Bronze Automaton. Real orbs' attack/idle cycles
    run offset from each other in a way that looks irregular over a short
    trace; approximated here as an independent 50/50 coin flip per turn
    rather than reverse-engineering an exact per-orb clock."""

    def __init__(self, hp: int):
        super().__init__("Bronze Orb", max_hp=hp)

    def intent_options(self) -> List[Tuple[float, str]]:
        return [(0.5, "Attack"), (0.5, "Idle")]

    def force_intent(self, move: str) -> None:
        if move == "Attack":
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(8), move)  # unchanged at A20
        else:
            self.intent = Intent(IntentType.DEFEND, None, move)
        self._pending_move = move

    def take_turn(self, combat) -> None:
        if self._pending_move == "Attack":
            combat.deal_attack_damage(self, combat.player, 8)


class Automaton(Monster):
    """Act 2 boss. Opens by spawning 2 Bronze Orbs (0-damage turn), then
    alternates an attack turn (Flail, or Hyper Beam every 3rd attack turn)
    with a Boost turn (+3 Strength, 0 damage)."""

    SPAWN = "Spawn Orbs"
    FLAIL = "Flail"
    HYPER_BEAM = "Hyper Beam"
    BOOST = "Boost"

    # A20: Automaton HP 300->320, orb HP 56/52->58/54 (see take_turn's SPAWN
    # branch), Flail 7->8/hit, Hyper Beam 45->50, Boost's Strength 3->4.
    def __init__(self):
        super().__init__("Bronze Automaton", max_hp=320)
        self.attack_turns = 0

    def intent_options(self) -> List[Tuple[float, str]]:
        if self.turn_count == 0:
            return [(1.0, self.SPAWN)]
        if self.turn_count % 2 == 1:
            return [(1.0, self.HYPER_BEAM if (self.attack_turns + 1) % 3 == 0 else self.FLAIL)]
        return [(1.0, self.BOOST)]

    def force_intent(self, move: str) -> None:
        if move == self.SPAWN:
            self.intent = Intent(IntentType.DEFEND, None, move)
        elif move == self.FLAIL:
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(8), move)  # hits twice
        elif move == self.HYPER_BEAM:
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(50), move)
        else:  # BOOST
            self.intent = Intent(IntentType.BUFF, None, move)
        self._pending_move = move

    def take_turn(self, combat) -> None:
        move = self._pending_move
        if move == self.SPAWN:
            orb1, orb2 = BronzeOrb(58), BronzeOrb(54)
            combat.monsters.append(orb1)
            combat.monsters.append(orb2)
            orb1.roll_intent(combat.rng)
            orb2.roll_intent(combat.rng)
        elif move == self.FLAIL:
            combat.deal_attack_damage(self, combat.player, 8)
            combat.deal_attack_damage(self, combat.player, 8)
            self.attack_turns += 1
        elif move == self.HYPER_BEAM:
            combat.deal_attack_damage(self, combat.player, 50)
            self.attack_turns += 1
        else:  # BOOST
            self.add_power(Strength(4))
        self.turn_count += 1


class TorchHead(Monster):
    """Spawned in pairs by The Collector. Deterministic single move, no
    chance branching -- same shape as MadGremlin/FatGremlin."""

    # A20: HP 38->44 (attack damage confirmed unchanged at 7).
    def __init__(self):
        super().__init__("Torch Head", max_hp=44)

    def intent_options(self) -> List[Tuple[float, str]]:
        return [(1.0, "Scorch")]

    def force_intent(self, move: str) -> None:
        self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(7), move)
        self._pending_move = move

    def take_turn(self, combat) -> None:
        combat.deal_attack_damage(self, combat.player, 7)


class Collector(Monster):
    """Act 2 boss. Opens by spawning 2 Torch Heads (0-damage turn), then
    cycles Fiery Bolt, Fiery Bolt, buff-self-and-torches (+3 Strength each)."""

    SPAWN = "Spawn Torch Heads"
    FIERY_BOLT = "Fiery Bolt"
    BUFF = "Mega Debuff"

    # A20: HP 282->300, Fiery Bolt 18->21, buff's Strength grant 3->5.
    def __init__(self):
        super().__init__("The Collector", max_hp=300)

    def intent_options(self) -> List[Tuple[float, str]]:
        if self.turn_count == 0:
            return [(1.0, self.SPAWN)]
        return [(1.0, self.BUFF if (self.turn_count - 1) % 3 == 2 else self.FIERY_BOLT)]

    def force_intent(self, move: str) -> None:
        if move == self.SPAWN:
            self.intent = Intent(IntentType.DEFEND, None, move)
        elif move == self.FIERY_BOLT:
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(21), move)
        else:  # BUFF
            self.intent = Intent(IntentType.BUFF, None, move)
        self._pending_move = move

    def take_turn(self, combat) -> None:
        move = self._pending_move
        if move == self.SPAWN:
            t1, t2 = TorchHead(), TorchHead()
            combat.monsters.append(t1)
            combat.monsters.append(t2)
            t1.roll_intent(combat.rng)
            t2.roll_intent(combat.rng)
        elif move == self.FIERY_BOLT:
            combat.deal_attack_damage(self, combat.player, 21)
        else:  # BUFF
            self.add_power(Strength(5))
            for m in combat.monsters:
                if isinstance(m, TorchHead) and not m.is_dead:
                    m.add_power(Strength(5))
        self.turn_count += 1


class GremlinWizard(Monster):
    """One of Gremlin Leader's two starting escorts. Charges for 3 turns,
    then Ultimate Blast on the 4th."""

    # A20: HP 21->22, Ultimate Blast 25->30.
    def __init__(self):
        super().__init__("Gremlin Wizard", max_hp=22)

    def intent_options(self) -> List[Tuple[float, str]]:
        return [(1.0, "Ultimate Blast" if self.turn_count % 4 == 1 else "Charging")]

    def force_intent(self, move: str) -> None:
        if move == "Ultimate Blast":
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(30), move)
        else:
            self.intent = Intent(IntentType.DEFEND, None, move)
        self._pending_move = move

    def take_turn(self, combat) -> None:
        if self._pending_move == "Ultimate Blast":
            combat.deal_attack_damage(self, combat.player, 30)
        self.turn_count += 1


class GremlinLeader(Monster):
    """Act 2 elite. Starts alongside a Gremlin Wizard + Fat Gremlin escort
    (see encounter_gremlin_leader). Alternates Stab (3x6 dmg) with a buff
    turn: Encourage (+3 Strength to itself and each living escort) normally,
    or Rally (re-summon any dead escort) if an escort has died since its
    last buff turn -- the actual mechanic this encounter exists to test,
    since a policy/search that ignores the escorts just eats free damage
    turn after turn instead of ever triggering Rally's replacement cost.

    intent_options() has no `combat` access (by the shared Monster
    interface), so it can't check escort HP itself; `_all_escorts_alive` is
    a cache written by take_turn() (which does have `combat`) for the
    immediately-following roll_intent() call to read -- same
    take-turn-then-roll-next-intent sequencing every monster already goes
    through in CombatState.enemy_turn()."""

    STAB = "Stab"
    ENCOURAGE = "Encourage"
    RALLY = "Rally"

    # A20: HP 141->146, Encourage's Strength grant 3->5 (Stab confirmed
    # unchanged at 6/hit).
    def __init__(self):
        super().__init__("Gremlin Leader", max_hp=146)
        self._all_escorts_alive = True

    def _escorts(self, combat) -> List[Monster]:
        return [m for m in combat.monsters if isinstance(m, (GremlinWizard, FatGremlin))]

    def intent_options(self) -> List[Tuple[float, str]]:
        if self.turn_count % 2 == 0:
            return [(1.0, self.STAB)]
        return [(1.0, self.ENCOURAGE if self._all_escorts_alive else self.RALLY)]

    def force_intent(self, move: str) -> None:
        if move == self.STAB:
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(6), move)  # hits 3x
        else:
            self.intent = Intent(IntentType.BUFF, None, move)
        self._pending_move = move

    def take_turn(self, combat) -> None:
        move = self._pending_move
        escorts = self._escorts(combat)
        if move == self.STAB:
            for _ in range(3):
                combat.deal_attack_damage(self, combat.player, 6)
        elif move == self.RALLY:
            for m in escorts:
                if m.is_dead:
                    replacement = type(m)()
                    combat.monsters.append(replacement)
                    replacement.roll_intent(combat.rng)
        else:  # ENCOURAGE
            self.add_power(Strength(5))
            for m in escorts:
                if not m.is_dead:
                    m.add_power(Strength(5))
        self._all_escorts_alive = all(not m.is_dead for m in self._escorts(combat))
        self.turn_count += 1


class Chosen(Monster):
    """Act 2 elite. Opens with Poke then Hex (a permanent-for-the-rest-of-
    combat effect: shuffles a Dazed into the draw pile whenever the player
    plays a non-Attack card -- the only monster that needs that hook, added
    to CombatState.play_card in sts/combat.py as a plain flag check rather
    than a new generic "monster reacts to card type" mechanism). After that,
    alternates a debuff pair (50% Debilitate/50% Drain) with an attack pair
    (60% Poke/40% Zap) every other turn."""

    POKE = "Poke"
    HEX = "Hex"
    DEBILITATE = "Debilitate"
    DRAIN = "Drain"
    ZAP = "Zap"

    # A20 (Ascension 7+): HP 98-103 (randomized per-instance, matching
    # Louse's precedent above). Poke/Debilitate/Drain/Zap numbers below are
    # the standard (non-ascension-scaled) values -- the wiki didn't surface
    # an ascension-specific damage tier for this monster beyond HP.
    def __init__(self, rng: Optional[random.Random] = None):
        rng = rng or random.Random()
        super().__init__("Chosen", max_hp=rng.randint(98, 103))
        self.hex_active = False

    def intent_options(self) -> List[Tuple[float, str]]:
        if self.turn_count == 0:
            return [(1.0, self.POKE)]
        if self.turn_count == 1:
            return [(1.0, self.HEX)]
        if (self.turn_count - 2) % 2 == 0:
            return [(0.5, self.DEBILITATE), (0.5, self.DRAIN)]
        return [(0.6, self.POKE), (0.4, self.ZAP)]

    def force_intent(self, move: str) -> None:
        if move == self.POKE:
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(5), move)  # hits 2x
        elif move == self.HEX:
            self.intent = Intent(IntentType.BUFF, None, move)
        elif move == self.DEBILITATE:
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(10), move)
        elif move == self.DRAIN:
            self.intent = Intent(IntentType.BUFF, None, move)
        else:  # ZAP
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(18), move)
        self._pending_move = move

    def take_turn(self, combat) -> None:
        move = self._pending_move
        if move == self.POKE:
            combat.deal_attack_damage(self, combat.player, 5)
            combat.deal_attack_damage(self, combat.player, 5)
        elif move == self.HEX:
            self.hex_active = True
        elif move == self.DEBILITATE:
            combat.deal_attack_damage(self, combat.player, 10)
            combat.player.add_power(Vulnerable(2))
        elif move == self.DRAIN:
            combat.player.add_power(Weak(3))
            self.add_power(Strength(3))
        else:  # ZAP
            combat.deal_attack_damage(self, combat.player, 18)
        self.turn_count += 1


class BookOfStabbing(Monster):
    """Act 2 elite. 85% Multi Stab / 15% Big Stab, can't repeat Multi Stab
    3x running or Big Stab 2x running. Multi Stab hits for a fixed 7 damage
    per hit, N times, where N = (times Multi Stab has been used this combat)
    + 2 -- so it starts at 2 hits and grows by one hit every time it's used
    again, the "gets more dangerous over time" mechanic this monster exists
    to test."""

    MULTI_STAB = "Multi Stab"
    BIG_STAB = "Big Stab"

    # A20 (Ascension 8+): HP 168-172 (randomized per-instance). Multi Stab
    # 6->7 dmg/hit (Ascension 3+). Big Stab 21->24 (Ascension 3+).
    def __init__(self, rng: Optional[random.Random] = None):
        rng = rng or random.Random()
        super().__init__("Book of Stabbing", max_hp=rng.randint(168, 172))
        self._multi_stab_uses = 0
        self._streak_move: Optional[str] = None
        self._streak_count = 0

    def intent_options(self) -> List[Tuple[float, str]]:
        # "3 times in a row" = 2 consecutive already happened is the last
        # legal state, so block *before* what would be the 3rd (streak>=2,
        # not >=3 -- that off-by-one let a real 3rd-in-a-row through, caught
        # by a smoke test rather than a guess). Same reasoning for Big
        # Stab's "twice in a row" cap: block after just 1 (streak>=1).
        if self._streak_move == self.MULTI_STAB and self._streak_count >= 2:
            return [(1.0, self.BIG_STAB)]
        if self._streak_move == self.BIG_STAB and self._streak_count >= 1:
            return [(1.0, self.MULTI_STAB)]
        return [(0.85, self.MULTI_STAB), (0.15, self.BIG_STAB)]

    def force_intent(self, move: str) -> None:
        if move == self.MULTI_STAB:
            hits = self._multi_stab_uses + 2
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(7), move)  # hits `hits` times
        else:  # BIG_STAB
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(24), move)
        if move == self._streak_move:
            self._streak_count += 1
        else:
            self._streak_move = move
            self._streak_count = 1
        self._pending_move = move

    def take_turn(self, combat) -> None:
        move = self._pending_move
        if move == self.MULTI_STAB:
            hits = self._multi_stab_uses + 2
            for _ in range(hits):
                combat.deal_attack_damage(self, combat.player, 7)
            self._multi_stab_uses += 1
        else:  # BIG_STAB
            combat.deal_attack_damage(self, combat.player, 24)
        self.turn_count += 1


# --- Act 3 roster. Same sourcing/approximation standard as the Act 2 block
# above: HP totals and move-damage/cadence read off direct sts_lightspeed
# construction + end-turn-only traces (real, not guessed); the underlying
# AI-state logic that produces the cadence is approximated where a short
# trace couldn't pin down an exact rule, same as every stochastic monster
# already in this file. Time Eater's turn-skip-on-too-many-cards-played
# gimmick (Time Warp) IS modeled -- see TimeEater.time_warp_counter and its
# check in CombatState.play_card. One mechanic is still deliberately NOT
# modeled: Nemesis's Intangible/burn-based revive (real effect unclear
# without a longer capture), which falls back to a plain stochastic move
# table instead, flagged here rather than silently pretending to be exact.
class Darkling(Monster):
    """One of a trio (see encounter_three_darklings). Each acts
    independently: mostly Nip (attack), sometimes Harden (block, no
    damage)."""

    # A20: HP default 50->55 (real trio rolls 51-58, per-instance variance
    # not modeled), Nip 8->11 (real per-instance range ~9-15, using the
    # rounded center same as the HP simplification above).
    def __init__(self, hp: int = 55):
        super().__init__("Darkling", max_hp=hp)

    def intent_options(self) -> List[Tuple[float, str]]:
        return [(0.6, "Nip"), (0.4, "Harden")]

    def force_intent(self, move: str) -> None:
        if move == "Nip":
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(11), move)
        else:
            self.intent = Intent(IntentType.DEFEND, None, move)
        self._pending_move = move

    def take_turn(self, combat) -> None:
        if self._pending_move == "Nip":
            combat.deal_attack_damage(self, combat.player, 11)
        else:
            self.gain_block(12)


class OrbWalker(Monster):
    """Act 3 basic (despite the boss-scale 95 HP/ramping Strength -- that's
    how the real encounter tier is categorized). Attacks every turn,
    alternating Claw/Laser, and gains +3 Strength every turn regardless of
    which move -- unlike Automaton's Boost, there's no dedicated non-attack
    buff turn here, the ramp is passive."""

    CLAW = "Claw"
    LASER = "Laser"

    # A20: HP 95->98, Claw 10->11, Laser 15->16, per-turn Strength gain 3->5.
    def __init__(self):
        super().__init__("Orb Walker", max_hp=98)

    def intent_options(self) -> List[Tuple[float, str]]:
        return [(1.0, self.LASER if self.turn_count % 2 == 1 else self.CLAW)]

    def force_intent(self, move: str) -> None:
        base = 16 if move == self.LASER else 11
        self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(base), move)
        self._pending_move = move

    def take_turn(self, combat) -> None:
        combat.deal_attack_damage(self, combat.player, 16 if self._pending_move == self.LASER else 11)
        self.add_power(Strength(5))
        self.turn_count += 1


class WrithingMass(Monster):
    """Act 3 basic. Stochastic move table (no clean deterministic cycle in
    the trace): Strong Left (big single hit), Flail (medium hit), Weak Left
    (3-hit), Implant (no damage)."""

    # A20: HP 160->175, Strong Left 32->38, Flail 15->16, Weak Left 7->9/hit.
    def __init__(self):
        super().__init__("Writhing Mass", max_hp=175)

    def intent_options(self) -> List[Tuple[float, str]]:
        return [(0.20, "Strong Left"), (0.40, "Flail"), (0.15, "Weak Left"), (0.25, "Implant")]

    def force_intent(self, move: str) -> None:
        if move == "Strong Left":
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(38), move)
        elif move == "Flail":
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(16), move)
        elif move == "Weak Left":
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(9), move)
        else:  # Implant
            self.intent = Intent(IntentType.DEFEND, None, move)
        self._pending_move = move

    def take_turn(self, combat) -> None:
        move = self._pending_move
        if move == "Strong Left":
            combat.deal_attack_damage(self, combat.player, 38)
        elif move == "Flail":
            combat.deal_attack_damage(self, combat.player, 16)
        elif move == "Weak Left":
            for _ in range(3):
                combat.deal_attack_damage(self, combat.player, 9)
        else:  # Implant
            self.gain_block(10)


class Spiker(Monster):
    """One of the "shapes" trash mobs (see encounter_three/four_shapes).
    Mostly passive, occasionally attacks."""

    # A20: HP 47->55, Spike 7->9.
    def __init__(self):
        super().__init__("Spiker", max_hp=55)

    def intent_options(self) -> List[Tuple[float, str]]:
        return [(0.3, "Spike"), (0.7, "Harden")]

    def force_intent(self, move: str) -> None:
        if move == "Spike":
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(9), move)
        else:
            self.intent = Intent(IntentType.DEFEND, None, move)
        self._pending_move = move

    def take_turn(self, combat) -> None:
        if self._pending_move == "Spike":
            combat.deal_attack_damage(self, combat.player, 9)
        else:
            self.gain_block(8)


class Repulsor(Monster):
    """One of the "shapes" trash mobs. Sparse attacker, mostly buffs/idles."""

    # A20: HP 34->36, Snap 11->13.
    def __init__(self, hp: int = 36):
        super().__init__("Repulsor", max_hp=hp)

    def intent_options(self) -> List[Tuple[float, str]]:
        return [(0.25, "Snap"), (0.75, "Buff")]

    def force_intent(self, move: str) -> None:
        if move == "Snap":
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(13), move)
        else:
            self.intent = Intent(IntentType.BUFF, None, move)
        self._pending_move = move

    def take_turn(self, combat) -> None:
        if self._pending_move == "Snap":
            combat.deal_attack_damage(self, combat.player, 13)
        else:
            self.add_power(Strength(1))


class Exploder(Monster):
    """One of the "shapes" trash mobs. Small hits for its first 2 turns,
    then a big self-destructing Explode that kills it -- matches the real
    trace exactly (9 dmg twice, then 30 dmg and gone)."""

    # A20: HP 30->35, Slam 9->11 (Explode's 30 confirmed unchanged).
    def __init__(self):
        super().__init__("Exploder", max_hp=35)

    def intent_options(self) -> List[Tuple[float, str]]:
        return [(1.0, "Explode" if self.turn_count >= 2 else "Slam")]

    def force_intent(self, move: str) -> None:
        self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(30 if move == "Explode" else 11), move)
        self._pending_move = move

    def take_turn(self, combat) -> None:
        if self._pending_move == "Explode":
            combat.deal_attack_damage(self, combat.player, 30)
            self.hp = 0
        else:
            combat.deal_attack_damage(self, combat.player, 11)
            self.turn_count += 1


class SphericGuardian(Monster):
    """Act 3 elite (also appears as the 3rd monster in
    encounter_sphere_and_two_shapes). Deceptively low 20 HP -- its real
    durability is the big block it opens with, not HP. Opens with Awaken
    (big block), then alternates Slam (1 hit) / Fierce Bash (2 hits)."""

    # A20: HP confirmed unchanged at 20; Slam/Fierce Bash 10->11 per hit.
    def __init__(self):
        super().__init__("Spheric Guardian", max_hp=20)

    def intent_options(self) -> List[Tuple[float, str]]:
        if self.turn_count == 0:
            return [(1.0, "Awaken")]
        return [(1.0, "Fierce Bash" if self.turn_count % 2 == 0 else "Slam")]

    def force_intent(self, move: str) -> None:
        if move == "Awaken":
            self.intent = Intent(IntentType.DEFEND, None, move)
        else:
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(11), move)
        self._pending_move = move

    def take_turn(self, combat) -> None:
        move = self._pending_move
        if move == "Awaken":
            self.gain_block(40)
        elif move == "Fierce Bash":
            combat.deal_attack_damage(self, combat.player, 11)
            combat.deal_attack_damage(self, combat.player, 11)
        else:  # Slam
            combat.deal_attack_damage(self, combat.player, 11)
        self.turn_count += 1


class Nemesis(Monster):
    """Act 3 elite. Stochastic move table -- Scythe (big single hit), Rake
    (3-hit), Intangible (buff, no damage; the real monster's Burn-based
    revive isn't modeled, see the block comment above this section)."""

    # A20: HP 185->200, Scythe unchanged at 45, Rake 6->7/hit.
    def __init__(self):
        super().__init__("Nemesis", max_hp=200)

    def intent_options(self) -> List[Tuple[float, str]]:
        return [(0.25, "Scythe"), (0.40, "Rake"), (0.35, "Intangible")]

    def force_intent(self, move: str) -> None:
        if move == "Scythe":
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(45), move)
        elif move == "Rake":
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(7), move)
        else:
            self.intent = Intent(IntentType.BUFF, None, move)
        self._pending_move = move

    def take_turn(self, combat) -> None:
        move = self._pending_move
        if move == "Scythe":
            combat.deal_attack_damage(self, combat.player, 45)
        elif move == "Rake":
            for _ in range(3):
                combat.deal_attack_damage(self, combat.player, 7)


class Dagger(Monster):
    """Reptomancer's persistent minion (see encounter_reptomancer). Mostly
    Stab (small hit, survives); sometimes Dagger Throw, a bigger hit that
    consumes/kills the dagger itself -- matches the real trace (a dagger
    hitting for 25 then disappearing).

    A20: HP/Stab/Dagger Throw all confirmed UNCHANGED from A0 -- one of the
    few monsters in the roster where nothing scales."""

    def __init__(self, hp: int = 22):
        super().__init__("Dagger", max_hp=hp)

    def intent_options(self) -> List[Tuple[float, str]]:
        return [(0.7, "Stab"), (0.3, "Dagger Throw")]

    def force_intent(self, move: str) -> None:
        self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(25 if move == "Dagger Throw" else 9), move)
        self._pending_move = move

    def take_turn(self, combat) -> None:
        if self._pending_move == "Dagger Throw":
            combat.deal_attack_damage(self, combat.player, 25)
            self.hp = 0
        else:
            combat.deal_attack_damage(self, combat.player, 9)


class Reptomancer(Monster):
    """Act 3 elite. Keeps up to 2 Daggers alive, re-summoning (Preparation)
    whenever it has fewer than 2; otherwise alternates Skewer (big single
    hit) / Flurry of Daggers (2-hit). Same escort-cache pattern as
    GremlinLeader (see its docstring for why intent_options() needs a
    take_turn-written cache rather than combat access of its own)."""

    SKEWER = "Skewer"
    FLURRY = "Flurry of Daggers"
    PREPARATION = "Preparation"

    # A20: HP 181->191, Skewer 30->34, Flurry 13->16/hit.
    def __init__(self):
        super().__init__("Reptomancer", max_hp=191)
        self._daggers_full = True

    def _daggers(self, combat) -> List[Monster]:
        return [m for m in combat.monsters if isinstance(m, Dagger)]

    def intent_options(self) -> List[Tuple[float, str]]:
        if not self._daggers_full:
            return [(1.0, self.PREPARATION)]
        return [(1.0, self.FLURRY if self.turn_count % 2 == 0 else self.SKEWER)]

    def force_intent(self, move: str) -> None:
        if move == self.SKEWER:
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(34), move)
        elif move == self.FLURRY:
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(16), move)
        else:
            self.intent = Intent(IntentType.BUFF, None, move)
        self._pending_move = move

    def take_turn(self, combat) -> None:
        move = self._pending_move
        if move == self.SKEWER:
            combat.deal_attack_damage(self, combat.player, 34)
        elif move == self.FLURRY:
            combat.deal_attack_damage(self, combat.player, 16)
            combat.deal_attack_damage(self, combat.player, 16)
        else:  # PREPARATION
            living = [d for d in self._daggers(combat) if not d.is_dead]
            for _ in range(2 - len(living)):
                new_dagger = Dagger()
                combat.monsters.append(new_dagger)
                new_dagger.roll_intent(combat.rng)
        self._daggers_full = len([d for d in self._daggers(combat) if not d.is_dead]) >= 2
        self.turn_count += 1


class AwakenedOneCultist(Monster):
    """One of Awakened One's 2 starting escorts. Idle turn 1, then Dark
    Strike every turn after while also self-buffing Strength every turn --
    distinct from the plain Cultist class (that one never ramps Strength),
    confirmed by the trace showing str increasing every single turn."""

    # A20: HP 54->56, Dark Strike unchanged at 6, per-turn Strength gain 3->5.
    def __init__(self):
        super().__init__("Cultist", max_hp=56)

    def intent_options(self) -> List[Tuple[float, str]]:
        return [(1.0, "Dark Strike" if self.turn_count > 0 else "Incantation")]

    def force_intent(self, move: str) -> None:
        if move == "Incantation":
            self.intent = Intent(IntentType.BUFF, None, move)
        else:
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(6), move)
        self._pending_move = move

    def take_turn(self, combat) -> None:
        if self._pending_move == "Dark Strike":
            combat.deal_attack_damage(self, combat.player, 6)
            self.add_power(Strength(5))
        self.turn_count += 1


class AwakenedOne(Monster):
    """Act 3 boss. Alternates Slam (1 hit) / Soul Strike (4-hit). Revives
    once at 0 HP instead of dying (real mechanic: transitions to a "Dark
    Phase" instead of dying; simplified here to reviving at half max HP with
    the same moveset rather than modeling a distinct second-phase kit)."""

    SLAM = "Slam"
    SOUL_STRIKE = "Soul Strike"

    # A20: HP 300->304. Slam/Soul Strike's damage confirmed UNCHANGED at
    # 20/6 per hit -- a live trace showed a constant str=2 reading
    # throughout the fight, but adding that as an actual Strength power
    # here would double-count on top of these already-correct observed
    # totals (they're whatever the real game computed, str included), not
    # a base value waiting to be buffed -- so it's noted, not modeled.
    def __init__(self):
        super().__init__("Awakened One", max_hp=304)
        self.revived = False

    def take_damage(self, incoming: int) -> int:
        hp_loss = super().take_damage(incoming)
        if self.hp <= 0 and not self.revived:
            self.revived = True
            self.hp = self.max_hp // 2
        return hp_loss

    def intent_options(self) -> List[Tuple[float, str]]:
        return [(1.0, self.SOUL_STRIKE if self.turn_count % 2 == 1 else self.SLAM)]

    def force_intent(self, move: str) -> None:
        base = 6 if move == self.SOUL_STRIKE else 20
        self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(base), move)
        self._pending_move = move

    def take_turn(self, combat) -> None:
        if self._pending_move == self.SOUL_STRIKE:
            for _ in range(4):
                combat.deal_attack_damage(self, combat.player, 6)
        else:
            combat.deal_attack_damage(self, combat.player, 20)
        self.turn_count += 1


class TimeEater(Monster):
    """Act 3 boss. Stochastic move table -- Ravage (big single hit), Rip
    and Tear (3-hit), Haste (buff, no damage). Also has Time Warp: whenever
    the player plays a card (any type, any turn -- the counter carries
    over between turns rather than resetting each one), after the 12th
    such card this monster gains 2 Strength and the player's turn ends
    immediately, before any further card can be played. See
    time_warp_counter below and its check in CombatState.play_card."""

    # A20: HP 456->480, Ravage 26->32, Rip and Tear 7->8/hit.
    def __init__(self):
        super().__init__("Time Eater", max_hp=480)
        self.time_warp_counter = 0

    def intent_options(self) -> List[Tuple[float, str]]:
        return [(0.25, "Ravage"), (0.50, "Rip and Tear"), (0.25, "Haste")]

    def force_intent(self, move: str) -> None:
        if move == "Ravage":
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(32), move)
        elif move == "Rip and Tear":
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(8), move)
        else:
            self.intent = Intent(IntentType.BUFF, None, move)
        self._pending_move = move

    def take_turn(self, combat) -> None:
        move = self._pending_move
        if move == "Ravage":
            combat.deal_attack_damage(self, combat.player, 32)
        elif move == "Rip and Tear":
            for _ in range(3):
                combat.deal_attack_damage(self, combat.player, 8)


class DonuDeca(Monster):
    """Donu and Deca (see encounter_donu_and_deca) -- fought as a pair, each
    an instance of this same class. Alternates Attack / Buff, offset from
    its sibling (whoever attacked last turn buffs this turn, and vice
    versa); Buff grants Strength to both itself and its sibling, same
    ally-lookup pattern as Mystic."""

    # A20: HP 250->265, Beam 10->12/hit (Buff's Strength grant confirmed
    # unchanged at 3).
    def __init__(self, starts_attacking: bool):
        super().__init__("Donu/Deca", max_hp=265)
        self._starts_attacking = starts_attacking

    def intent_options(self) -> List[Tuple[float, str]]:
        attacking = (self.turn_count % 2 == 0) == self._starts_attacking
        return [(1.0, "Beam" if attacking else "Buff")]

    def force_intent(self, move: str) -> None:
        if move == "Beam":
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(12), move)  # hits twice
        else:
            self.intent = Intent(IntentType.BUFF, None, move)
        self._pending_move = move

    def _sibling(self, combat):
        for m in combat.living_monsters:
            if m is not self and isinstance(m, DonuDeca):
                return m
        return None

    def take_turn(self, combat) -> None:
        if self._pending_move == "Beam":
            combat.deal_attack_damage(self, combat.player, 12)
            combat.deal_attack_damage(self, combat.player, 12)
        else:
            self.add_power(Strength(3))
            sibling = self._sibling(combat)
            if sibling is not None:
                sibling.add_power(Strength(3))
        self.turn_count += 1


class GiantHead(Monster):
    """Act 3 unique encounter (a single-monster fight, not a normal elite).
    On higher ascensions, the first 3 turns alternate Count/Glare 50/50
    (never the same move 3x running); after that it repeats It Is Time
    forever, whose damage climbs 5 per prior use up to a cap of 70."""

    COUNT = "Count"
    GLARE = "Glare"
    IT_IS_TIME = "It Is Time"

    # A20 (Ascension 8+): HP 520 (fixed, not randomized). It Is Time starts
    # at 40 (Ascension 3+, base is lower) and climbs 5/use, capped at 70.
    def __init__(self):
        super().__init__("Giant Head", max_hp=520)
        self._it_is_time_uses = 0
        self._streak_move: Optional[str] = None
        self._streak_count = 0

    def intent_options(self) -> List[Tuple[float, str]]:
        if self.turn_count >= 3:
            return [(1.0, self.IT_IS_TIME)]
        if self._streak_move == self.COUNT and self._streak_count >= 2:
            return [(1.0, self.GLARE)]
        if self._streak_move == self.GLARE and self._streak_count >= 2:
            return [(1.0, self.COUNT)]
        return [(0.5, self.COUNT), (0.5, self.GLARE)]

    def force_intent(self, move: str) -> None:
        if move == self.COUNT:
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(13), move)
        elif move == self.GLARE:
            self.intent = Intent(IntentType.DEFEND, None, move)
        else:  # IT_IS_TIME
            dmg = min(70, 40 + 5 * self._it_is_time_uses)
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(dmg), move)
        if move == self._streak_move:
            self._streak_count += 1
        else:
            self._streak_move = move
            self._streak_count = 1
        self._pending_move = move

    def take_turn(self, combat) -> None:
        move = self._pending_move
        if move == self.COUNT:
            combat.deal_attack_damage(self, combat.player, 13)
        elif move == self.GLARE:
            combat.player.add_power(Weak(1))
        else:  # IT_IS_TIME
            dmg = min(70, 40 + 5 * self._it_is_time_uses)
            combat.deal_attack_damage(self, combat.player, dmg)
            self._it_is_time_uses += 1
        self.turn_count += 1


class ShelledParasite(Monster):
    """Act 3 monster. At Ascension 17+ always opens with Fell (overriding
    the base-game "cannot use Fell on turn 1" restriction); afterward, 40%
    Double Strike / 40% Life Suck / 20% Fell, never Double Strike or Life
    Suck 3x running, never Fell 2x running."""

    DOUBLE_STRIKE = "Double Strike"
    LIFE_SUCK = "Life Suck"
    FELL = "Fell"

    # A20 (Ascension 7+): HP 70-75 (randomized per-instance). Numbers below
    # are the Ascension 2+ tier (the wiki didn't surface a higher-ascension
    # damage tier beyond that for this monster).
    def __init__(self, rng: Optional[random.Random] = None):
        rng = rng or random.Random()
        super().__init__("Shelled Parasite", max_hp=rng.randint(70, 75))
        self._streak_move: Optional[str] = None
        self._streak_count = 0

    def intent_options(self) -> List[Tuple[float, str]]:
        if self.turn_count == 0:
            return [(1.0, self.FELL)]
        # Same off-by-one fix as Book of Stabbing above: "3 times in a row"
        # blocks at streak>=2 (the would-be 3rd), "twice in a row" blocks at
        # streak>=1 (the would-be 2nd) -- not >=3/>=2 respectively.
        if self._streak_move in (self.DOUBLE_STRIKE, self.LIFE_SUCK) and self._streak_count >= 2:
            others = [m for m in (self.DOUBLE_STRIKE, self.LIFE_SUCK, self.FELL) if m != self._streak_move]
            return [(0.5, others[0]), (0.5, others[1])]
        if self._streak_move == self.FELL and self._streak_count >= 1:
            return [(0.5, self.DOUBLE_STRIKE), (0.5, self.LIFE_SUCK)]
        return [(0.4, self.DOUBLE_STRIKE), (0.4, self.LIFE_SUCK), (0.2, self.FELL)]

    def force_intent(self, move: str) -> None:
        if move == self.DOUBLE_STRIKE:
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(7), move)  # hits 2x
        elif move == self.LIFE_SUCK:
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(12), move)
        else:  # FELL
            self.intent = Intent(IntentType.ATTACK, self.calc_attack_damage(21), move)
        if move == self._streak_move:
            self._streak_count += 1
        else:
            self._streak_move = move
            self._streak_count = 1
        self._pending_move = move

    def take_turn(self, combat) -> None:
        move = self._pending_move
        if move == self.DOUBLE_STRIKE:
            combat.deal_attack_damage(self, combat.player, 7)
            combat.deal_attack_damage(self, combat.player, 7)
        elif move == self.LIFE_SUCK:
            healed = combat.deal_attack_damage(self, combat.player, 12)
            self.hp = min(self.max_hp, self.hp + healed)
        else:  # FELL
            combat.deal_attack_damage(self, combat.player, 21)
            combat.player.add_power(Frail(2))
        self.turn_count += 1
