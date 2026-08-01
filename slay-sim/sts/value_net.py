"""A learned leaf-evaluation value function for the expectimax/MCTS search.

Why this exists: search.py's hand-written evaluate() scores a state as
`hp*2 - enemy_hp*0.5 + block*0.5` -- it has NO term for powers, so at
turns_left=1 (the only depth fast enough for live play) any buff whose payoff
lands on a *future* turn is invisible. Banked Strength, Poison ticking over
several enemy turns, Vulnerable on an enemy you're not finishing this turn --
all score identically to not having them. Deep search (turns_left=2+) *does*
see them, but blows up to ~26s/decision on a 3-monster fight (measured), so
it's a non-starter for a live tool.

This module replaces the hand-written terms with a function LEARNED from
actual fight outcomes. The key property, and the reason it was chosen over
adding hand-tuned power terms: it generalizes to any character for free. The
encoder reads powers by a fixed vocabulary (all 21 that exist today, plus an
"other" bucket), so when Silent's Poison decks or Defect's Focus arrive, the
value of their buffs is picked up by re-training on data that includes them,
with ZERO new hand-weights and no edits to evaluate(). "How much is 1
Strength worth in HP?" stops being a number a human picks and becomes
something regressed from how fights with banked Strength actually end.

Split by concern:
  * encode_state()  -- CombatState -> fixed-size np.float32 vector, power-aware
  * ValueNet        -- pure-numpy forward pass (fast per-call inference in the
                       search hot path; no torch import on this path)
  * training lives in sts/train_value.py (torch), which exports weights to a
    .npz this module loads.
"""

from __future__ import annotations

from typing import Optional

# numpy is imported lazily inside the functions/class that need it (encoding
# and inference), NOT at module top -- so `from . import value_net` in
# search.py keeps the core search path dependency-free for the plain-python3
# users who never load a net. numpy is only touched once someone actually
# trains/loads one (which happens in the .venv that has it).

from .combat import CombatState, Result

# Fixed power vocabulary -- every Power subclass in powers.py today. Order is
# the encoding order and MUST stay stable for a trained net to keep meaning;
# append-only if extended (never reorder/insert), and anything not listed
# falls into the trailing "other" bucket so an unknown power never changes
# the vector's SIZE, only retraining changes what the net does with it.
POWER_VOCAB = [
    "Strength", "Dexterity", "Vulnerable", "Weak", "Frail", "Ritual",
    "Poison", "Focus", "Combust", "DarkEmbrace", "FeelNoPain", "Evolve",
    "Metallicize", "Rage", "FireBreathing", "Rupture", "Barricade",
    "DemonForm", "BerserkEnergy", "Brutality", "Corruption",
]
_POWER_INDEX = {name: i for i, name in enumerate(POWER_VOCAB)}
_N_POWERS = len(POWER_VOCAB)

# Feature layout (see below): 6 player scalars + powers(+other) player side
# + 3 enemy scalars + powers(+other) enemy side.
_PLAYER_SCALARS = 6
_ENEMY_SCALARS = 3
STATE_DIM = _PLAYER_SCALARS + (_N_POWERS + 1) + _ENEMY_SCALARS + (_N_POWERS + 1)


def _power_vector(creature):
    """A (_N_POWERS + 1) vector of this creature's power amounts, scaled;
    the trailing slot accumulates any power outside POWER_VOCAB so an
    unrecognized status still contributes signal instead of vanishing."""
    import numpy as np
    vec = np.zeros(_N_POWERS + 1, dtype=np.float32)
    for name, power in creature.powers.items():
        idx = _POWER_INDEX.get(name)
        if idx is None:
            vec[_N_POWERS] += power.amount / 10.0
        else:
            vec[idx] = power.amount / 10.0
    return vec


def _incoming_damage(combat: CombatState) -> float:
    """Telegraphed HP damage the living monsters intend next, summed. This
    is the same 'what's coming at me' signal the hand eval lacks a term for;
    the net can learn to weigh it against the player's block/hp itself."""
    total = 0
    for m in combat.living_monsters:
        intent = m.intent
        if intent is not None and intent.damage:
            total += intent.damage
    return total


def encode_state(combat: CombatState):
    """CombatState -> fixed-size feature vector. Enemy features are
    AGGREGATED (totals across living monsters), not per-slot, so the vector
    is a constant size for 1-enemy and 5-enemy fights alike -- the search
    calls this at every leaf and needs one stable shape."""
    import numpy as np
    p = combat.player
    living = combat.living_monsters

    player_scalars = np.array([
        p.hp / max(p.max_hp, 1),
        p.block / 50.0,
        p.energy / max(p.max_energy, 1),
        len(combat.hand) / 10.0,
        (len(combat.draw_pile) + len(combat.discard_pile) + len(combat.exhaust_pile)) / 30.0,
        combat.turn / 20.0,
    ], dtype=np.float32)

    total_hp = sum(m.hp for m in living)
    total_max = sum(m.max_hp for m in living) or 1
    enemy_scalars = np.array([
        total_hp / total_max,
        len(living) / 3.0,
        _incoming_damage(combat) / 30.0,
    ], dtype=np.float32)

    enemy_powers = np.zeros(_N_POWERS + 1, dtype=np.float32)
    for m in living:
        enemy_powers += _power_vector(m)

    return np.concatenate([
        player_scalars, _power_vector(p), enemy_scalars, enemy_powers,
    ])


class ValueNet:
    """Pure-numpy forward pass of the MLP trained in train_value.py. Kept
    numpy (not torch) specifically because the search calls it one state at a
    time, thousands of times per decision -- torch's per-call Python/dispatch
    overhead dominates at that granularity, while a hand-rolled 3-layer
    matmul is trivially fast and adds no torch dependency to the search path.
    Output is tanh in [-1, 1]; evaluate() rescales it into HP units."""

    def __init__(self, W1, b1, W2, b2, W3, b3):
        self.W1, self.b1 = W1, b1
        self.W2, self.b2 = W2, b2
        self.W3, self.b3 = W3, b3

    @classmethod
    def load(cls, path: str) -> "ValueNet":
        import numpy as np
        d = np.load(path)
        return cls(d["W1"], d["b1"], d["W2"], d["b2"], d["W3"], d["b3"])

    def __call__(self, x) -> float:
        import numpy as np
        h = np.maximum(0.0, x @ self.W1 + self.b1)
        h = np.maximum(0.0, h @ self.W2 + self.b2)
        out = np.tanh(h @ self.W3 + self.b3)
        return float(out[0])


# --- integration hook: evaluate() consults this if a net is loaded ---
_ACTIVE_NET: Optional[ValueNet] = None


def set_value_net(net: Optional[ValueNet]) -> None:
    """Install (or clear, with None) the value net that evaluate() uses for
    non-terminal states. Terminal states are never routed here -- their value
    is exact ground truth (win => hp carried, loss => the loss floor)."""
    global _ACTIVE_NET
    _ACTIVE_NET = net


def get_value_net() -> Optional[ValueNet]:
    return _ACTIVE_NET


def learned_value(combat: CombatState) -> float:
    """Non-terminal state value in HP units, from the active net. Rescales
    the net's tanh output by the player's max_hp so the number is on the same
    HP scale the hand eval uses (and, more importantly, so expectimax's
    argmax over sibling states is meaningful). Assumes a net is loaded --
    evaluate() guards that."""
    x = encode_state(combat)
    return _ACTIVE_NET(x) * combat.player.max_hp
