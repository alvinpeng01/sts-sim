"""A Slay the Spire (StS1) combat simulator.

Foundation for a combat AI: an expectimax/search-based fight solver on top,
and later a run-level RL meta-policy. This package is just the fight engine.
"""

from .combat import CombatState, Result
from .creatures import Player
from .enemies import JawWorm, Cultist, Louse
from .cards import ironclad_starter_deck, varied_ironclad_deck, common_card_pool
from .encounters import encounter_jaw_worm, encounter_cultist, encounter_louse_pair
from .search import choose_action, evaluate

__all__ = [
    "CombatState", "Result", "Player",
    "JawWorm", "Cultist", "Louse",
    "ironclad_starter_deck", "varied_ironclad_deck", "common_card_pool",
    "encounter_jaw_worm", "encounter_cultist", "encounter_louse_pair",
    "choose_action", "evaluate",
]
