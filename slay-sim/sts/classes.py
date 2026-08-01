"""Per-class starter decks, card pools, and starting HP -- the class-specific
content layer. The engine/cards/powers underneath are entirely class-agnostic
(a Card is just cost+effect, a Power is just a hook); this module is the only
place that says which cards belong to which class."""

from __future__ import annotations

from enum import Enum
from typing import Callable, Dict, List

from .cards import (
    Card,
    ironclad_starter_deck, common_card_pool,
    silent_starter_deck, silent_card_pool,
    defect_starter_deck, defect_card_pool,
    watcher_starter_deck, watcher_card_pool,
)
from .creatures import Player


class CharClass(Enum):
    IRONCLAD = "Ironclad"
    SILENT = "Silent"
    DEFECT = "Defect"
    WATCHER = "Watcher"


STARTING_HP: Dict[CharClass, int] = {
    CharClass.IRONCLAD: 80,
    CharClass.SILENT: 70,
    CharClass.DEFECT: 75,
    CharClass.WATCHER: 72,
}

STARTER_DECK: Dict[CharClass, Callable[[], List[Card]]] = {
    CharClass.IRONCLAD: ironclad_starter_deck,
    CharClass.SILENT: silent_starter_deck,
    CharClass.DEFECT: defect_starter_deck,
    CharClass.WATCHER: watcher_starter_deck,
}

CARD_POOL: Dict[CharClass, Callable[[], List[Card]]] = {
    CharClass.IRONCLAD: common_card_pool,
    CharClass.SILENT: silent_card_pool,
    CharClass.DEFECT: defect_card_pool,
    CharClass.WATCHER: watcher_card_pool,
}


def make_player(char_class: CharClass, max_energy: int = 3) -> Player:
    return Player(name=char_class.value, max_hp=STARTING_HP[char_class], max_energy=max_energy)


def varied_deck(char_class: CharClass, pool_copies: int = 1) -> List[Card]:
    """Starter deck plus ``pool_copies`` of every card in that class's pool."""
    deck = STARTER_DECK[char_class]()
    for _ in range(pool_copies):
        deck += CARD_POOL[char_class]()
    return deck
