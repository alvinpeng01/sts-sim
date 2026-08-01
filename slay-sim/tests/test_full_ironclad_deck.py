"""Stress-test: every implemented Ironclad card, played for real across many
fights with a 30-card deck. This isn't testing tactical quality (a random
policy is deliberately dumb) -- it's testing that every card's effect
function runs without crashing and that the engine hooks (exhaust events,
ethereal cleanup, X-cost, status-card draws, orbs-unused-here, Corruption/
Barricade special cases) all compose correctly under real play."""

import random

from sts.combat import CombatState, Result
from sts.creatures import Player
from sts.cards import big_ironclad_deck, common_card_pool, ironclad_starter_deck
from sts.enemies import JawWorm, Cultist


def random_legal_policy(combat, rng):
    actions = combat.legal_actions()
    non_end = [a for a in actions if a[0] != "end"]
    if non_end and rng.random() < 0.85:
        return rng.choice(non_end)
    return ("end",)


def play_one_fight(deck_factory, monster_factory, seed):
    rng = random.Random(seed)
    player = Player(max_hp=9999)  # remove HP as a confound; we're testing crashes, not tactics
    combat = CombatState(player, [monster_factory()], deck_factory(), rng=rng)
    turns = 0
    while combat.result() == Result.ONGOING and turns < 200:
        combat.start_player_turn()
        while combat.result() == Result.ONGOING:
            action = random_legal_policy(combat, rng)
            if action[0] == "end":
                break
            combat.play_card(action[1], action[2])
        if combat.result() != Result.ONGOING:
            break
        combat.end_player_turn()
        combat.enemy_turn()
        turns += 1
    return combat, turns


def test_big_deck_runs_to_completion_many_seeds():
    for seed in range(60):
        monster_factory = JawWorm if seed % 2 == 0 else Cultist
        combat, turns = play_one_fight(big_ironclad_deck, monster_factory, seed)
        assert combat.result() == Result.WIN, (
            f"seed {seed}: fight should reach WIN with near-infinite player HP "
            f"(after {turns} turns) -- got {combat.result()}"
        )


def test_every_pool_card_playable_at_least_once_across_many_fights():
    """Weaker than exercising every card in one fight (a 5-card hand can't),
    but with enough independent fights every card should get drawn+played at
    least once. Confirms no card silently never triggers due to a
    construction bug (e.g. wrong targeted= flag hiding it from legal_actions)."""
    def full_pool_deck():
        return ironclad_starter_deck() + common_card_pool()

    played_names = set()
    orig_play_card = CombatState.play_card

    def tracking_play_card(self, card, target):
        played_names.add(card.name)
        return orig_play_card(self, card, target)

    CombatState.play_card = tracking_play_card
    try:
        for seed in range(150):
            monster_factory = JawWorm if seed % 2 == 0 else Cultist
            play_one_fight(full_pool_deck, monster_factory, seed)
    finally:
        CombatState.play_card = orig_play_card

    all_names = {c.name for c in full_pool_deck()}
    never_played = all_names - played_names
    assert not never_played, f"cards never drawn+played across 150 fights: {never_played}"


if __name__ == "__main__":
    import sys, traceback
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); passed += 1
            except Exception:
                failed += 1; print(f"FAIL {name}"); traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
