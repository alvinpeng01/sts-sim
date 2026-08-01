"""Drive the simulator with a couple of trivial policies.

Run:  python demo.py
Shows one narrated fight, then batches many fights to show speed + that the
engine reaches terminal states cleanly. These policies are placeholders --
the real expectimax fight solver replaces `policy` later.
"""

import random
import time

from sts.combat import CombatState, Result
from sts.creatures import Player
from sts.enemies import JawWorm
from sts.cards import ironclad_starter_deck


def greedy_policy(combat: CombatState):
    """A dumb baseline: Bash > Strike if unblocked incoming, else Defend/Strike.

    Blocks when a big hit is telegraphed, otherwise attacks. Good enough to be
    a debugging oracle, nowhere near optimal.
    """
    incoming = sum(
        m.intent.damage for m in combat.living_monsters
        if m.intent and m.intent.damage
    )
    def target():
        return max(combat.living_monsters, key=lambda m: -m.hp)  # lowest HP

    # If a real hit is coming and we have low block, defend.
    if incoming - combat.player.block >= 10:
        for c in combat.hand:
            if c.name == "Defend" and c.cost <= combat.player.energy:
                return ("play", c, None)
    # Otherwise attack: prefer Bash (applies Vulnerable) then Strike.
    for wanted in ("Bash", "Strike"):
        for c in combat.hand:
            if c.name == wanted and c.cost <= combat.player.energy:
                return ("play", c, target())
    # Fall back to any playable card, else end turn. Status cards (Wound,
    # Dazed, ...) can end up in hand via enemies like Sentry that shuffle
    # them into the draw pile -- c.playable=False excludes those, since
    # play_card() rejects them outright. Also must check extra_legal_check
    # (Clash: "only playable if every card in hand is an Attack") the same
    # way legal_actions()/play_card() do -- omitting it let this fallback
    # hand back a card play() would then reject, uncaught until a deck
    # actually included such a card.
    for c in combat.hand:
        if (c.playable and c.cost <= combat.player.energy
                and (c.extra_legal_check is None or c.extra_legal_check(combat))):
            return ("play", c, target() if c.targeted else None)
    return ("end",)


def play_fight(policy, seed=None, verbose=False):
    rng = random.Random(seed)
    player = Player(max_hp=80)
    combat = CombatState(player, [JawWorm()], ironclad_starter_deck(),
                         rng=rng, verbose=verbose)
    while combat.result() == Result.ONGOING:
        combat.start_player_turn()
        # Play cards until the policy ends the turn.
        while combat.result() == Result.ONGOING:
            action = policy(combat)
            if action[0] == "end":
                break
            combat.play_card(action[1], action[2])
            if combat.result() != Result.ONGOING:
                break
        if combat.result() != Result.ONGOING:
            break
        combat.end_player_turn()
        combat.enemy_turn()
    return combat


def main():
    print("### One narrated fight (greedy policy vs Jaw Worm) ###")
    combat = play_fight(greedy_policy, seed=7, verbose=True)
    res = combat.result()
    print(f"\nResult: {res.value.upper()} | player HP {combat.player.hp}/80\n")

    print("### Batch: 20000 fights, greedy policy ###")
    n = 20000
    wins = hp_total = 0
    t0 = time.perf_counter()
    for i in range(n):
        c = play_fight(greedy_policy, seed=i)
        if c.result() == Result.WIN:
            wins += 1
            hp_total += c.player.hp
    dt = time.perf_counter() - t0
    print(f"win rate: {wins/n:.1%}")
    print(f"avg HP remaining (wins): {hp_total/max(wins,1):.1f}")
    print(f"speed: {n/dt:,.0f} fights/sec ({dt:.2f}s total)")


if __name__ == "__main__":
    main()
