"""Damage prediction from CommunicationMod's own combat-state JSON.

Deliberately does NOT re-simulate monster AI or look further ahead than the
current turn. That's a scope choice, not a shortcut: our own sts/enemies.py
only implements a fraction of the real game's monster roster (Act 1 mostly,
a slice of Act 2), so reusing it here would mean "predictions work in the
same handful of fights we already modeled, silently wrong or missing
everywhere else" -- exactly the kind of gap that's dangerous to ship
unflagged in a tool meant to be trusted mid-run.

CommunicationMod sidesteps this entirely: by the time a state hits us, the
game has ALREADY rolled each monster's intent for this round and reports its
damage directly (`move_adjusted_damage` / `move_hits`), for any monster in
the real game, any act. Reading that number is correct by construction --
we're not predicting the monster's behavior, just doing arithmetic on what
the game already decided. The cost is horizon: this only answers "what
happens if I end my turn right now," not "what's my best line over the next
three turns" (that would need our own monster AI back, i.e. the same
coverage gap). A card-by-card what-if (playing a block card changes the
number before you end turn) is the natural next layer, but needs a verified
card-id mapping against a real captured state first -- see the TODO at the
bottom of this file -- so it's not included yet.
"""

from __future__ import annotations

from typing import Dict, List

# CommunicationMod's `monsters[].intent` is a string enum; these are the
# variants that deal HP damage to the player if unchanged. (DEBUFF/BUFF/
# DEFEND/ESCAPE/SLEEP/STUN/UNKNOWN/NONE all deal none.) Not independently
# verified against a live capture yet -- see module docstring/TODO.
DAMAGING_INTENTS = {
    "ATTACK", "ATTACK_BUFF", "ATTACK_DEBUFF", "ATTACK_DEFEND",
}


def monster_incoming_damage(monster: dict) -> int:
    """Damage this one monster deals to the player if its currently
    telegraphed intent resolves unchanged. 0 for a dead/gone monster, a
    non-damaging intent, or a monster whose damage hasn't been telegraphed
    yet (the very first state push of a fight, before intents are rolled)."""
    if monster.get("is_gone") or monster.get("current_hp", 0) <= 0:
        return 0
    if monster.get("intent") not in DAMAGING_INTENTS:
        return 0
    per_hit = monster.get("move_adjusted_damage")
    if per_hit is None or per_hit < 0:
        return 0
    hits = monster.get("move_hits") or 1
    return per_hit * hits


def net_incoming_damage(combat_state: dict) -> int:
    """HP the player would lose if they ended their turn right now, given
    every living monster's currently telegraphed intent and the player's
    current block."""
    monsters = combat_state.get("monsters", [])
    total = sum(monster_incoming_damage(m) for m in monsters)
    block = combat_state.get("player", {}).get("block", 0)
    return max(0, total - block)


def summarize(combat_state: dict) -> Dict[str, object]:
    """Human-readable prediction bundle for one state push -- what
    communication_mod.py logs each turn."""
    monsters = combat_state.get("monsters", [])
    per_monster = [
        {"name": m.get("name") or m.get("id"), "damage": monster_incoming_damage(m)}
        for m in monsters if not m.get("is_gone") and m.get("current_hp", 0) > 0
    ]
    player = combat_state.get("player", {})
    return {
        "turn": combat_state.get("turn"),
        "player_hp": player.get("current_hp"),
        "player_block": player.get("block"),
        "net_incoming_damage": net_incoming_damage(combat_state),
        "per_monster": per_monster,
    }


# TODO(next session, needs a live capture): verify DAMAGING_INTENTS and the
# move_adjusted_damage/move_hits field names against a real CommunicationMod
# state dump (run the bridge, let it write raw states to a log, inspect one
# mid-fight) before trusting this for real decisions -- these are written
# from best available knowledge of the mod's schema, not confirmed against
# a live sample. Once verified, the natural next layer is a card-by-card
# "what if I play this" delta using sts/cards.py's own block-granting logic
# for the player's hand, which needs a verified CommunicationMod-card-id ->
# our-card-name mapping (their ids don't all match our display names, e.g.
# likely "Strike_R"/"Defend_R" vs our "Strike"/"Defend") -- don't guess that
# table blind, confirm it against real hand data first.
