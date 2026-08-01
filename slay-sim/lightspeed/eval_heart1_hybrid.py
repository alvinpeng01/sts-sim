"""Evaluate Heart1's overworld policy with lightspeed's native combat MCTS.

The two engines have incompatible native modules, so this intentionally runs
entirely against ``sts_lightspeed``: Heart1 observes and acts on that GameContext,
while ``native_playout_current_battle`` owns every combat decision.
"""
from __future__ import annotations

import argparse
import json
import random
import time

import torch
import slaythespire as sts

from lightspeed.search_config import apply_search_config
from silverbot.network import ModelHP, NN, load_network_backward_compatible
from silverbot.playouts import NNService, construct_choice, choose_overworld_action, take_free_rewards


def play(seed: int, service: NNService, sims: int, ascension: int, temperature: float) -> dict:
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, ascension)
    rng = random.Random(seed)
    trace: list[dict] = []
    battles = 0
    started = time.perf_counter()
    while gc.outcome == sts.GameOutcome.UNDECIDED:
        if gc.screen_state == sts.ScreenState.BATTLE:
            before = (gc.floor_num, gc.cur_hp, gc.max_hp, int(gc.encounter))
            sts.native_playout_current_battle(gc, sims)
            battles += 1
            trace.append({"kind": "battle", "before": before, "after_hp": gc.cur_hp})
            continue
        take_free_rewards(gc)
        if gc.outcome != sts.GameOutcome.UNDECIDED:
            break
        actions = sts.GameAction.getAllActionsInState(gc)
        if not actions:
            raise RuntimeError(f"no legal overworld actions: state={gc.screen_state}")
        try:
            choice = construct_choice(gc, sts.getNNRepresentation(gc), actions)
        except ValueError:
            # lightspeed's legacy boss-relic action encoding differs from
            # the checkpoint's expected RELIC/SKIP form.  Keep the hybrid
            # run moving with a deterministic legal fallback on that one
            # screen; all ordinary overworld decisions still use Heart1.
            choice = None
        if choice is not None and (len(choice.cards_offered) + len(choice.relics_offered)
                                   + len(choice.potions_offered) + len(choice.fixed_actions)
                                   + len(choice.paths_offered) > 1):
            action, description, _, _, _, _ = choose_overworld_action(
                service, choice, gc, rng, temperature=temperature)
        else:
            action, description = actions[0], actions[0].getDesc(gc)
        if not action.isValidAction(gc):
            raise RuntimeError(f"Heart1 selected invalid action {description} in {gc.screen_state}")
        trace.append({"kind": "overworld", "floor": gc.floor_num, "screen": str(gc.screen_state),
                      "hp": gc.cur_hp, "action": description})
        action.execute(gc)
    return {"seed": seed, "outcome": str(gc.outcome), "floor": gc.floor_num, "act": gc.act,
            "hp": gc.cur_hp, "max_hp": gc.max_hp, "keys": int(gc.red_key) + int(gc.green_key) + int(gc.blue_key),
            "battles": battles, "elapsed_seconds": round(time.perf_counter() - started, 3), "trace": trace}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="../silverbot-reference/runs/heart1.pt")
    p.add_argument("--sims", type=int, default=10_000)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--seed-base", type=int, default=1_000_000)
    p.add_argument("--ascension", type=int, default=20)
    p.add_argument("--temperature", type=float, default=0.0, help="0 is deterministic argmax")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    with open("lightspeed/tuned_search_params.json", encoding="utf-8") as f:
        apply_search_config(json.load(f))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = NN(ModelHP(use_value_head=True, dim=256, n_layers=4)).to(device)
    net = load_network_backward_compatible(net, torch.load(args.ckpt, map_location=device, weights_only=True))
    net.eval()
    service = NNService(net, batch_size=8, batch_size_factor=1, torch_compile_mode="no")
    try:
        rows = [play(args.seed_base + i, service, args.sims, args.ascension, args.temperature)
                for i in range(args.runs)]
    finally:
        service.stop()
    for row in rows:
        print(json.dumps({k: v for k, v in row.items() if k != "trace"}), flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
