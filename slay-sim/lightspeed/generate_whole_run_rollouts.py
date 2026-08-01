"""Generate stratified, soft strategic labels from matched native rollouts.

Each row stores the exact variable-sized observation plus a probability
distribution over its legal actions.  Candidate actions are evaluated with
the same copied game state and the same policy-sampling seed, reducing noise
between candidates without using any external model or data.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
import random
import time

import numpy as np
import slaythespire as sts
import torch

from .whole_run_env import RunConfig, WholeRunEnv, partition_legal_actions
from .whole_run_transformer import WholeRunTransformerPolicy
from .whole_run_transformer_v27 import WholeRunTransformerPolicyV27


DECISION_TYPES = (
    "neow", "event", "map", "rewards", "shop",
    "rest", "boss_relic", "card_select", "treasure",
)


def decision_type(gc) -> str:
    screen = gc.screen_state
    if screen == sts.ScreenState.EVENT_SCREEN:
        return "neow" if gc.cur_event == sts.Event.NEOW else "event"
    if screen == sts.ScreenState.MAP_SCREEN:
        return "map"
    if screen == sts.ScreenState.REWARDS:
        return "rewards"
    if screen == sts.ScreenState.SHOP_ROOM:
        return "shop"
    if screen == sts.ScreenState.REST_ROOM:
        return "rest"
    if screen == sts.ScreenState.BOSS_RELIC_REWARDS:
        return "boss_relic"
    if screen == sts.ScreenState.CARD_SELECT:
        return "card_select"
    if screen == sts.ScreenState.TREASURE_ROOM:
        return "treasure"
    return "other"


def load_policy(path: str, device) -> WholeRunTransformerPolicy:
    state = torch.load(path, map_location=device, weights_only=True)
    policy_class = (
        WholeRunTransformerPolicyV27
        if any(key.startswith("decision_experts.") for key in state)
        else WholeRunTransformerPolicy
    )
    policy = policy_class().to(device)
    missing, unexpected = policy.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"compatible policy load new={missing} unused={unexpected}", flush=True)
    print(f"policy architecture={policy_class.__name__}", flush=True)
    return policy.eval()


def rollout_score(
    env: WholeRunEnv,
    start_floor: int,
    start_act: int,
    floor_weight: float,
    act_weight: float,
    loss_penalty: float,
    boss_progress_weight: float,
) -> float:
    gc = env.gc
    floor_gain = max(0, int(gc.floor_num) - start_floor)
    act_gain = max(0, int(gc.act) - start_act)
    hp_fraction = max(0, gc.cur_hp) / max(1, gc.max_hp)
    score = floor_weight * floor_gain + act_weight * act_gain + 0.12 * hp_fraction
    if gc.outcome == sts.GameOutcome.PLAYER_LOSS:
        score -= loss_penalty
        battle = env.last_battle_result
        if battle and battle.get("is_boss", False):
            # Same-floor boss deaths used to be indistinguishable. Reward how
            # much of the boss encounter the branch actually removed, while
            # keeping any victory categorically better than a near-win.
            remaining = float(battle.get("monster_hp_fraction", 1.0))
            score += boss_progress_weight * max(0.0, min(1.0, 1.0 - remaining))
    elif gc.outcome == sts.GameOutcome.PLAYER_VICTORY:
        score += 3.0
    if gc.act >= 4:
        score += 0.05 * (
            int(gc.red_key) + int(gc.green_key) + int(gc.blue_key))
    return score



def act_for_floor(floor: float) -> int:
    """Act boundaries: 1-16, 17-33, 34-50, 51+."""
    return 1 + (floor >= 17) + (floor >= 34) + (floor >= 51)


def bootstrap_score(
    env: WholeRunEnv,
    policy,
    start_floor: int,
    start_act: int,
    floor_weight: float,
    act_weight: float,
    loss_penalty: float,
) -> float:
    """Estimate a truncated branch's score from the terminal_floor auxiliary head.

    Playing every continuation to terminal makes each score carry the variance of
    an entire run, which is the dominant term in the label noise (measured paired
    SNR ~1.0, with ~47% of labels unable to separate the best action from the
    runner-up). Truncating and bootstrapping trades that random variance for a
    *deterministic* state-dependent error, which partially cancels between
    sibling branches in a way Monte Carlo noise cannot.

    Whether the trade is favourable is an empirical question -- terminal_floor
    predicts with R2~0.66 / MAE~4.3 floors against a between-branch signal of
    ~1.7 floors -- so this is off by default and must be measured against
    untruncated labels using the stored per_rollout_scores.
    """
    if not hasattr(policy, "forward_detailed"):
        raise RuntimeError(
            "--truncate-after needs a policy with auxiliary heads "
            "(WholeRunTransformerPolicyV27); this checkpoint has none")
    with torch.inference_mode():
        auxiliary = policy.forward_detailed(env.observation())[2]
    if auxiliary is None or "terminal_floor" not in auxiliary:
        raise RuntimeError("policy exposes no terminal_floor auxiliary head")
    current_floor = float(env.gc.floor_num)
    predicted = float(auxiliary["terminal_floor"]) * 56.0
    predicted = max(current_floor, min(56.0, predicted))
    hp_fraction = max(0, env.gc.cur_hp) / max(1, env.gc.max_hp)
    score = (
        floor_weight * max(0.0, predicted - start_floor)
        + act_weight * max(0, act_for_floor(predicted) - start_act)
        + 0.12 * hp_fraction)
    # Nearly every A20 run ends in a loss, so terminal scores almost always carry
    # -loss_penalty. Applying it here too keeps truncated and terminal branches on
    # one scale; omitting it would systematically favour branches that survived
    # long enough to be truncated.
    return score - loss_penalty


def continue_branch(
    gc,
    policy: WholeRunTransformerPolicy,
    combat_sims: int,
    max_decisions: int,
    sample_seed: int,
    temperature: float,
    floor_weight: float,
    act_weight: float,
    loss_penalty: float,
    boss_progress_weight: float,
    deterministic_combat: bool,
    harvest: list | None = None,
    harvest_rate: float = 0.0,
    harvest_rng: random.Random | None = None,
    truncate_after: int = 0,
) -> float:
    # `harvest` collects (state, action, return) rows from inside the
    # continuation. These decisions are simulated either way: one label costs
    # rollouts x actions continuations of up to `max_decisions` each — roughly
    # 1,400 simulated decisions — and only the single label survives, a ~0.07%
    # retention rate. The rows are correlated and slightly off-policy, so they
    # are wrong for the policy head, but they are exactly what the value head and
    # the six auxiliary heads need and those are currently starved on the same
    # 4,008 rows as everything else. Sampled rather than exhaustive: keeping all
    # of them would cost ~60 MB per episode.
    start_floor, start_act = int(gc.floor_num), int(gc.act)
    harvest_start = len(harvest) if harvest is not None else 0
    env = WholeRunEnv(RunConfig(
        ascension=int(gc.ascension), combat_sims=combat_sims,
        max_decisions=max_decisions,
        deterministic_combat=deterministic_combat))
    env.gc = gc
    env.steps = 0
    env.battles = 0
    env.search_seed_base = sample_seed
    env.last_battle_result = None
    env._reset_combat_audit()
    env._resolve_battles()
    torch.manual_seed(sample_seed)
    truncated = False
    for step in range(max_decisions):
        if env.gc.outcome != sts.GameOutcome.UNDECIDED:
            break
        if truncate_after and step >= truncate_after:
            truncated = True
            break
        actions = env.legal_actions()
        if not actions:
            break
        obs = env.observation()
        with torch.inference_mode():
            logits, _ = policy(obs)
            distribution = torch.distributions.Categorical(
                logits=logits / max(temperature, 1e-3))
            action_index = int(distribution.sample())
        if (harvest is not None and harvest_rate > 0.0
                and (harvest_rng or random).random() < harvest_rate):
            harvest.append({
                "observation": compact_observation(obs),
                "action": action_index,
                "num_actions": len(actions),
                "decision_type": decision_type(env.gc),
                "floor": int(env.gc.floor_num),
                "act": int(env.gc.act),
            })
        _, _, done, _ = env.step(action_index)
        if done:
            break
    if truncated:
        score = bootstrap_score(
            env, policy, start_floor, start_act,
            floor_weight, act_weight, loss_penalty)
    else:
        score = rollout_score(
            env, start_floor, start_act,
            floor_weight, act_weight, loss_penalty, boss_progress_weight)
    if harvest is not None:
        # Each harvested decision's return is the same terminal outcome measured
        # from *its own* floor/act, which is exactly what rollout_score computes
        # given a different start. No extra simulation.
        for record in harvest[harvest_start:]:
            # A truncated branch never reached a terminal state, so rollout_score
            # would read a non-terminal floor/outcome and silently produce a
            # wrong return. Bootstrap those the same way the branch itself was
            # scored, measured from the harvested row's own floor/act.
            if truncated:
                record["return"] = bootstrap_score(
                    env, policy, record["floor"], record["act"],
                    floor_weight, act_weight, loss_penalty)
            else:
                record["return"] = rollout_score(
                    env, record["floor"], record["act"],
                    floor_weight, act_weight, loss_penalty, boss_progress_weight)
            record["bootstrapped"] = bool(truncated)
    return score


def label_state(
    gc,
    policy: WholeRunTransformerPolicy,
    combat_sims: int,
    rollouts: int,
    rollout_decisions: int,
    temperature: float,
    label_temperature: float,
    common_seed: int,
    floor_weight: float,
    act_weight: float,
    loss_penalty: float,
    boss_progress_weight: float,
    deterministic_combat: bool,
    harvest: list | None = None,
    harvest_rate: float = 0.0,
    harvest_rng: random.Random | None = None,
    truncate_after: int = 0,
):
    actions, _ = partition_legal_actions(gc)
    if len(actions) < 2:
        return None
    action_scores: list[list[float]] = [[] for _ in actions]
    for rollout_index in range(rollouts):
        matched_seed = common_seed + rollout_index
        for action_index, action in enumerate(actions):
            branch = gc.copy()
            if not action.isValidAction(branch):
                raise RuntimeError(
                    "counterfactual action became invalid after GameContext.copy(): "
                    f"index={action_index} bits={action.bits} "
                    f"screen={gc.screen_state} floor={gc.floor_num} "
                    f"map=({gc.cur_map_node_x},{gc.cur_map_node_y})")
            action.execute(branch)
            score = continue_branch(
                branch, policy, combat_sims, rollout_decisions,
                matched_seed, temperature,
                floor_weight, act_weight, loss_penalty,
                boss_progress_weight,
                deterministic_combat,
                harvest=harvest, harvest_rate=harvest_rate,
                harvest_rng=harvest_rng, truncate_after=truncate_after)
            action_scores[action_index].append(score)
    means = np.asarray([np.mean(scores) for scores in action_scores], dtype=np.float32)
    standard_errors = np.asarray([
        np.std(scores) / math.sqrt(max(1, len(scores))) for scores in action_scores
    ], dtype=np.float32)
    # Noisy or nearly tied candidates should remain soft rather than becoming
    # a brittle one-hot target.
    effective_temperature = label_temperature + float(np.mean(standard_errors))
    centered = (means - float(np.max(means))) / max(effective_temperature, 1e-3)
    probabilities = np.exp(centered)
    probabilities /= probabilities.sum()
    # Keep the raw per-rollout scores, shape (actions, rollouts). `standard_errors`
    # above is each action's *absolute* SE, but sibling branches share
    # `matched_seed` (common random numbers), so their errors are correlated and
    # the SE of a *paired difference* — which is what actually decides the label's
    # argmax — is smaller. Measured on v31, absolute SNR (gap / SE) has a median
    # of 0.90 with 53% of labels below 1.0; whether ranking is really that noisy
    # cannot be answered without the paired scores, and that answer gates whether
    # the estimator needs replacing.
    per_rollout = np.asarray(action_scores, dtype=np.float32)
    return probabilities.astype(np.float32), means, standard_errors, per_rollout


def compact_observation(obs):
    return {key: value for key, value in obs.items() if key != "action_text"}


def attach_episode_auxiliary_targets(
        episode_rows, combat_events, rest_steps, act_entry_hp,
        terminal_floor: int, terminal_act: int, victory: bool) -> None:
    """Attach normalized future outcomes to labeled states in one trajectory."""
    for row in episode_rows:
        decision = int(row.pop("_episode_decision"))
        row_act = int(row.get("act", 1))
        targets = {
            "next_rest_reach": float(any(
                step >= decision for step in rest_steps)),
            "terminal_floor": min(56, max(0, terminal_floor)) / 56.0,
        }
        next_combat = next(
            (event for event in combat_events
             if int(event["decision"]) >= decision),
            None)
        if next_combat is not None:
            targets["next_combat_survival"] = float(
                next_combat["survived"])
            targets["next_combat_hp"] = float(next_combat["hp_fraction"])
        if row_act < 4:
            reached_next_act = terminal_act > row_act or victory
            targets["act_boss_survival"] = float(reached_next_act)
            targets["next_act_entry_hp"] = float(
                act_entry_hp.get(row_act + 1, 0.0))
        row["auxiliary_targets"] = targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument(
        "--torch-threads", type=int, default=1,
        help="CPU intra-op threads; one avoids worker oversubscription")
    parser.add_argument("--out", required=True)
    parser.add_argument("--ascension", type=int, default=20)
    parser.add_argument("--combat-sims", type=int, default=25)
    parser.add_argument("--rollouts", type=int, default=3)
    parser.add_argument("--rollout-decisions", type=int, default=24)
    parser.add_argument("--policy-temperature", type=float, default=1.15)
    parser.add_argument("--label-temperature", type=float, default=0.12)
    parser.add_argument("--floor-weight", type=float, default=0.10)
    parser.add_argument("--act-weight", type=float, default=1.50)
    parser.add_argument("--loss-penalty", type=float, default=0.40)
    parser.add_argument(
        "--boss-progress-weight", type=float, default=0.0,
        help="credit [0, weight] for monster HP removed in a terminal boss loss")
    parser.add_argument(
        "--types", default=",".join(DECISION_TYPES),
        help="comma-separated decision types to collect")
    parser.add_argument("--per-type", type=int, default=4)
    parser.add_argument("--max-labels", type=int, default=36)
    parser.add_argument("--max-episodes", type=int, default=250)
    parser.add_argument(
        "--max-labels-per-episode", type=int, default=0,
        help="cap correlated labels from one run; zero keeps the legacy unlimited behavior")
    parser.add_argument("--min-act", type=int, default=1,
                        help="only label states at or after this act")
    parser.add_argument("--max-act", type=int, default=4,
                        help="only label states at or before this act")
    parser.add_argument("--min-floor", type=int, default=0)
    parser.add_argument("--max-floor", type=int, default=56)
    parser.add_argument(
        "--stochastic-combat", action="store_true",
        help="disable matched deterministic MCTS seeds during collection and labeling")
    parser.add_argument(
        "--seed-results", default=None,
        help="optional evaluation JSONL whose matching seeds become collection episodes")
    parser.add_argument(
        "--seed-checkpoint", default=None,
        help="checkpoint basename to select from --seed-results")
    parser.add_argument("--result-floor", type=int, default=None)
    parser.add_argument("--result-act", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1_020_000)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument(
        "--resume", action="store_true",
        help="continue from OUT.partial if a previous shard was interrupted")
    parser.add_argument(
        "--truncate-after", type=int, default=0,
        help="stop each continuation after N decisions and bootstrap the "
             "remaining value from the terminal_floor auxiliary head. 0 plays "
             "to terminal as before. Trades Monte Carlo variance for a "
             "deterministic estimator error -- measure before trusting")
    parser.add_argument(
        "--harvest-rate", type=float, default=0.0,
        help="fraction of continuation decisions kept as (state, action, return) "
             "rows. These are already simulated; 0 discards them as before")
    parser.add_argument(
        "--priority-accept-base", type=float, default=1.0,
        help=("base probability for labeling an eligible state; values below 1 "
              "favor low-HP, early-floor, later-act, rare, uncertain, and "
              "safety-filtered decisions"))
    parser.add_argument(
        "--trajectory-auxiliary-targets", action="store_true",
        help="label each retained state with observed future run outcomes")
    args = parser.parse_args()
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_policy(args.policy, device)
    requested_types = tuple(
        item.strip() for item in args.types.split(",") if item.strip())
    invalid_types = sorted(set(requested_types) - set(DECISION_TYPES))
    if invalid_types:
        raise ValueError(f"unknown decision types: {invalid_types}")
    rng = random.Random(args.seed)
    selected_run_seeds = None
    if args.seed_results:
        selected_run_seeds = []
        seen = set()
        with open(args.seed_results, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if args.seed_checkpoint and row.get("checkpoint") != args.seed_checkpoint:
                    continue
                if args.result_floor is not None and int(row.get("floor", -1)) != args.result_floor:
                    continue
                if args.result_act is not None and int(row.get("act", -1)) != args.result_act:
                    continue
                seed = int(row["seed"])
                if seed not in seen:
                    seen.add(seed)
                    selected_run_seeds.append(seed)
        if not selected_run_seeds:
            raise RuntimeError("no seeds matched --seed-results filters")
        print(f"selected_failure_seeds={len(selected_run_seeds)}", flush=True)
    rows = []
    # Harvested continuation rows go to a sibling file rather than into `rows`:
    # they are a different kind of supervision (scalar return, not a distribution
    # over actions) and mixing them would corrupt the policy-head dataset.
    harvest_rows: list | None = [] if args.harvest_rate > 0.0 else None
    harvest_rng = random.Random(args.seed ^ 0x5EED)
    counts = Counter()
    start_episode = 0
    partial_path = args.out + ".partial"
    if args.resume and os.path.exists(partial_path):
        partial = torch.load(partial_path, map_location="cpu", weights_only=False)
        rows = partial.get("rows", [])
        counts.update(row["decision_type"] for row in rows)
        partial_metadata = partial.get("metadata", {})
        start_episode = int(partial_metadata.get("next_episode", len(rows)))
        if "rng_state" in partial_metadata:
            rng.setstate(partial_metadata["rng_state"])
        print(
            f"resumed={partial_path} rows={len(rows)} "
            f"next_episode={start_episode}", flush=True)
    started = time.perf_counter()
    episode_count = (
        min(args.max_episodes, len(selected_run_seeds))
        if selected_run_seeds is not None else args.max_episodes)
    for episode in range(start_episode, episode_count):
        if len(rows) >= args.max_labels:
            break
        episode_row_start = len(rows)
        episode_labels = 0
        episode_eligible = 0
        episode_capped = 0
        episode_priority_rejected = 0
        episode_decision = 0
        combat_events = []
        rest_steps = []
        act_entry_hp = {}
        env = WholeRunEnv(RunConfig(
            ascension=args.ascension, combat_sims=args.combat_sims,
            deterministic_combat=not args.stochastic_combat))
        run_seed = (
            selected_run_seeds[episode] if selected_run_seeds is not None
            else rng.randrange(1, 2**31))
        obs = env.reset(run_seed)
        act_entry_hp[int(env.gc.act)] = (
            max(0, int(env.gc.cur_hp)) / max(1, int(env.gc.max_hp)))
        while (env.gc.outcome == sts.GameOutcome.UNDECIDED
               and env.steps < env.config.max_decisions):
            actions, filtered_actions = env._partition_legal_actions()
            if not actions:
                break
            kind = decision_type(env.gc)
            if kind == "rest":
                rest_steps.append(episode_decision)
            policy_logits = None
            # Structural eligibility: could this decision ever be labelled, before
            # any budget cap or priority sampling is applied? Counting it
            # separately is what makes the discard rate visible — the generator
            # plays every one of these with full MCTS whether or not it keeps it.
            structurally_eligible = (
                len(actions) >= 2 and kind in requested_types
                and args.min_act <= int(env.gc.act) <= args.max_act
                and args.min_floor <= int(env.gc.floor_num) <= args.max_floor
            )
            episode_eligible += int(structurally_eligible)
            should_label = (
                structurally_eligible
                and counts[kind] < args.per_type
                and len(rows) < args.max_labels
                and (args.max_labels_per_episode <= 0
                     or episode_labels < args.max_labels_per_episode)
            )
            episode_capped += int(structurally_eligible and not should_label)
            wanted_before_priority = should_label
            priority_score = 0.0
            if should_label and args.priority_accept_base < 1.0:
                hp_fraction = env.gc.cur_hp / max(1.0, float(env.gc.max_hp))
                priority_score += 2.0 * float(hp_fraction <= 0.35)
                priority_score += float(int(env.gc.floor_num) <= 8)
                priority_score += float(int(env.gc.act) >= 2)
                priority_score += float(
                    kind in ("event", "shop", "rest", "boss_relic"))
                priority_score += 6.0 * float(filtered_actions > 0)
                with torch.inference_mode():
                    policy_logits, _ = policy(obs)
                if len(policy_logits) >= 2:
                    top = torch.topk(policy_logits, 2).values
                    priority_score += float(float(top[0] - top[1]) < 0.5)
                accept_probability = min(
                    1.0, args.priority_accept_base * (2.0 ** priority_score))
                should_label = rng.random() < accept_probability
                episode_priority_rejected += int(
                    wanted_before_priority and not should_label)
            selected = None
            if should_label:
                result = label_state(
                    env.gc, policy, args.combat_sims, args.rollouts,
                    args.rollout_decisions, args.policy_temperature,
                    args.label_temperature,
                    common_seed=args.seed + episode * 10_000 + len(rows) * 100,
                    floor_weight=args.floor_weight,
                    act_weight=args.act_weight,
                    loss_penalty=args.loss_penalty,
                    boss_progress_weight=args.boss_progress_weight,
                    deterministic_combat=not args.stochastic_combat,
                    harvest=harvest_rows,
                    harvest_rate=args.harvest_rate,
                    harvest_rng=harvest_rng,
                    truncate_after=args.truncate_after)
                if result is not None:
                    probabilities, scores, errors, per_rollout = result
                    rows.append({
                        "observation": compact_observation(obs),
                        "target_probabilities": probabilities,
                        "mean_scores": scores,
                        "standard_errors": errors,
                        "per_rollout_scores": per_rollout,
                        "decision_type": kind,
                        "seed": run_seed,
                        "floor": int(env.gc.floor_num),
                        "act": int(env.gc.act),
                        "priority_score": priority_score,
                        "immediate_loss_actions_filtered": filtered_actions,
                        **({"_episode_decision": episode_decision}
                           if args.trajectory_auxiliary_targets else {}),
                    })
                    counts[kind] += 1
                    episode_labels += 1
                    selected = int(np.argmax(probabilities))
                    print(
                        f"label={len(rows)} type={kind} floor={env.gc.floor_num} "
                        f"actions={len(actions)} probs={np.round(probabilities, 3).tolist()} "
                        f"counts={dict(counts)}", flush=True)
                    if (args.save_every
                            and not args.trajectory_auxiliary_targets
                            and len(rows) % args.save_every == 0):
                        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
                        torch.save({
                            "rows": rows,
                            "metadata": {
                                **vars(args), "counts": dict(counts),
                                # Resume at the next run rather than midway
                                # through the current trajectory.
                                "next_episode": episode + 1,
                                "rng_state": rng.getstate(),
                            },
                        }, partial_path)
            if selected is None:
                if policy_logits is None:
                    with torch.inference_mode():
                        policy_logits, _ = policy(obs)
                selected = int(torch.argmax(policy_logits))
            battles_before = env.battles
            act_before = int(env.gc.act)
            obs, _, done, _ = env.step(selected)
            if env.battles > battles_before and env.last_battle_result:
                battle = env.last_battle_result
                combat_events.append({
                    "decision": episode_decision,
                    "survived": int(battle.get("player_hp", 0)) > 0,
                    "hp_fraction": (
                        max(0, int(battle.get("player_hp", 0)))
                        / max(1, int(battle.get("player_max_hp", 1)))),
                })
            if int(env.gc.act) > act_before:
                act_entry_hp[int(env.gc.act)] = (
                    max(0, int(env.gc.cur_hp))
                    / max(1, int(env.gc.max_hp)))
            episode_decision += 1
            if done:
                break
        print(
            f"episode={episode} decisions={episode_decision} "
            f"eligible={episode_eligible} labeled={episode_labels} "
            f"capped={episode_capped} priority_rejected={episode_priority_rejected}",
            flush=True)
        if args.trajectory_auxiliary_targets:
            attach_episode_auxiliary_targets(
                rows[episode_row_start:], combat_events, rest_steps,
                act_entry_hp, int(env.gc.floor_num), int(env.gc.act),
                env.gc.outcome == sts.GameOutcome.PLAYER_VICTORY)
            if args.save_every and len(rows) % args.save_every == 0:
                os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
                torch.save({
                    "rows": rows,
                    "metadata": {
                        **vars(args), "counts": dict(counts),
                        "next_episode": episode + 1,
                        "rng_state": rng.getstate(),
                    },
                }, partial_path)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({
        "rows": rows,
        "metadata": {
            **vars(args), "counts": dict(counts),
            "seconds": time.perf_counter() - started,
        },
    }, args.out)
    if harvest_rows:
        harvest_path = f"{os.path.splitext(args.out)[0]}.harvest.pt"
        torch.save({
            "rows": harvest_rows,
            "metadata": {**vars(args), "kind": "continuation-harvest"},
        }, harvest_path)
        print(
            f"saved_harvest={harvest_path} rows={len(harvest_rows)} "
            f"({len(harvest_rows)/max(1,len(rows)):.1f} per label)", flush=True)
    print(
        f"saved={args.out} labels={len(rows)} counts={dict(counts)} "
        f"seconds={time.perf_counter() - started:.1f}", flush=True)


if __name__ == "__main__":
    main()
