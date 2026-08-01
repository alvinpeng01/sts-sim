"""The run-level critic PPO collects against.

`collect_run_value_data.py` measured three predictors of on-policy return over
484,486 states:

    floor+act+screen only          val R2 +0.2232
    state (96-d, the model's head) val R2 +0.2973
    state + (floor, act, screen)   val R2 +0.3202

The model's own `value.*` head cannot take the third form -- it is
`Linear(96,96) -> GELU -> Linear(96,1)` and the checkpoint shape is fixed -- so
the critic lives here instead of inside the policy.  Keeping it separate also
matches what the RL loop needs: the critic is updated every iteration against
the reward the environment actually emits, while the policy checkpoint format
stays untouched.

The scalars are not redundant with `state`.  A lookup table keyed on floor alone
scores +0.2540, so depth is most of the signal and the trunk does not surface it
in a form a small head can read; supplying it directly is worth ~+0.02 R2 over
`state` and ~+0.10 over the scalars alone.

Run from slay-sim/:
    python -m lightspeed.run_critic --fit --out runs/run_critic_v37.pt
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from torch import nn

SCALAR_DIM = 3


def scalars_from_obs(obs) -> np.ndarray:
    """The same three scalars the fit measured, in the same normalization."""
    return np.asarray([
        float(obs.get("floor", 0)) / 56.0,
        float(obs.get("act", 1)) / 4.0,
        float(obs.get("screen", 0)) / 10.0,
    ], dtype=np.float32)


class RunCritic(nn.Module):
    """V(s) over the frozen trunk's state token plus run-position scalars."""

    def __init__(self, dim: int = 96, hidden: int = 96):
        super().__init__()
        self.dim = dim
        self.body = nn.Sequential(
            nn.Linear(dim + SCALAR_DIM, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, state: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state, scalars = state.unsqueeze(0), scalars.unsqueeze(0)
        return self.body(torch.cat((state, scalars), dim=-1)).squeeze(-1)

    @torch.inference_mode()
    def value_of(self, state: np.ndarray, obs) -> float:
        return float(self(torch.from_numpy(state),
                         torch.from_numpy(scalars_from_obs(obs)))[0])


def load(path: str, device=None) -> RunCritic:
    payload = torch.load(path, map_location=device or "cpu", weights_only=False)
    critic = RunCritic(dim=payload.get("dim", 96), hidden=payload.get("hidden", 96))
    critic.load_state_dict(payload["state_dict"])
    return critic.eval()


def fit_from_cache(cache: str, out: str, epochs: int, batch: int, lr: float,
                   validation_fraction: float) -> float:
    from .collect_run_value_data import stack
    from .train_value_from_harvest import fit_head, r2

    payload = torch.load(cache, weights_only=False, map_location="cpu")
    episodes = payload["episodes"]
    cut = int(len(episodes) * (1.0 - validation_fraction))
    train, validation = stack(episodes[:cut]), stack(episodes[cut:])

    def design(part):
        scalars = np.stack([part["floors"] / 56.0, part["acts"] / 4.0,
                            part["screens"] / 10.0], axis=1).astype(np.float32)
        return np.concatenate([part["features"], scalars], axis=1)

    train_x, val_x = design(train), design(validation)
    print(f"fitting critic on {len(train_x)} states, "
          f"validating on {len(val_x)} (seed-disjoint)", flush=True)

    critic = RunCritic(dim=train["features"].shape[1])
    # fit_head drives any nn.Module mapping [N, D] -> [N, 1]; RunCritic splits
    # its own input, so wrap it to keep one training routine for both.
    wrapper = _Flat(critic)
    wrapper, best = fit_head(wrapper, train_x, train["returns"], val_x,
                             validation["returns"], epochs, batch, lr, "critic")
    torch.save({"state_dict": critic.state_dict(), "dim": critic.dim,
                "hidden": 96, "val_r2": best, "cache": cache}, out)
    baseline = r2(np.full_like(validation["returns"], train["returns"].mean()),
                  validation["returns"])
    print(f"\ncritic val R2 {best:+.4f} (predict-the-mean {baseline:+.4f})")
    print(f"wrote {out}")
    return best


class _Flat(nn.Module):
    def __init__(self, critic: RunCritic):
        super().__init__()
        self.critic = critic

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x[:, :self.critic.dim],
                           x[:, self.critic.dim:]).unsqueeze(-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--cache", default="runs/run_value_data.pt")
    parser.add_argument("--out", default="runs/run_critic_v37.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.threads))
    torch.manual_seed(0)
    if not args.fit:
        raise SystemExit("nothing to do; pass --fit")
    fit_from_cache(args.cache, args.out, args.epochs, args.batch, args.lr,
                   args.validation_fraction)


if __name__ == "__main__":
    main()
