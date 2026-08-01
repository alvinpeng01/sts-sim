"""Compact policy/value model for WholeRunEnv.

It scores the variable set of legal actions directly, so no copied fixed
Silverbot action vocabulary is required.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class WholeRunPolicy(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.state = nn.Sequential(nn.Linear(18, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.action = nn.Sequential(nn.Linear(6, hidden), nn.Tanh())
        self.content = nn.Embedding(700, 16)
        self.score = nn.Sequential(nn.Linear(hidden * 2 + 16, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.value = nn.Linear(hidden, 1)

    @staticmethod
    def tensors(obs, device=None):
        fixed = torch.as_tensor(obs["fixed"], dtype=torch.float32, device=device) / 200.0
        deck = torch.as_tensor(obs["deck_ids"], dtype=torch.float32, device=device)
        relics = torch.as_tensor(obs["relic_ids"], dtype=torch.float32, device=device)
        potions = torch.as_tensor(obs["potions"], dtype=torch.float32, device=device)
        # Fixed-width summary keeps the model independent of deck/relic length.
        state = torch.cat((fixed, torch.tensor([
            len(deck) / 96.0, float(deck.sum()) / 96.0,
            len(relics) / 40.0, float(relics.sum()) / 181.0,
            len(potions) / 5.0, float(potions.sum()) / 44.0,
            float(obs["map_x"]) / 7.0, float(obs["map_y"]) / 16.0,
        ], dtype=torch.float32, device=device)))
        actions = torch.as_tensor(obs["action_features"], dtype=torch.float32, device=device)
        content = torch.as_tensor(obs["action_content_ids"], dtype=torch.long, device=device)
        return state, actions, content

    def forward(self, obs):
        state, actions, content = self.tensors(obs, next(self.parameters()).device)
        h = self.state(state)
        logits = self.score(torch.cat((h.expand(actions.shape[0], -1), self.action(actions), self.content(content)), dim=1)).squeeze(-1)
        return logits, self.value(h).squeeze(-1)

    def act(self, obs, sample=True):
        logits, value = self(obs)
        dist = torch.distributions.Categorical(logits=logits)
        idx = dist.sample() if sample else torch.argmax(logits)
        return int(idx), dist.log_prob(idx), value, logits
