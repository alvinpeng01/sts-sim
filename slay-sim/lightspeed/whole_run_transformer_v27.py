"""Isolated v27 whole-run policy with scoped experts and uncertainty."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .v27_features import (
    CARD_STRUCTURE_SIZE,
    DECK_SUMMARY_SIZE,
    STRATEGIC_CONTEXT_SIZE,
    augment_v27_observation,
)
from .whole_run_transformer import WholeRunTransformerPolicy


def _zero_residual(module: nn.Sequential) -> None:
    nn.init.zeros_(module[-1].weight)
    nn.init.zeros_(module[-1].bias)


class WholeRunTransformerPolicyV27(WholeRunTransformerPolicy):
    """Backward-compatible policy; every new policy residual begins as zero."""

    def __init__(self, dim: int = 96, layers: int = 2, heads: int = 4):
        super().__init__(dim=dim, layers=layers, heads=heads)
        self.deck_summary_adapter = nn.Sequential(
            nn.Linear(DECK_SUMMARY_SIZE, dim), nn.GELU(), nn.Linear(dim, dim))
        self.strategic_context_adapter = nn.Sequential(
            nn.Linear(STRATEGIC_CONTEXT_SIZE, dim), nn.GELU(), nn.Linear(dim, dim))
        self.action_card_structure_adapter = nn.Sequential(
            nn.Linear(CARD_STRUCTURE_SIZE, dim), nn.GELU(), nn.Linear(dim, dim))
        # IDs 1..8 mirror native screens; 9 is Neow and 0 is fallback.
        self.decision_experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, 1))
            for _ in range(10)
        ])
        # Independently bootstrapped residuals provide epistemic disagreement.
        self.uncertainty_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, 1))
            for _ in range(3)
        ])
        self.auxiliary_heads = nn.ModuleDict({
            "next_combat_survival": nn.Linear(dim, 1),
            "next_combat_hp": nn.Linear(dim, 1),
            "next_rest_reach": nn.Linear(dim, 1),
            "act_boss_survival": nn.Linear(dim, 1),
            "next_act_entry_hp": nn.Linear(dim, 1),
            "terminal_floor": nn.Linear(dim, 1),
        })
        for module in (
                self.deck_summary_adapter, self.strategic_context_adapter,
                self.action_card_structure_adapter):
            _zero_residual(module)
        for module in self.decision_experts:
            _zero_residual(module)
        for module in self.uncertainty_heads:
            _zero_residual(module)
        self._fast_ensemble_weights = None

    def _refresh_fast_ensemble(self) -> None:
        """Fuse three independent MLP heads for exact-mean inference."""
        with torch.no_grad():
            first_weight = torch.cat(
                [head[0].weight for head in self.uncertainty_heads], dim=0)
            first_bias = torch.cat(
                [head[0].bias for head in self.uncertainty_heads], dim=0)
            second_weight = torch.cat(
                [head[2].weight for head in self.uncertainty_heads],
                dim=1) / len(self.uncertainty_heads)
            second_bias = torch.stack(
                [head[2].bias for head in self.uncertainty_heads]).mean(dim=0)
        # Deliberately not registered as parameters/buffers: these are an
        # inference cache derived from checkpoint parameters and should never
        # appear in a state dict or receive optimizer updates.
        self._fast_ensemble_weights = (
            first_weight, first_bias, second_weight, second_bias)

    def load_state_dict(self, *args, **kwargs):
        result = super().load_state_dict(*args, **kwargs)
        self._refresh_fast_ensemble()
        return result

    @staticmethod
    def _expert_id(obs) -> int:
        screen = min(8, max(0, int(obs.get("screen", 0))))
        if screen == 1:
            bonuses = obs.get("action_neow_bonuses", [])
            drawbacks = obs.get("action_neow_drawbacks", [])
            if any(int(value) for value in bonuses) or any(
                    int(value) for value in drawbacks):
                return 9
        return screen

    def forward_detailed(self, raw_obs, diagnostics: bool = True):
        obs = augment_v27_observation(raw_obs)
        device = next(self.parameters()).device
        encoded = self.encoder(self._state_tokens(obs, device))[0]
        state = encoded[0]
        state = state + self.deck_summary_adapter(self._tensor(
            obs["v27_deck_summary"], torch.float32, device))
        state = state + self.strategic_context_adapter(self._tensor(
            obs["v27_strategic_context"], torch.float32, device))

        features = self._tensor(obs["action_features"], torch.float32, device)
        content = self._tensor(
            obs["action_content_ids"], torch.long, device).clamp(0, 699)
        action = self.action_features(features)
        action_content = self.action_content(content)
        target_rooms = self._tensor(
            obs.get("action_target_rooms", [0] * len(content)),
            torch.long, device).clamp(0, 15)
        target_coords = self._tensor(
            obs.get("action_target_coords", [[-1.0, -1.0]] * len(content)),
            torch.float32, device)
        route_cones = self._tensor(
            obs.get("action_route_cones", [[0.0] * 4] * len(content)),
            torch.float32, device)
        route_resources = self._tensor(
            obs.get("action_route_resources", [[0.0] * 4] * len(content)),
            torch.float32, device)
        prices = self._tensor(
            obs.get("action_prices", [[0.0, 0.0]] * len(content)),
            torch.float32, device)
        consequences = self._tensor(
            obs.get("action_consequences", [[0.0] * 10] * len(content)),
            torch.float32, device)
        event_ids = self._tensor(
            obs.get("action_event_ids", [0] * len(content)),
            torch.long, device).clamp(0, 63)
        bonuses = self._tensor(
            obs.get("action_neow_bonuses", [0] * len(content)),
            torch.long, device).clamp(0, 31)
        drawbacks = self._tensor(
            obs.get("action_neow_drawbacks", [0] * len(content)),
            torch.long, device).clamp(0, 15)
        card_structure = self._tensor(
            obs["v27_action_card_structure"], torch.float32, device)
        act_id = torch.tensor(
            min(4, max(0, int(obs.get("act", 1)))),
            dtype=torch.long, device=device)
        action = (
            action + self.target_room(target_rooms)
            + self.target_coord(target_coords) + self.route_cone(route_cones)
            + self.route_resources(route_resources) + self.shop_price(prices)
            + self.event(event_ids) + self.neow_bonus(bonuses)
            + self.neow_drawback(drawbacks)
            + self.action_card_structure_adapter(card_structure)
        )
        if int(obs.get("screen", 0)) in (2, 3, 8):
            action = action + self.action_consequence(consequences)
        action = (
            action * (1.0 + self.act_action_scale(act_id).unsqueeze(0))
            + self.act_action(act_id).unsqueeze(0))
        state_expanded = state.unsqueeze(0).expand(features.shape[0], -1)
        score_input = torch.cat(
            (state_expanded, action, action_content), dim=1)
        base_logits = self.score(score_input).squeeze(-1)
        base_logits = (
            base_logits
            + self.act_score[int(act_id)](score_input).squeeze(-1))
        floor = max(0, int(obs.get("floor", 0)))
        phase_index = int(act_id) * 4 + min(3, floor // 6)
        base_logits = (
            base_logits
            + self.phase_score[phase_index](score_input).squeeze(-1))
        if int(obs.get("screen", 0)) in (2, 3):
            base_logits = (
                base_logits + self.human_score(score_input).squeeze(-1))
        expert_logits = (
            base_logits
            + self.decision_experts[self._expert_id(obs)](
                score_input).squeeze(-1))
        if diagnostics:
            ensemble_logits = torch.stack([
                expert_logits + head(score_input).squeeze(-1)
                for head in self.uncertainty_heads
            ])
            logits = ensemble_logits.mean(dim=0)
            uncertainty = ensemble_logits.std(dim=0, unbiased=False)
            auxiliary = {
                name: head(state).squeeze(-1)
                for name, head in self.auxiliary_heads.items()
            }
        else:
            # Production action selection needs the ensemble mean but not
            # member logits, disagreement, or auxiliary predictions. Sum the
            # same three residual heads directly: this is numerically
            # equivalent to stack(...).mean(0) while avoiding the stack/std
            # kernels and six unused auxiliary-head calls.
            cached = self._fast_ensemble_weights
            if (cached is None or cached[0].device != score_input.device
                    or cached[0].dtype != score_input.dtype):
                self._refresh_fast_ensemble()
                cached = self._fast_ensemble_weights
            first_weight, first_bias, second_weight, second_bias = cached
            ensemble_residual = F.linear(
                F.gelu(F.linear(score_input, first_weight, first_bias)),
                second_weight, second_bias).squeeze(-1)
            logits = expert_logits + ensemble_residual
            ensemble_logits = None
            uncertainty = None
            auxiliary = None
        value = self.value(state).squeeze(-1)
        return logits, value, auxiliary, uncertainty, ensemble_logits

    def forward(self, obs):
        logits, value, _, _, _ = self.forward_detailed(
            obs, diagnostics=False)
        return logits, value

    def act_with_uncertainty(self, obs, sample: bool = True):
        logits, value, auxiliary, uncertainty, _ = self.forward_detailed(obs)
        distribution = torch.distributions.Categorical(logits=logits)
        index = distribution.sample() if sample else torch.argmax(logits)
        return (
            int(index), distribution.log_prob(index), value, logits,
            uncertainty, auxiliary)
