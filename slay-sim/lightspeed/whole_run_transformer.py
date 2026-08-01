"""Native whole-run transformer policy.

Unlike the compact baseline, this model attends over the actual deck, relics,
potions, and map nodes.  It still scores only the currently legal candidate
actions, which keeps the action space variable and simulator-native.
"""
from __future__ import annotations

import torch
from torch import nn


class WholeRunTransformerPolicy(nn.Module):
    def __init__(self, dim: int = 96, layers: int = 2, heads: int = 4):
        super().__init__()
        self.dim = dim
        self.card = nn.Embedding(400, dim)
        self.relic = nn.Embedding(200, dim)
        self.potion = nn.Embedding(100, dim)
        self.room = nn.Embedding(16, dim)
        self.action_content = nn.Embedding(700, dim)
        self.fixed = nn.Sequential(nn.Linear(10, dim), nn.GELU(), nn.Linear(dim, dim))
        self.upgrade = nn.Linear(1, dim)
        self.counter = nn.Linear(1, dim)
        self.coord = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        # Each map-node token also carries its three outgoing x-coordinates.
        # This lets the transformer distinguish maps with the same room
        # layout but different reachable routes.
        self.map_edges = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.action_features = nn.Sequential(nn.Linear(6, dim), nn.GELU(), nn.Linear(dim, dim))
        self.target_room = nn.Embedding(16, dim)
        self.target_coord = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.route_cone = nn.Sequential(nn.Linear(4, dim), nn.GELU(), nn.Linear(dim, dim))
        self.route_resources = nn.Sequential(nn.Linear(4, dim), nn.GELU(), nn.Linear(dim, dim))
        self.shop_price = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.action_consequence = nn.Sequential(
            nn.Linear(10, dim), nn.GELU(), nn.Linear(dim, dim))
        self.event = nn.Embedding(64, dim)
        self.neow_bonus = nn.Embedding(32, dim)
        self.neow_drawback = nn.Embedding(16, dim)
        # A small act-specific policy adapter allows later-act replay to
        # improve without perturbing the already validated Act 1 policy.
        self.act_action = nn.Embedding(5, dim)
        self.act_action_scale = nn.Embedding(5, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, dim_feedforward=dim * 4,
                                           dropout=0.0, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.score = nn.Sequential(nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, 1))
        self.act_score = nn.ModuleList([
            nn.Sequential(nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, 1))
            for _ in range(5)
        ])
        # Four floor phases per act. This supports isolated late-act fixes:
        # training Act 1 floors 12-17 cannot alter early Act 1 or later acts.
        self.phase_score = nn.ModuleList([
            nn.Sequential(nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, 1))
            for _ in range(5 * 4)
        ])
        # Shared human-demonstration prior. This is deliberately additive and
        # zero-initialized: archive imitation can be trained/evaluated in
        # isolation without rewriting the proven simulator-trained policy.
        self.human_score = nn.Sequential(
            nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, 1))
        self.value = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
        # These are additive extensions to checkpoints trained before map
        # target/edge features existed.  Start them as no-ops so compatible
        # loading preserves the proven policy, then let fine-tuning learn the
        # new signal rather than injecting random action preferences.
        nn.init.zeros_(self.target_room.weight)
        nn.init.zeros_(self.map_edges[-1].weight)
        nn.init.zeros_(self.map_edges[-1].bias)
        nn.init.zeros_(self.target_coord[-1].weight)
        nn.init.zeros_(self.target_coord[-1].bias)
        nn.init.zeros_(self.route_cone[-1].weight)
        nn.init.zeros_(self.route_cone[-1].bias)
        nn.init.zeros_(self.route_resources[-1].weight)
        nn.init.zeros_(self.route_resources[-1].bias)
        nn.init.zeros_(self.shop_price[-1].weight)
        nn.init.zeros_(self.shop_price[-1].bias)
        nn.init.zeros_(self.action_consequence[-1].weight)
        nn.init.zeros_(self.action_consequence[-1].bias)
        nn.init.zeros_(self.event.weight)
        nn.init.zeros_(self.neow_bonus.weight)
        nn.init.zeros_(self.neow_drawback.weight)
        nn.init.zeros_(self.act_action.weight)
        nn.init.zeros_(self.act_action_scale.weight)
        for residual_head in self.act_score:
            nn.init.zeros_(residual_head[-1].weight)
            nn.init.zeros_(residual_head[-1].bias)
        for residual_head in self.phase_score:
            nn.init.zeros_(residual_head[-1].weight)
            nn.init.zeros_(residual_head[-1].bias)
        nn.init.zeros_(self.human_score[-1].weight)
        nn.init.zeros_(self.human_score[-1].bias)

    @staticmethod
    def _tensor(value, dtype, device):
        return torch.as_tensor(value, dtype=dtype, device=device)

    def _state_tokens(self, obs, device):
        fixed = self._tensor(obs["fixed"], torch.float32, device).clamp(0, 200) / 200.0
        tokens = [self.fixed(fixed).unsqueeze(0)]  # learned summary/CLS token
        deck = self._tensor(obs["deck_ids"], torch.long, device).clamp(0, 399)
        if len(deck):
            upgrades = self._tensor(obs["deck_upgrades"], torch.float32, device).unsqueeze(-1) / 20.0
            tokens.append(self.card(deck) + self.upgrade(upgrades))
        relics = self._tensor(obs["relic_ids"], torch.long, device).clamp(0, 199)
        if len(relics):
            counts = self._tensor(obs["relic_counters"], torch.float32, device).unsqueeze(-1) / 100.0
            tokens.append(self.relic(relics) + self.counter(counts))
        potions = self._tensor(obs["potions"], torch.long, device).clamp(0, 99)
        if len(potions):
            tokens.append(self.potion(potions))
        rooms = self._tensor(obs["map_rooms"], torch.long, device).clamp(0, 15)
        if len(rooms):
            xy = torch.stack((self._tensor(obs["map_xs"], torch.float32, device) / 7.0,
                              self._tensor(obs["map_ys"], torch.float32, device) / 16.0), dim=1)
            paths = self._tensor(obs.get("map_paths", []), torch.float32, device)
            if paths.numel() == len(rooms) * 3:
                # -1 means no outgoing edge; shifting keeps it distinct from
                # a real edge to x=0 while retaining a compact range.
                paths = (paths + 1.0) / 8.0
                path_features = self.map_edges(paths.reshape(-1, 3))
            else:
                path_features = torch.zeros_like(xy[:, :1]).expand(-1, self.dim)
            tokens.append(self.room(rooms) + self.coord(xy) + path_features)
        return torch.cat(tokens, dim=0).unsqueeze(0)

    def forward(self, obs):
        device = next(self.parameters()).device
        encoded = self.encoder(self._state_tokens(obs, device))[0]
        state = encoded[0]  # CLS has attended to every variable-size token
        features = self._tensor(obs["action_features"], torch.float32, device)
        content = self._tensor(obs["action_content_ids"], torch.long, device).clamp(0, 699)
        action = self.action_features(features)
        action_content = self.action_content(content)
        target_rooms = self._tensor(obs.get("action_target_rooms", [0] * len(content)), torch.long, device).clamp(0, 15)
        target_coords = self._tensor(obs.get("action_target_coords", [[-1.0, -1.0]] * len(content)), torch.float32, device)
        route_cones = self._tensor(obs.get("action_route_cones", [[0.0] * 4] * len(content)), torch.float32, device)
        route_resources = self._tensor(
            obs.get("action_route_resources", [[0.0] * 4] * len(content)),
            torch.float32, device)
        prices = self._tensor(obs.get("action_prices", [[0.0, 0.0]] * len(content)), torch.float32, device)
        consequences = self._tensor(
            obs.get("action_consequences", [[0.0] * 10] * len(content)),
            torch.float32, device)
        event_ids = self._tensor(obs.get("action_event_ids", [0] * len(content)), torch.long, device).clamp(0, 63)
        bonuses = self._tensor(obs.get("action_neow_bonuses", [0] * len(content)), torch.long, device).clamp(0, 31)
        drawbacks = self._tensor(obs.get("action_neow_drawbacks", [0] * len(content)), torch.long, device).clamp(0, 15)
        act_id = torch.tensor(
            min(4, max(0, int(obs.get("act", 1)))),
            dtype=torch.long, device=device)
        action = (action + self.target_room(target_rooms) + self.target_coord(target_coords)
                  + self.route_cone(route_cones)
                  + self.route_resources(route_resources)
                  + self.shop_price(prices) + self.event(event_ids)
                  + self.neow_bonus(bonuses) + self.neow_drawback(drawbacks))
        # Immediate deltas are reliable discriminators for acquiring rewards,
        # selecting a boss relic, and purchasing shop inventory.  Do not let
        # this adapter leak shortcuts learned there into routing, Neow, or
        # rest/event strategy, where long-horizon value dominates.
        if int(obs.get("screen", 0)) in (2, 3, 8):
            action = action + self.action_consequence(consequences)
        action = (action * (1.0 + self.act_action_scale(act_id).unsqueeze(0))
                  + self.act_action(act_id).unsqueeze(0))
        state_expanded = state.unsqueeze(0).expand(features.shape[0], -1)
        score_input = torch.cat((state_expanded, action, action_content), dim=1)
        logits = self.score(score_input).squeeze(-1)
        logits = logits + self.act_score[int(act_id)](score_input).squeeze(-1)
        floor = max(0, int(obs.get("floor", 0)))
        floor_phase = min(3, floor // 6)
        phase_index = int(act_id) * 4 + floor_phase
        logits = logits + self.phase_score[phase_index](score_input).squeeze(-1)
        # The archive importer currently produces complete candidate sets
        # only for ordinary card rewards and boss relics. Do not extrapolate
        # this residual into map/event/rest/shop decisions that had no human
        # counterfactual supervision.
        if int(obs.get("screen", 0)) in (2, 3):
            logits = logits + self.human_score(score_input).squeeze(-1)
        return logits, self.value(state).squeeze(-1)

    def act(self, obs, sample=True, temperature: float = 1.0):
        """Choose an action; `temperature` sharpens (<1) or flattens (>1) sampling.

        Argmax is not a neutral readout of a policy that has only learned the
        marginal. On campfire decisions the net emits ~P(REST)=0.41 /
        P(SMITH)=0.35 in every state — matching the label marginal closely — but
        argmax turns that into REST 100% of the time, while the labels prefer
        SMITH in 26% of states. Sampling preserves the learned proportions;
        temperature trades that against the variance sampling introduces.
        """
        logits, value = self(obs)
        if sample:
            scaled = logits / max(1e-6, temperature)
            dist = torch.distributions.Categorical(logits=scaled)
            index = dist.sample()
            log_prob = dist.log_prob(index)
        else:
            index = torch.argmax(logits)
            log_prob = logits[index] - logits.logsumexp(dim=-1)
        return int(index), log_prob, value, logits
