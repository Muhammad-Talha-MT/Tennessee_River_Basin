"""Lumped forecast LSTM baseline (one basin at a time, no river network).

Follows NeuralHydrology's handoff forecast LSTM (Kratzert et al.; Nearing et al.
2022 for observed flow as an input): an encoder LSTM reads the basin's history
(basin-mean weather, observed flow and its availability flag, attributes); a
linear state handoff turns its final hidden and cell states into the initial
states of a decoder LSTM, which reads the forecast rain, the attributes and a
lead embedding at each of the forecast steps. A linear head gives normalized log
flow per lead.
"""

import torch
from torch import nn


class ForecastLSTM(nn.Module):
    def __init__(self, dynamic=18, static=48, future=2, hidden=256, horizon=40, dropout=0.4):
        super().__init__()
        self.horizon = horizon
        self.encoder = nn.LSTM(dynamic + static, hidden, batch_first=True)
        self.handoff_h = nn.Linear(hidden, hidden)
        self.handoff_c = nn.Linear(hidden, hidden)
        self.lead_embedding = nn.Embedding(horizon, 16)
        self.decoder = nn.LSTM(future + static + 16, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, history, static, future):
        """history [B, time, dynamic], static [B, static], future [B, lead, future] -> [B, lead]."""
        batch, steps, _ = history.shape
        encoded, (h, c) = self.encoder(
            torch.cat([history, static[:, None].expand(-1, steps, -1)], dim=-1)
        )
        h, c = torch.tanh(self.handoff_h(h)), self.handoff_c(c)
        lead = self.lead_embedding(torch.arange(self.horizon, device=future.device))
        decoded, _ = self.decoder(
            torch.cat(
                [
                    future,
                    static[:, None].expand(-1, self.horizon, -1),
                    lead[None].expand(batch, -1, -1),
                ],
                dim=-1,
            ),
            (h, c),
        )
        return self.head(self.dropout(decoded))[..., 0]
