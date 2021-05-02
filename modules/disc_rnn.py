import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from typing import List


class RNNDisc(nn.Module):
    """
    RNN based Discriminator: D(traj) -> logit
    """

    def __init__(
        self,
        dummy_inputs: List,  # provide a list to infer size from, eg: [state0] or [state0, act0]
        flatten: bool = True,
        hidden_size: int = 100,
        dropout: float = 0.0,
        bidirectional: bool = True,
    ):
        super(RNNADisc, self).__init__()

        self.input_size = input_dim
        self.hidden_size = hidden_size
        self.layers = 1
        self.dropout = encoder_dropout
        self.bi = bidirectional

        self.rnn_block = nn.LSTM(
            self.input_size,
            self.hidden_size,
            self.layers,
            dropout=self.dropout,
            bidirectional=self.bi,
            batch_first=True,
        )

        fin_h_size = self.hidden_size * 2 if self.bi else self.hidden_size

        self.classify_linear = nn.Sequential(
                nn.ReLU(),
                nn.Linear(fin_h_size, 1),
            )

    def forward(self, inputs, input_lengths):

        packed_input = pack_padded_sequence(
            inputs, input_lengths, enforce_sorted=False, batch_first=True
        )

        self.batch_size = inputs.size()[0]
        self.inferred_device = inputs.device

        _, (h_n, _) = self.rnn_block(packed_input)

        logits = self.classify_linear(h_n.p)

        return logits

    def classify_user(self, h_n):
        classify_h_n = h_n.contiguous()
        classify_h_n = classify_h_n.permute(1, 0, 2)

        return self.classify_linear(classify_h_n.reshape(-1, 200))
