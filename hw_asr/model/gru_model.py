import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

class GRULayer(nn.Module):

    def __init__(
            self,
            n_feats: int,
            hidden_size: int = 512,
            bidirectional: bool = False,
            dropout_p: float = 0.1,
    ):
        super(GRULayer, self).__init__()
        self.hidden_size = hidden_size
        self.batch_norm = nn.BatchNorm1d(n_feats)
        self.gru = nn.GRU(
            input_size=n_feats,
            hidden_size=hidden_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=bidirectional,
        )

    def forward(self, spectrogram : Tensor, spectrogram_length : Tensor, *args, **kwargs):

        spectrogram = F.relu(self.batch_norm(spectrogram))
        spectrogram = spectrogram.transpose(1, 2)

        outputs = pack_padded_sequence(spectrogram, spectrogram_length, batch_first=True, enforce_sorted=False)
        outputs, hidden_states = self.gru(outputs)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)

        return outputs


class GRUModel(nn.Module):

    def __init__(
            self,
            n_feats: int,
            n_class: int,
            num_rnn_layers : int,
            hidden_size: int = 512,
            bidirectional: bool = False,
            dropout_p: float = 0.1,
    ):
        super(GRUModel, self).__init__()
        self.gru_layers = nn.ModuleList()

        for idx in range(num_rnn_layers):
            self.gru_layers.append(
                GRULayer(
                    n_feats=n_feats,
                    hidden_size=hidden_size,
                    bidirectional=bidirectional,
                    dropout_p=dropout_p,
                )
            )

        self.head = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=n_class)
        )

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)

    def forward(self, spectrogram : Tensor, spectrogram_length : Tensor, *args, **kwargs):

        outputs, output_lengths = spectrogram, spectrogram_length

        for gru_layer in self.gru_layers:
            outputs = gru_layer(outputs, output_lengths)

        outputs = self.head(outputs)

        return outputs

    def transform_input_lengths(self, input_lengths):
        """
        Input length transformation function.
        For example: if your NN transforms spectrogram of time-length `N` into an
            output with time-length `N / 2`, then this function should return `input_lengths // 2`
        """
        return input_lengths