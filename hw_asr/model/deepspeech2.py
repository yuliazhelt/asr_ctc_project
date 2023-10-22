import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from hw_asr.model.gru_model import GRULayer

class ConvDS2(nn.Module):

    def __init__(
            self
    ):
        super(ConvDS2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20),
        )


    def forward(self, spectrogram : Tensor,  spectrogram_length : Tensor, *args, **kwargs):
        outputs = self.conv(spectrogram.unsqueeze(1))

        bs, ch, n_feats_conv, length = outputs.size()
        outputs = outputs.permute(0, 3, 1, 2)
        outputs = outputs.view(bs, length, ch * n_feats_conv)

        return outputs, self.transform_input_lengths(spectrogram_length)

    def transform_input_lengths(self, input_lengths: Tensor) -> Tensor:
        """
        Input length transformation function.
        For example: if your NN transforms spectrogram of time-length `N` into an
        output with time-length `N / 2`, then this function should return `input_lengths // 2`
        """
        input_lengths = ((input_lengths - 11) / 2 + 1).floor()
        input_lengths = (input_lengths - 11) + 1
        return input_lengths

    def get_output_dim(self, n_feats : int) -> int:
        """
        n_feats transformation function for conv layer output
        """
        n_feats_out = int((n_feats - 41) / 2 + 1)
        n_feats_out = int((n_feats_out - 21) / 2 + 1)
        return 32 * n_feats_out


class DeepSpeech2(nn.Module):

    def __init__(
            self,
            n_feats: int,
            n_class: int,
            num_rnn_layers : int,
            hidden_size: int = 512,
            bidirectional: bool = False,
            dropout_p: float = 0.1,
    ):
        super(DeepSpeech2, self).__init__()
        self.conv = ConvDS2()

        self.gru_layers = nn.ModuleList()
        gru_output_size = hidden_size * 2 if bidirectional else hidden_size

        for idx in range(num_rnn_layers):
            self.gru_layers.append(
                GRULayer(
                    n_feats=gru_output_size if idx != 0 else self.conv.get_output_dim(n_feats=n_feats), #? conv output
                    hidden_size=hidden_size,
                    bidirectional=bidirectional,
                    dropout_p=dropout_p,
                )
            )

        self.head = nn.Sequential(
            nn.Linear(in_features=gru_output_size, out_features=n_class)
        )

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)

    def forward(self, spectrogram : Tensor, spectrogram_length : Tensor, *args, **kwargs):
        outputs, outputs_length = self.conv(spectrogram, spectrogram_length)
        outputs = outputs.transpose(1, 2)
        for gru_layer in self.gru_layers:
            outputs = gru_layer(outputs, outputs_length)

        outputs = self.head(outputs.transpose(1, 2))

        return outputs

    def transform_input_lengths(self, input_lengths):
        """
        Input length transformation function.
        For example: if your NN transforms spectrogram of time-length `N` into an
            output with time-length `N / 2`, then this function should return `input_lengths // 2`
        """
        return input_lengths