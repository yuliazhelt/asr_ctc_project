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
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(2, 2)),
            nn.BatchNorm2d(32)
        )


    def forward(self, spectrogram : Tensor, spectrogram_length : Tensor, *args, **kwargs):
        print(spectrogram.unsqueeze(1).shape)
        outputs = self.conv(spectrogram.unsqueeze(1))

        print(outputs.shape)
        bs, ch, n_feats_conv, length = outputs.size()
        outputs = outputs.permute(0, 3, 1, 2)
        outputs = outputs.view(bs, length, ch * n_feats_conv)

        self.output_dim = ch * n_feats_conv

        print(outputs.shape)
        
        outputs = F.hardtanh(F.relu(outputs), 0, 20)

        return outputs


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
        self.linear = nn.Linear(in_features=n_feats, out_features=hidden_size)
        self.conv = ConvDS2()

        self.gru_layers = nn.ModuleList()
        gru_output_size = hidden_size << 1 if bidirectional else hidden_size

        for idx in range(num_rnn_layers):
            self.gru_layers.append(
                GRULayer(
                    n_feats=gru_output_size if idx != 0 else 32, #?
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

        outputs = self.linear(spectrogram.transpose(1, 2)).transpose(1, 2)
        outputs = F.hardtanh(F.relu(outputs), 0, 20)

        outputs = self.conv(outputs)

        for gru_layer in self.gru_layers:
            outputs = gru_layer(outputs, spectrogram_length)

        outputs = self.head(outputs.transpose(1, 2))

        return outputs

    def transform_input_lengths(self, input_lengths):
        """
        Input length transformation function.
        For example: if your NN transforms spectrogram of time-length `N` into an
            output with time-length `N / 2`, then this function should return `input_lengths // 2`
        """
        return input_lengths