from typing import List, NamedTuple

import torch

from .char_text_encoder import CharTextEncoder
from pyctcdecode import build_ctcdecoder
import kenlm

import os
import wget

import numpy as np
import scipy

class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None, lm_path : str = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        # self.lm_model = None
        # if lm_path:
        #     self.lm = kenlm.LanguageModel(os.path.join(lm_path))
        labels = ''.join(self.ind2char.values())

        kenlm_model_path = '3-gram.pruned.1e-7.arpa.gz'
        if not os.path.exists(kenlm_model_path):
            print('Downloading pruned 3-gram model.')
            lm_url = 'http://www.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz'
            kenlm_model_path = wget.download(lm_url)
            print('Downloaded the 3-gram language model.')
        else:
            print('Pruned .arpa.gz already exists.')

        self.decoder = build_ctcdecoder(
            labels,
            kenlm_model_path,
            alpha=0.5,  # tuned on a val set 
            beta=1.0,  # tuned on a val set 
        )


    def ctc_decode(self, inds: List[int]) -> str:
        decoded_list = []
        last_char = self.EMPTY_TOK
        for ind in inds:
            curr_char = self.ind2char[ind]
            if curr_char == self.EMPTY_TOK:
                last_char = curr_char
                continue
            if curr_char != last_char:
                decoded_list.append(curr_char)
            last_char = curr_char
        return "".join(decoded_list)

    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos: List[Hypothesis] = []

        decoded_beams = self.decoder.decode_beams(probs[:probs_length].detach().numpy(), beam_width=beam_size)
        return [Hypothesis(text=decoded_beams[i][0], prob=np.exp(decoded_beams[i][3])) for i in range(len(decoded_beams))]
