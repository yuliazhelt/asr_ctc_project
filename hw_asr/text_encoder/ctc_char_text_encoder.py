from typing import List, NamedTuple

import torch

from .char_text_encoder import CharTextEncoder
from pyctcdecode import build_ctcdecoder
import kenlm
from collections import defaultdict

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

        kenlm_model_path = 'lowercase_3-gram.pruned.1e-7.arpa.gz'
        unigram_path = 'librispeech-vocab.txt'
        if not os.path.exists(kenlm_model_path):
            print('Downloading pruned 3-gram model.')
            lm_url = 'http://www.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz'
            kenlm_model_path = wget.download(lm_url)
            print('Downloaded the 3-gram language model.')
        else:
            print('Pruned .arpa.gz already exists.')
        if not os.path.exists(unigram_path):
            print('Downloading unigram list.')
            unigram_url = 'http://www.openslr.org/resources/11/librispeech-vocab.txt'
            unigram_path = wget.download(unigram_url)
            print('Downloaded the unigram list.')
        else:
            print('unigram list already exists.')

        with open(unigram_path) as f:
            unigram_list = [t.lower() for t in f.read().strip().split("\n")]

        self.decoder = build_ctcdecoder(
            labels,
            kenlm_model_path,
            unigram_list,
            alpha=0.7,  # tuned on a val set 
            beta=3.0,  # tuned on a val set
        )

    def ctc_decode_text(self, text: str) -> str:
        decoded_list = []
        last_char = self.EMPTY_TOK
        for curr_char in text:
            if curr_char == self.EMPTY_TOK:
                last_char = curr_char
                continue
            if curr_char != last_char:
                decoded_list.append(curr_char)
            last_char = curr_char
        return "".join(decoded_list)

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

        decoded_beams = self.decoder.decode_beams(probs[:probs_length].detach().numpy(), beam_width=beam_size, token_min_logp=-1)
        return [Hypothesis(text=decoded_beams[i][0], prob=decoded_beams[i][3]) for i in range(len(decoded_beams))]

    def ctc_beam_search_custom(self, probs: torch.tensor, probs_length, beam_size: int = 100):
        state = {('',  self.EMPTY_TOK): 1.0}
        for frame in probs:
            state = self.extend_and_merge(frame, state)
            state = self.truncate(state, beam_size)
        return [Hypothesis(text=v[0][0], prob=v[-1]) for v in list(state.items())]

    def extend_and_merge(self, frame, state):
        new_state = defaultdict(float)
        for next_char_index, next_char_proba in enumerate(frame):
            for (pref, last_char), pref_proba in state.items():
                next_char = self.ind2char[next_char_index]
                if next_char == last_char:
                    new_pref = pref
                else:
                    if next_char != self.EMPTY_TOK:
                        new_pref = pref + next_char
                    else:
                        new_pref = pref
                last_char = next_char
                new_state[(new_pref, last_char)] += pref_proba * next_char_proba
        return new_state

    def truncate(self, state, beam_size):
        state_list = list(state.items())
        state_list.sort(key=lambda x: -x[1])
        return dict(state_list[:beam_size])