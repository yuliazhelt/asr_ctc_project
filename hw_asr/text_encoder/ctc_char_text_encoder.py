from typing import List, NamedTuple

import torch

from .char_text_encoder import CharTextEncoder
from ctcdecoder import beam_search


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

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

        alphabet = ''.join(self.ind2char.values())
        probs = probs.exp()
        # probs = probs.cpu().detach().numpy()
        print(beam_search(probs, alphabet, beam_size=beam_size, lm_model=self.lm, lm_alpha=0.2, lm_beta=0.0001))
        res = beam_search(probs, alphabet, beam_size=beam_size, lm_model=self.lm, lm_alpha=0.2, lm_beta=0.0001)[0]
        return res


