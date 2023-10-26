import logging
from typing import List
from collections import defaultdict

import torch
from torch.nn.utils.rnn import pad_sequence
import random
logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    result_batch_list = defaultdict(list)
    result_batch = {}
    
    for item in dataset_items:
        result_batch_list['spectrogram'].append(item['spectrogram'].squeeze(0).T)
        result_batch_list['text_encoded'].append(item['text_encoded'].squeeze(0).T)
        
        result_batch_list['text_encoded_length'].append(item['text_encoded'].shape[1])
        result_batch_list['spectrogram_length'].append(item['spectrogram'].shape[2])

        result_batch_list['text'].append(item['text'])
        result_batch_list['audio_path'].append(item['audio_path'])

    result_batch['audio_sample'] = random.choice(dataset_items)['audio'].cpu()
    
    result_batch['text_encoded_length'] = torch.tensor(result_batch_list['text_encoded_length'])
    result_batch['spectrogram_length'] = torch.tensor(result_batch_list['spectrogram_length'])
    
    result_batch['spectrogram'] = pad_sequence(result_batch_list['spectrogram'], batch_first=True).transpose(1, 2)
    result_batch['text_encoded'] = pad_sequence(result_batch_list['text_encoded'], batch_first=True)
    result_batch['text'] = result_batch_list['text']
    result_batch['audio_path'] = result_batch_list['audio_path']

    return result_batch