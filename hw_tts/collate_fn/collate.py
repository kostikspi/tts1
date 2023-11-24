import logging
from typing import List
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(x, (0, length - x.shape[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_1D_tensor(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = F.pad(x, (0, length - x.shape[0]))
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = torch.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(x, (0, max_len - np.shape(x)[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad_2D_tensor(inputs, maxlen=None):
    def pad(x, max_len):
        if x.size(0) > max_len:
            raise ValueError("not max_len")

        s = x.size(1)
        x_padded = F.pad(x, (0, 0, 0, max_len - x.size(0)))
        return x_padded[:, :s]

    if maxlen:
        output = torch.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(x.size(0) for x in inputs)
        output = torch.stack([pad(x, max_len) for x in inputs])

    return output


def process_text(train_text_path):
    with open(train_text_path, "r", encoding="utf-8") as f:
        txt = []
        for line in f.readlines():
            txt.append(line)

        return txt


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = {"mel_target": [], "text": [], "duration": [],
                    "mel_pos": [], "src_pos": [], "mel_max_len": 0,
                    "audio_path": [], 'pitch_target': [], 'energy_target': []}

    for item in dataset_items:
        result_batch["mel_target"].append(item["mel_target"].squeeze(0).T)
        result_batch["text"].append(torch.tensor(item["text"]))
        result_batch["duration"].append(torch.tensor(item["duration"]))
        result_batch["audio_path"].append(item["audio_path"])
        result_batch["pitch_target"].append(item["pitch_target"].squeeze(0))
        result_batch["energy_target"].append(item["energy_target"].squeeze(0))

    length_text = np.array([])

    for text in result_batch["text"]:
        length_text = np.append(length_text, text.shape[0])

    src_pos = list()

    max_len = int(max(length_text))
    for length_src_row in length_text:
        src_pos.append(np.pad([i + 1 for i in range(int(length_src_row))],
                              (0, max_len - int(length_src_row)), 'constant'))
    src_pos = torch.from_numpy(np.array(src_pos))

    length_mel = np.array(list())
    for mel in result_batch["mel_target"]:
        length_mel = np.append(length_mel, mel.size(0))

    mel_pos = list()
    max_mel_len = int(max(length_mel))
    for length_mel_row in length_mel:
        mel_pos.append(np.pad([i + 1 for i in range(int(length_mel_row))],
                              (0, max_mel_len - int(length_mel_row)), 'constant'))
    mel_pos = torch.from_numpy(np.array(mel_pos))

    result_batch['src_seq'] = pad_1D_tensor(result_batch["text"])
    result_batch['duration'] = pad_1D_tensor(result_batch['duration'])
    result_batch['mel_target'] = pad_2D_tensor(result_batch['mel_target'])
    result_batch['pitch_target'] = pad_1D_tensor(result_batch['pitch_target'])
    result_batch['energy_target'] = pad_1D_tensor(result_batch['energy_target'])

    result_batch['src_pos'] = src_pos
    result_batch['mel_pos'] = mel_pos

    result_batch['mel_max_length'] = max_mel_len

    return result_batch
