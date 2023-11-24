import torch
from torch import nn

from hw_tts.model.FFTBlock import FFTBlock
from hw_tts.model.nn_blocks import get_attn_key_pad_mask, get_non_pad_mask


class Encoder(nn.Module):
    def __init__(self, model_config):
        super(Encoder, self).__init__()

        len_max_seq = model_config["max_seq_len"]
        n_position = len_max_seq + 1
        n_layers = model_config["encoder_n_layer"]
        self.padding_idx = model_config["PAD"]

        self.src_word_emb = nn.Embedding(
            model_config["vocab_size"],
            model_config["encoder_dim"],
            padding_idx=model_config["PAD"]
        )

        self.position_enc = nn.Embedding(
            n_position,
            model_config["encoder_dim"],
            padding_idx=model_config["PAD"]
        )

        self.layer_stack = nn.ModuleList([FFTBlock(
            model_config["encoder_dim"],
            model_config["encoder_conv1d_filter_size"],
            model_config["encoder_head"],
            model_config["encoder_dim"] // model_config["encoder_head"],
            model_config["encoder_dim"] // model_config["encoder_head"],
            dropout=model_config["dropout"]
        ) for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq, padding_idx=self.padding_idx)
        non_pad_mask = get_non_pad_mask(src_seq, padding_idx=self.padding_idx)

        # -- Forward
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        return enc_output, non_pad_mask

