from torch import nn

from hw_tts.model.FFTBlock import FFTBlock
from hw_tts.model.nn_blocks import get_attn_key_pad_mask, get_non_pad_mask


class Decoder(nn.Module):
    """ Decoder """

    def __init__(self, model_config):

        super(Decoder, self).__init__()

        len_max_seq = model_config["max_seq_len"]
        n_position = len_max_seq + 1
        n_layers = model_config["decoder_n_layer"]

        self.position_enc = nn.Embedding(
            n_position,
            model_config["encoder_dim"],
            padding_idx=model_config["PAD"],
        )

        self.layer_stack = nn.ModuleList([FFTBlock(
            model_config["encoder_dim"],
            model_config["encoder_conv1d_filter_size"],
            model_config["encoder_head"],
            model_config["encoder_dim"] // model_config["encoder_head"],
            model_config["encoder_dim"] // model_config["encoder_head"],
            dropout=model_config["dropout"]
        ) for _ in range(n_layers)])

    def forward(self, enc_seq, enc_pos, return_attns=False):

        dec_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=enc_pos, seq_q=enc_pos)
        non_pad_mask = get_non_pad_mask(enc_pos)

        pos = self.position_enc(enc_pos.long())
        # -- Forward
        dec_output = enc_seq + self.position_enc(enc_pos.long())

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output
