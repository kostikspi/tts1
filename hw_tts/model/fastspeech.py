import torch
import torch.nn as nn

from hw_tts.model.Decoder import Decoder
from hw_tts.model.Encoder import Encoder
from hw_tts.model.nn_blocks import LengthRegulator
from hw_tts.model.VarianceAdaptor import VarianceAdaptor


def get_mask_from_lengths(lengths, max_len=None):
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len, 1, device=lengths.device)
    mask = (ids < lengths.unsqueeze(1)).bool()

    return mask


def mask_tensor(mel_output, position, mel_max_length):
    lengths = torch.max(position, -1)[0]
    mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
    mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
    return mel_output.masked_fill(mask, 0.)


class FastSpeech(nn.Module):
    """ FastSpeech """

    def __init__(self, model_config):
        super(FastSpeech, self).__init__()

        self.encoder = Encoder(model_config)
        # self.length_regulator = LengthRegulator(model_config)
        self.variance_adaptor = VarianceAdaptor(model_config)
        self.decoder = Decoder(model_config)

        self.mel_linear = nn.Linear(model_config["decoder_dim"], model_config["num_mels"])

    def forward(self, src_seq, src_pos, mel_pos=None, mel_max_length=None, duration=None, pitch_target=None,
                energy_target=None, alpha=1.0,
                *args,
                **kwargs):

        print(duration.sum())
        print(mel_pos.shape)
        enc_output, non_pad_mask = self.encoder(src_seq, src_pos, return_attns=True)

        # enc_output, duration_predicted = self.length_regulator(enc_output, alpha=1,
        #                                                        mel_max_length=mel_max_length,
        #                                                        duration=duration)

        (enc_output, duration_predicted,
         pitch_predicted, energy_predicted) = self.variance_adaptor(enc_output, mel_max_length=mel_max_length,
                                                                    pitch_target=pitch_target,
                                                                    energy_target=energy_target)
        if mel_pos is None:
            mel_pos = duration_predicted

        output = self.decoder(enc_output, mel_pos)

        output = self.mel_linear(output)

        if mel_max_length is not None:
            output = mask_tensor(output, mel_pos, mel_max_length)

        return output, duration_predicted, pitch_predicted, energy_predicted
