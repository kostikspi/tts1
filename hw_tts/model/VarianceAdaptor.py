import torch
import torch.nn as nn
import torch.nn.functional as F

from hw_tts.model.nn_blocks import LengthRegulator


class Predictor(nn.Module):
    def __init__(self, model_config):
        super(Predictor, self).__init__()
        self.input_size = model_config["encoder_dim"]
        self.filter_size = model_config["pitch_predictor_filter_size"]
        self.kernel = model_config["pitch_predictor_kernel_size"]
        self.conv_output_size = model_config["pitch_predictor_filter_size"]
        self.dropout = 0.5
        self.conv1 = nn.Conv1d(self.input_size, self.filter_size,
                               kernel_size=self.kernel,
                               padding=(self.kernel - 1) // 2)
        self.ln1 = nn.LayerNorm(self.filter_size)
        self.dropout1 = nn.Dropout(self.dropout)
        self.conv2 = nn.Conv1d(self.filter_size, self.filter_size,
                               kernel_size=self.kernel,
                               padding=(self.kernel - 1) // 2)
        self.ln2 = nn.LayerNorm(self.filter_size)
        self.dropout2 = nn.Dropout(self.dropout)
        self.fc = nn.Linear(self.filter_size, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x.mT)).mT
        x = self.ln1(x)
        x = self.dropout1(x)
        x = F.relu(self.conv2(x.mT)).mT

        x = self.ln2(x)
        x = self.dropout2(x)
        x = self.fc(x)
        return x


class VarianceAdaptor(nn.Module):
    def __init__(self, model_config):
        super(VarianceAdaptor, self).__init__()
        self.length_regulator = LengthRegulator(model_config)
        self.pitch_predictor = Predictor(model_config)
        self.n_bins = 256
        self.p_min = 0
        self.p_max = 200
        self.pitch_bins = torch.exp(torch.linspace(torch.log(torch.tensor(self.p_min)).item(),
                                                   torch.log(torch.tensor(self.p_max)).item(),
                                                   self.n_bins - 1))

        self.pitch_embedding = nn.Embedding(
            self.n_bins,
            model_config["encoder_dim"]
        )

        self.e_min = 0
        self.e_max = 200
        self.energy_predictor = Predictor(model_config)
        self.energy_bins = torch.exp(torch.linspace(torch.log(torch.tensor(self.e_min)).item(),
                                                    torch.log(torch.tensor(self.e_max)).item(),
                                                    self.n_bins - 1))
        self.energy_embedding = nn.Embedding(
            self.n_bins,
            model_config["encoder_dim"]
        )

    def forward(self, x, alpha=1.0, duration=None, mel_max_length=None, pitch_target=None, energy_target=None):
        x, duration_predicted = self.length_regulator(x, alpha, duration, mel_max_length)

        pitch = self.pitch_predictor(x)

        pitch_predicted = torch.bucketize(pitch, self.pitch_bins)

        if pitch_target is not None:
            pitch_embedding = self.pitch_embedding(torch.bucketize(pitch_target, self.pitch_bins))
        else:
            pitch_embedding = self.pitch_embedding(pitch_predicted)

        x = x + pitch_embedding[:, :x.shape[1], :] * alpha

        energy = self.energy_predictor(x)

        if energy_target is not None:
            energy_embedding = self.energy_embedding(torch.bucketize(energy_target, self.pitch_bins))
        else:
            energy_embedding = self.energy_embedding(torch.bucketize(energy, self.energy_bins))

        x = x + energy_embedding[:, :x.shape[1], :] * alpha

        return x, duration_predicted, pitch, energy
