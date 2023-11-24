import torch
import torch.nn as nn


class FastSpeechLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.duration_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.pitch_loss = nn.MSELoss()
        self.energy_loss = nn.MSELoss()

    def forward(self, mel_predicted, duration_predicted, mel_target, duration,
                pitch_predicted, pitch_target, energy_predicted, energy_target, **batch):
        mel_loss = self.l1_loss(mel_predicted, mel_target)

        duration_predictor_loss = self.duration_loss(duration_predicted,
                                                     duration.float())

        pitch_predictor_loss = self.pitch_loss(pitch_predicted,
                                               pitch_target)

        energy_predictor_loss = self.energy_loss(energy_predicted,
                                                 energy_target)

        return mel_loss + duration_predictor_loss + pitch_predictor_loss + energy_predictor_loss
