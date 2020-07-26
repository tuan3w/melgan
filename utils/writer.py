from tensorboardX import SummaryWriter
from utils.stft import TacotronSTFT
from .plotting import plot_waveform_to_numpy, plot_spectrogram_to_numpy
import torch

class MyWriter(SummaryWriter):
    def __init__(self, hp, logdir):
        super(MyWriter, self).__init__(logdir)
        self.sample_rate = hp.audio.sampling_rate
        self.stft = TacotronSTFT(filter_length=hp.audio.filter_length,
                            hop_length=hp.audio.hop_length,
                            win_length=hp.audio.win_length,
                            n_mel_channels=hp.audio.n_mel_channels,
                            sampling_rate=hp.audio.sampling_rate,
                            mel_fmin=hp.audio.mel_fmin,
                            mel_fmax=hp.audio.mel_fmax)
        self.is_first = True

    def log_training(self, loss, sc_loss, mag_loss, phase_loss, step):
        self.add_scalar('train.loss', loss, step)
        self.add_scalar('train.sc_loss', sc_loss, step)
        self.add_scalar('train.mag_loss', mag_loss, step)
        self.add_scalar('train.phase_loss', phase_loss, step)

    def log_audio(self, predict, target , step):
        self.add_audio('predicted_audio', predict, step, self.sample_rate)
        self.add_audio('target_audio', target, step, self.sample_rate)
        wav = torch.from_numpy(predict).unsqueeze(0)
        mel = self.stft.mel_spectrogram(wav)  # mel [1, num_mel, T]
        self.add_image('melspectrogram_prediction', plot_spectrogram_to_numpy(mel.squeeze(0).data.cpu().numpy()),
                       step, dataformats='HWC')

    def log_histogram(self, model, step):
        for tag, value in model.named_parameters():
            self.add_histogram(tag.replace('.', '/'), value.cpu().detach().numpy(), step)
