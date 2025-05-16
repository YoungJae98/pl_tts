import os
import argparse
import torch
import logging
import torch.nn.functional as F

from torch_stoi import NegSTOILoss
from vits.utils.mel_processing import mel_spectrogram_torch, spectrogram_torch
from vits.utils import utils


def get_spec(hps, waves, waves_len):
    spec_np = []
    spec_lengths = torch.LongTensor(len(waves))

    for index, wave in enumerate(waves):
        audio_norm = wave[:, :waves_len[index]]
        # print(audio_norm.shape)
        spec = spectrogram_torch(audio_norm,
                                 hps.filter_length, hps.sampling_rate,
                                 hps.hop_length, hps.win_length,
                                 center=False)
        
        spec = torch.squeeze(spec, 0)
        spec_np.append(spec)
        spec_lengths[index] = spec.size(1)

    max_spec_len = max(spec_lengths)
    spec_padded = torch.zeros(
        len(waves),
        spec_np[0].size(0),
        max_spec_len,
        device=waves.device,
        dtype=spec_np[0].dtype
    )
    for i, spec in enumerate(spec_np):
        spec_padded[i, :, :spec_lengths[i]] = spec
    # spec_padded = torch.FloatTensor(len(waves), spec_np[0].size(0), max_spec_len)
    # spec_padded.zero_()

    # for i, spec in enumerate(waves):
    #     spec_padded[i][:, :spec_lengths[i]] = spec_np[i]
    
    # print(spec_lengths)

    return spec_padded, spec_lengths


def save_noise(hps, u_noise, step):
    path = f"{hps.model_dir}/univ_noises_{step}.pt"
    torch.save(u_noise.detach().cpu(), path)

def tile_and_shift(u_noise: torch.Tensor, target_len: int):
    """repeat u_noise until >= target_len, then crop & random circular-shift"""
    patch_len = u_noise.shape[-1]
    reps = (target_len + patch_len - 1) // patch_len
    n = u_noise.repeat_interleave(reps, dim=-1)[..., :target_len]  # [1,1,T]
    shift = torch.randint(0, patch_len, ()).item()
    return torch.roll(n, shifts=shift, dims=-1)                    # [1,1,T]

def apply_univ_noise(wav, u_noise, snr_db=15., percentile=.6, beta=.1):
    """
    wav     : [B, 1, T]  (–1~1 범위)
    u_noise : [1, 1, P]  (universal patch, P ≪ T)
    """
    batch_noisy = []
    for w in wav:                                   # loop over batch
        n = tile_and_shift(u_noise, w.shape[-1])    # 길이 맞추고 shift
        gate = energy_gate(w.unsqueeze(0), percentile=percentile, beta=beta)     # energy gate
        n = n * gate                                # 저에너지 억제
        n = match_local_snr(w.unsqueeze(0), n, snr_db=snr_db)
        w_noisy = torch.clamp(w + n.squeeze(0), -1., 1.)
        batch_noisy.append(w_noisy)

    return torch.stack(batch_noisy, dim=0)          # [B,1,T]


def energy_gate(wav: torch.Tensor,
                frame_len: int = 400,   # 25 ms @16 kHz
                hop: int = 160,         # 10 ms
                percentile: float = .6,
                beta: float = .1):
    """
    Soft 0-1 gate based on frame energy.
    0 : 저에너지 구간, 1 : 고에너지(말소리) 구간
    """
    len_w = wav.size(-1)
    frames = wav.unfold(-1, frame_len, hop)      # [..., F, frame_len]
    E = frames.pow(2).mean(-1)                   # [..., F]
    T = torch.quantile(E, percentile, dim=-1, keepdim=True)
    w = torch.sigmoid((E - T) / (beta * T))      # [..., F] ∈ (0,1)
    

    # [1, F] → [1,1,F] → [1,1,len_w]
    # gate = w.unsqueeze(1)
    # print(gate.shape)
    gate = F.interpolate(w, size=len_w, mode='linear', align_corners=False)
    return gate.clamp(0, 1).unsqueeze(1)


    # gate = F.pad(w, (0, 1)).repeat_interleave(hop, dim=-1)[..., :len_w]

    # return gate.unsqueeze(1)  # [..., 1, len_w]


def match_local_snr(wav: torch.Tensor, noise: torch.Tensor, snr_db: float = 15.):
    """프레임 단위로 target SNR에 맞춰 noise 스케일"""
    pow_sig = wav.pow(2).mean(-1, keepdim=True)
    pow_noise = noise.pow(2).mean(-1, keepdim=True) + 1e-12
    alpha = torch.sqrt(pow_sig / (pow_noise * 10**(snr_db / 10)))
    return noise * alpha


def compute_kl_divergence(hps, x_hat, z):
    '''
        Return the KL-divergence loss of the input distributions.
    '''
    x_mel = mel_spectrogram_torch(
        x_hat.squeeze(1).float(),
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.mel_fmin,
        hps.data.mel_fmax
    )
    z_mel = mel_spectrogram_torch(
        z.squeeze(1).float(),
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.mel_fmin,
        hps.data.mel_fmax
    )

    p_log = F.log_softmax(x_mel, dim=-1)
    q = F.softmax(z_mel, dim=-1)

    kl_divergence = F.kl_div(p_log, q, reduction="batchmean")

    return kl_divergence


def compute_reconstruction_loss(hps, wav, wav_hat):
    '''
        Return the mel loss of the real and synthesized speech.
    '''
    wav_mel = mel_spectrogram_torch(
        wav.squeeze(1).float(),
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.mel_fmin,
        hps.data.mel_fmax
    )
    wav_hat_mel = mel_spectrogram_torch(
        wav_hat.squeeze(1).float(),
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.mel_fmin,
        hps.data.mel_fmax
    )
    loss_mel_wav = F.l1_loss(wav_mel, wav_hat_mel) * hps.train.c_mel

    return loss_mel_wav



def compute_stoi(sample_rate, waveforms, perturb_waveforms):
    '''
        Return the STOI loss of the clean and protected speech
    '''
    device = waveforms.device
    stoi_function = NegSTOILoss(sample_rate=sample_rate).to(device)

    loss_stoi = stoi_function(waveforms, perturb_waveforms).mean()
    return loss_stoi


def compute_stft(waveforms, perturb_waveforms):
    '''
        Return the STFT loss with L_2 norm of the clean and protected speech
    '''
    stft_clean = torch.stft(waveforms, n_fft=2048, win_length=2048, hop_length=512, return_complex=False)
    stft_p = torch.stft(perturb_waveforms, n_fft=2048, win_length=2048, hop_length=512, return_complex=False)
    loss_stft = torch.norm(stft_p - stft_clean, p=2)

    return loss_stft


def compute_perceptual_loss(hps, p_wav, wav):
    '''
        Return the proposed perceptual loss  of the clean and protected speech
    '''
    loss_stoi = compute_stoi(hps.data.sampling_rate, wav, p_wav)
    loss_stft = compute_stft(wav.squeeze(1), p_wav.squeeze(1))
    loss_perceptual = loss_stoi + loss_stft

    return loss_perceptual

def get_hparams(init=True):
  parser = argparse.ArgumentParser()


  parser.add_argument('-c', '--config', type=str, default="vits/configs/vits_base.json",help='JSON file for configuration')
  parser.add_argument('-tf', '--training_files', type=str, required=True,help='dataset name')
  parser.add_argument('-me', '--max_epochs', type=int, required=True,help='epochs')
  parser.add_argument('-bs', '--batch_size', type=int, required=True,help='batch size')
  parser.add_argument('-m', '--model', type=str, required=True,help='Model name')
  parser.add_argument("-path", "--pretrained_path", type=str, required=True, help="The checkpoint path of the pre-trained model.")
  parser.add_argument('-ns', '--n_speakers', type=int, required=True,help='batch size')
  parser.add_argument('-a', '--alpha', type=float, default=0.05, help="alpha for perceptual loss")
  parser.add_argument('-lr', '--learning_rate', type=float, default=5e-2 ,help='learning rate')
  parser.add_argument("-ep", "--epsilon", type=float, default=8, help="The protective radius of the embedded perturbation by l_p norm.")
  parser.add_argument("--snr_db", type=float, default=15.0)
  parser.add_argument("--percentile", type=float, default=0.6)
  parser.add_argument("--beta", type=float, default=0.1)
  parser.add_argument("--lambda_pen", type=float, default=1e-3)

  args = parser.parse_args()

  hparams = utils.get_hparams_from_file(args.config)
  model_dir = os.path.join("noise_pop/checkpoints", args.model)

  hparams.model_dir = model_dir
  hparams.config_path = args.config
  hparams.train.training_files = args.training_files
  hparams.train.learning_rate = args.learning_rate
  hparams.train.max_epochs = args.max_epochs
  hparams.train.batch_size = args.batch_size
  hparams.train.pretrained_path = args.pretrained_path
  hparams.train.epsilon = args.epsilon
  hparams.train.snr_db = args.snr_db
  hparams.train.percentile = args.percentile
  hparams.train.alpha = args.alpha
  hparams.train.beta = args.beta
  hparams.train.lambda_pen = args.lambda_pen
  hparams.data.n_speakers = args.n_speakers
  
  torch.manual_seed(hparams.train.seed)
  torch.cuda.manual_seed(hparams.train.seed)

  return hparams


def get_logger(model_dir, filename="train.log"):
  global logger
  logger = logging.getLogger(os.path.basename(model_dir))
  logger.setLevel(logging.DEBUG)
  
  formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  h = logging.FileHandler(os.path.join(model_dir, filename))
  h.setLevel(logging.DEBUG)
  h.setFormatter(formatter)
  logger.addHandler(h)
  return logger