"""
universal_protect.py
────────────────────
• “짧은 패치 노이즈” 하나( u_noise )만 학습
• 패치를 tile-&-shift → 음성 길이에 맞춰 적용
• 에너지-기반 gate + 저에너지 패널티로
  - 청감 노출 ↓  - 무음-편향 ↓
────────────────────
사용 예)
CUDA_VISIBLE_DEVICES=0 python universal_protect.py \
  --config_path dataset/parastar_24000/config.json \
  --save_name parastar_univ --patch_sec 0.5 --epsilon 1e-4
"""

import torch
from torch.utils.data import DataLoader
from torch.optim import SGD

from vits.utils.data_utils import TextAudioSpeakerLoader, TextAudioSpeakerCollate
from vits.model.models import SynthesizerTrn
from vits.text.symbols import symbols
from vits.utils import utils
from vits.model import commons

from noise_pop.utils import energy_gate, mel_recon_loss, apply_univ_noise, save_noise, get_spec

import pytorch_lightning as pl

# ──────────────────────────────────────────────────────────────────────────
# Main --------------------------------------------------------------------
# ──────────────────────────────────────────────────────────────────────────

class UnivNoiseModule_ver1(pl.LightningModule):
    def __init__(
        self,
        hps,
        logger,
        patch_sec: float = 0.1,
    ):
        super().__init__()
        self.hps = hps

        # build VITS generator (frozen)
        self.net_g = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        )

        _, _, _, _ = utils.load_checkpoint(
            utils.latest_checkpoint_path(self.hps.train.pretrained_path, "G_*.pth"),
            self.net_g
        )
        self.net_g.eval()
        for p in self.net_g.parameters():
            p.requires_grad = False

        # learnable universal noise
        patch_len = int(hps.data.sampling_rate * patch_sec)
        self.u_noise = torch.nn.Parameter(torch.zeros(1, 1, patch_len))
        self.logger_ = logger
        self.per_log_step = 100
        self.per_save_step = 2000


    def training_step(self, batch, batch_idx):
        text, text_len, _, _, wav, wav_len, spk = batch

        p_wav = apply_univ_noise(wav, self.u_noise, self.hps.train.snr_db, self.hps.train.percentile, self.hps.train.beta)


        with torch.no_grad():
            p_spec, p_len = get_spec(self.hps.data, p_wav, wav_len)
            p_spec = p_spec.half().cuda()
            p_len = p_len.half().cuda()

            wav_hat, l_length, attn, ids_slice, x_mask, z_mask, \
                (z, z_p, m_p, logs_p, m_q, logs_q) = self.net_g(text, text_len, p_spec, p_len, spk,
                                                        is_fixed=True, is_clip=True)

            if ids_slice is not None:
                p_wav_slice = commons.slice_segments(p_wav.squeeze(1), ids_slice * self.hps.data.hop_length, self.hps.train.segment_size)
                
            else:
                p_wav_slice = p_wav

            # wav_hat, _, *_ = self.net_g(text, text_len, p_spec, p_len, spk, is_fixed=True, is_clip=True)

        loss_adv = mel_recon_loss(self.hps, wav_hat, p_wav_slice)
        gate = energy_gate(wav, percentile=self.hps.train.percentile, beta=self.hps.train.beta)
        loss_pen = self.hps.train.lambda_pen * torch.mean(((1 - gate) * (p_wav - wav)) ** 2)
        loss = loss_adv + loss_pen
        self.losses = [loss_adv, loss_pen]
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.global_rank == 0:
            if self.global_step % self.per_log_step == 0:
                self.logger_.info(f"Epoch:{self.current_epoch} / Step:{self.global_step} / Losses:{[x.item() for x in self.losses]}")
            if self.global_step % self.per_save_step == 0:
                save_noise(self.hps, self.u_noise, self.global_step)

    def on_after_backward(self):
        eps = self.hps.train.epsilon
        with torch.no_grad():
            # ① grad clipping (optional)
            torch.nn.utils.clip_grad_value_([self.u_noise], clip_value=1.0)
            # ② L∞-projection
            self.u_noise.clamp_(-eps, eps)

    def on_fit_end(self):
        save_noise(self.hps, self.u_noise, self.global_step)
        print(f"[UniversalNoise] Saved universal noise → {self.global_step}")


    def configure_optimizers(self):
        return SGD([self.u_noise], lr=self.hps.train.learning_rate, momentum=0)


    def train_dataloader(self):
        train_dataset = TextAudioSpeakerLoader(self.hps.train.training_files, self.hps.data)
        collate_fn = TextAudioSpeakerCollate()
        loader = DataLoader(
                train_dataset,
                num_workers=32,
                batch_size=self.hps.train.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                pin_memory=True,
                drop_last=False,
                )
        return loader