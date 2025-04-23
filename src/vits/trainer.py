import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
# from torch.amp import autocast, GradScaler

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy

from vits.model import commons
from vits.model.models import (
    SynthesizerTrn,
    MultiPeriodDiscriminator,
)
from vits.model.losses import (
    generator_loss,
    discriminator_loss,
    feature_loss,
    kl_loss,
)
from vits.utils import utils
from vits.utils.data_utils import (
    TextAudioSpeakerLoader,
    TextAudioSpeakerCollate,
    DistributedBucketSampler,
)
from vits.utils.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from vits.text.symbols import symbols

class VitsTrainer(pl.LightningModule):
    def __init__(self, hps, gpu_num, logger):
        super().__init__()
        self.hps = hps
        self.n_gpus = gpu_num
        self.automatic_optimization = False

        # model components
        self.generator = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        )
        self.discriminator = MultiPeriodDiscriminator(hps.model.use_spectral_norm)

        # utils
        self.logger_ = logger
        # self.scaler = GradScaler(enabled=hps.train.fp16_run)

        # bookkeeping
        self.vits_epoch = 1
        self.vits_step = 1

    def forward(self, *args, **kwargs):
        return self.generator(*args, **kwargs)
        
    def training_step(self, batch, batch_idx):
        # manual optimization
        optim_g, optim_d = self.optimizers()
        x, x_lengths, spec, spec_lengths, y, y_lengths, speakers = batch

        # discriminator update
        y_hat, l_length, attn, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = self(
            x, x_lengths, spec, spec_lengths, speakers
        )
        mel = spec_to_mel_torch(
            spec,
            self.hps.data.filter_length,
            self.hps.data.n_mel_channels,
            self.hps.data.sampling_rate,
            self.hps.data.mel_fmin,
            self.hps.data.mel_fmax,
        )
        y_mel = commons.slice_segments(
            mel, ids_slice, self.hps.train.segment_size // self.hps.data.hop_length
        )
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1),
            self.hps.data.filter_length,
            self.hps.data.n_mel_channels,
            self.hps.data.sampling_rate,
            self.hps.data.hop_length,
            self.hps.data.win_length,
            self.hps.data.mel_fmin,
            self.hps.data.mel_fmax,
        )
        y = commons.slice_segments(
            y, ids_slice * self.hps.data.hop_length, self.hps.train.segment_size
        )
        y_d_hat_r, y_d_hat_g, _, _ = self.discriminator(y, y_hat.detach())
        
        loss_disc, *_ = discriminator_loss(y_d_hat_r, y_d_hat_g)

        optim_d.zero_grad()
        self.manual_backward(loss_disc, retain_graph=True)
        optim_d.step()
        # self.scaler.scale(loss_disc).backward(retain_graph=True)
        # self.scaler.unscale_(optim_d)
        # self.scaler.step(optim_d)

        # generator update
        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.discriminator(y, y_hat)
        loss_dur = torch.sum(l_length.float())
        loss_mel = F.l1_loss(y_mel, y_hat_mel) * self.hps.train.c_mel
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * self.hps.train.c_kl
        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, *_ = generator_loss(y_d_hat_g)
        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl

        self.losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl]

        optim_g.zero_grad()
        self.manual_backward(loss_gen_all)
        optim_g.step()
        # self.scaler.scale(loss_gen_all).backward()
        # self.scaler.unscale_(optim_g)
        # self.scaler.step(optim_g)
        # self.scaler.update()
        self.vits_step += 1

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # checkpoint saving & info
        if self.global_rank == 0:
            if self.vits_step % self.hps.train.per_save_step == 0:
                optim_g, optim_d = self.optimizers()
                utils.save_checkpoint(
                    self.generator,
                    optim_g,
                    self.hps.train.learning_rate,
                    self.vits_epoch,
                    os.path.join(self.hps.model_dir, f"G_{self.vits_step}.pth"),
                )
                utils.save_checkpoint(
                    self.discriminator,
                    optim_d,
                    self.hps.train.learning_rate,
                    self.vits_epoch,
                    os.path.join(self.hps.model_dir, f"D_{self.vits_step}.pth"),
                )

            if self.vits_step % self.hps.train.per_log_step == 0:
                self.logger_.info(f"Epoch:{self.vits_epoch} / Step:{self.vits_step} / Losses:{[x.item() for x in self.losses]}")

    def on_train_epoch_end(self):
        # log epoch summary
        # if self.global_rank == 0:
        #     self.logger_.info(f"Epoch:{self.vits_epoch} / Step:{self.vits_step} / Losses:{[x.item() for x in self.losses]}")
        self.vits_epoch += 1

    def configure_optimizers(self):
        optim_g = torch.optim.AdamW(
            self.generator.parameters(),
            self.hps.train.learning_rate,
            betas=self.hps.train.betas,
            eps=self.hps.train.eps,
        )
        optim_d = torch.optim.AdamW(
            self.discriminator.parameters(),
            self.hps.train.learning_rate,
            betas=self.hps.train.betas,
            eps=self.hps.train.eps,
        )

        # resume checkpoints
        try:
            _, _, _, self.vits_epoch = utils.load_checkpoint(
                utils.latest_checkpoint_path(self.hps.model_dir, "G_*.pth"),
                self.generator,
                optim_g,
            )
            _, _, _, self.vits_epoch = utils.load_checkpoint(
                utils.latest_checkpoint_path(self.hps.model_dir, "D_*.pth"),
                self.discriminator,
                optim_d,
            )
            self.vits_epoch += 1
        except Exception:
            self.vits_epoch = 1

        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            optim_g,
            gamma=self.hps.train.lr_decay,
            last_epoch=self.vits_epoch - 2,
        )
        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
            optim_d,
            gamma=self.hps.train.lr_decay,
            last_epoch=self.vits_epoch - 2,
        )

        return [optim_g, optim_d], [scheduler_g, scheduler_d]
    
    # not need to prepare data
    def prepare_data(self):
        return

    # dataloader
    
    def train_dataloader(self):
        dataset = TextAudioSpeakerLoader(self.hps.training_files, self.hps.data)
        sampler = DistributedBucketSampler(
            dataset,
            self.hps.train.batch_size,
            [200, 500, 1000, 1500, 2000, 3000, 4000],
            num_replicas=self.n_gpus,
            rank=self.global_rank,
            shuffle=True,
        )
        collate_fn = TextAudioSpeakerCollate()
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=32,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        if self.vits_epoch != 1:
            self.vits_step = self.vits_epoch * len(loader)
        return loader