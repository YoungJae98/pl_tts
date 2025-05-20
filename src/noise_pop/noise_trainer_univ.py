#!/usr/bin/env python3
# universal_perturbation

import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from vits.utils.data_utils import TextAudioSpeakerLoader, TextAudioSpeakerCollate
from vits.model.models import SynthesizerTrn
from vits.text.symbols import symbols
from vits.utils import utils
from vits.model import commons
from noise_pop.utils import get_spec, mel_recon_loss, energy_gate, tile_and_shift
torch.backends.cudnn.enabled = False



class UnivNoiseModule_ver2(pl.LightningModule):
    def __init__(self, hps, logger):
        super().__init__()
        self.hps = hps
        # load VITS generator (frozen)
        self.net_g = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model
        )
        ckpt = utils.latest_checkpoint_path(hps.train.pretrained_path, "G_*.pth")
        utils.load_checkpoint(ckpt, self.net_g)
        self.net_g.train()
        for p in self.net_g.parameters():
            p.requires_grad = False

        # for m in self.net_g.modules():
        #     if isinstance(m, torch.nn.Dropout):
        #         m.eval()    

        # hyperparameters from hps.train
        self.epsilon = hps.train.epsilon
        self.inner_steps = 20
        self.alpha = self.epsilon / max(1, self.inner_steps)
        # self.patch_sec = 0.1
        # patch_len = int(hps.data.sampling_rate * self.patch_sec)
        patch_len = 8192
        # universal delta
        self.delta = torch.zeros(1, 1, patch_len).cuda()
        self._logger = logger
        self.custom_step = 0
        self.automatic_optimization = False

    def pgd_delta(self, x, x_len, spk, text, text_len):
        device = x.device
        δ_i = torch.zeros_like(self.delta, device=device, requires_grad=True)
        # compute energy gate once per sample
        gate = energy_gate(x, percentile=self.hps.train.percentile, beta=self.hps.train.beta)
        
        for _ in range(self.inner_steps):
            # apply universal + sample delta, then mask by gate
            δ_total = (self.delta + δ_i).clamp(-self.epsilon, self.epsilon) 
            δ_full = tile_and_shift(δ_total, x.size(-1))
            δ_masked = δ_full * gate
            p_wav = (δ_masked.squeeze(0) + x).clamp(-1, 1).requires_grad_()
            
            p_spec, p_len = get_spec(self.hps.data, p_wav, x_len)
            p_spec = p_spec.half().to(device)
            p_len = p_len.half().to(device)
            
            wav_hat, l_length, attn, ids_slice, x_mask, z_mask, \
                (z, z_p, m_p, logs_p, m_q, logs_q) = self.net_g(text, text_len, p_spec, p_len, spk,
                                                        is_fixed=True, is_clip=True)
            
            if ids_slice is not None:
                p_wav_slice = commons.slice_segments(p_wav, ids_slice * self.hps.data.hop_length, self.hps.train.segment_size)
            else:
                p_wav_slice = p_wav

            loss_adv = mel_recon_loss(self.hps, wav_hat, p_wav_slice).mean()
            loss_adv.backward(retain_graph=True)
            # self.manual_backward(loss_adv, params=[δ_i])
            # print(torch.sum(δ_i))

            grad = δ_i.grad.sign()
            δ_i.data = (δ_i + self.alpha * grad * -1).clamp(-self.epsilon, self.epsilon)
            
            # s = δ_i.sum().item()

            δ_i.grad.zero_()

        return δ_i.detach(), loss_adv.item()

    def training_step(self, batch, batch_idx):
        text, text_len, _, _, wav, wav_len, spk = batch

        δ_i, loss_adv = self.pgd_delta(wav, wav_len, spk, text, text_len)
        self.delta = (self.delta + δ_i).clamp(-self.epsilon, self.epsilon)

        if self.custom_step % 100 == 0 :
            self._logger.info((f"{self.custom_step} : mel_recon:", loss_adv))
        self.custom_step += 1
        return None

    def configure_optimizers(self):
        # manual update; no optimizer
        return None

    def train_dataloader(self):
        ds = TextAudioSpeakerLoader(self.hps.train.training_files, self.hps.data)
        return DataLoader(
            ds,
            batch_size=self.hps.train.batch_size,
            shuffle=True,
            collate_fn=TextAudioSpeakerCollate(),
            pin_memory=True,
            num_workers=getattr(self.hps.train, 'num_workers', 32)
        )

    def on_train_epoch_end(self):
        final_path = os.path.join(self.hps.model_dir, f"universal_delta_epoch{self.current_epoch + 1}.pt")
        torch.save(self.delta.cpu(), final_path)
        print(f"[Done] universal δ saved → {final_path}")
