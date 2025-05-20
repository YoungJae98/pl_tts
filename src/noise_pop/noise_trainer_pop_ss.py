#!/usr/bin/env python3
# universal_perturbation

import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from vits.utils.data_utils import TextAudioSpeakerLoader2, TextAudioSpeakerCollate2
from vits.model.models import SynthesizerTrn
from vits.text.symbols import symbols
from torch.autograd import Variable
import torch.optim as optim
from vits.utils import utils
from vits.model import commons
from noise_pop.utils import get_spec, compute_reconstruction_loss, compute_kl_divergence, compute_perceptual_loss, compute_stft, compute_stoi
torch.backends.cudnn.enabled = False



class NoiseModule(pl.LightningModule):
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

        self.epsilon = hps.train.epsilon / 255
        self.inner_steps = 200
        self.alpha = self.epsilon / 10
        self.weight_alpha = hps.train.alpha
        self.weight_beta = 10
    
        
        self._logger = logger
        self.automatic_optimization = False
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def safespeech_train(self, hps, net_g, batch_data, epsilon, alpha, max_epoch, weights, device, mode="SafeSpeech"):
        '''
            The perturbation generation function based on settings of SafeSpeech.
            Output: loss items for presentation and noise for protection.
        '''
        weight_alpha, weight_beta = weights

        text, text_len, spec, spec_len, wav, wav_len, speakers, fnames = batch_data
        
        text, text_len = text.to(device), text_len.to(device)
        wav, wav_len = wav.to(device), wav_len.to(device)
        speakers = speakers.to(device)

        noise = torch.zeros(wav.shape).to(device)

        ori_wav = wav
        p_wav = Variable(ori_wav.data + noise, requires_grad=True)
        p_wav = Variable(torch.clamp(p_wav, min=-1., max=1.), requires_grad=True)

        opt_noise = torch.optim.SGD([p_wav], lr=5e-2)

        net_g.train()
        for iteration in range(max_epoch):

            opt_noise.zero_grad()

            p_spec, spec_len = get_spec(hps.data, p_wav, wav_len)
            p_spec = p_spec.half().to(device)
            spec_len = spec_len.half().to(device)

            wav_hat, l_length, attn, ids_slice, x_mask, z_mask, \
                (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(text, text_len, p_spec, spec_len, speakers, is_clip=False)

            torch.manual_seed(1234)
            random_z = torch.randn(wav_hat.shape).to(device)

            # The pivotal objective, i.e, the mel loss
            loss_mel = compute_reconstruction_loss(hps, p_wav, wav_hat)

            # Speech PErturbative Concealment based on KL-divergence
            loss_kl = compute_kl_divergence(hps, wav_hat, random_z)
            loss_nr = compute_reconstruction_loss(hps, wav_hat, random_z)

            if mode == "SPEC":
                loss = loss_mel + weight_beta * (loss_nr + loss_kl)
                loss_items = {
                    "loss_mel": f"{loss_mel.item():.6f}", 
                    "loss_nr": f"{loss_nr.item():.6f}", 
                    "loss_kl": f"{loss_kl.item():.6f}"
                }
            elif mode == "SafeSpeech":
                # Conbining SPEC with perceptual loss for human perception
                loss_perceptual = compute_perceptual_loss(hps, p_wav, wav)

                loss = loss_mel + weight_beta * (loss_nr + loss_kl) + weight_alpha * loss_perceptual
                loss_items = {
                    "loss_mel": f"{loss_mel.item():.6f}", 
                    "loss_nr": f"{loss_nr.item():.6f}", 
                    "loss_kl": f"{loss_kl.item():.6f}",
                    "loss_perception": f"{loss_perceptual.item():.6f}"
                }
            else:
                raise TypeError("The protective mode is wrong!")

            p_wav.retain_grad = True
            loss.backward()
            grad = p_wav.grad

            # Update the perturbation
            noise = alpha * torch.sign(grad) * -1.
            p_wav = Variable(p_wav.data + noise, requires_grad=True)
            noise = torch.clamp(p_wav.data - ori_wav.data, min=-epsilon, max=epsilon)
            p_wav = Variable(ori_wav.data + noise, requires_grad=True)
            p_wav = Variable(torch.clamp(p_wav, min=-1., max=1.), requires_grad=True)


        return noise, loss_items, fnames



    def pop_train(self, hps, net_g, epsilon, alpha, max_epoch, batch_data, device, mode="POP"):
        text, text_len, spec, spec_len, wav, wav_len, speakers, fnames = batch_data
        text, text_len = text.to(device), text_len.to(device)
        wav, wav_len = wav.to(device), wav_len.to(device)
        speakers = speakers.to(device)
        noise = torch.zeros(wav.shape).to(device)

        p_wav = Variable(wav.data + noise, requires_grad=True)
        p_wav = Variable(torch.clamp(p_wav, min=-1., max=1.), requires_grad=True)

        lr_noise = 5e-2
        opt_noise = optim.SGD([p_wav], lr=lr_noise, weight_decay=0.95)

        net_g.train()
        loss = 0.0
        for iteration in range(max_epoch):
            opt_noise.zero_grad()
            p_spec, p_len = get_spec(hps.data, p_wav, wav_len)
            p_spec = p_spec.half().to(device)
            p_len = p_len.half().to(device)

            is_fixed = True if mode != "RSP" else False
            is_clip = True if mode != "ESP" else False

            wav_hat, l_length, attn, ids_slice, x_mask, z_mask, \
                (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(text, text_len, p_spec, spec_len, speakers,
                                                            is_fixed=is_fixed, is_clip=is_clip)

            if ids_slice is not None:
                p_wav_slice = commons.slice_segments(p_wav, ids_slice * hps.data.hop_length, hps.train.segment_size)
            else:
                p_wav_slice = p_wav

            loss_mel = (self.hps, p_wav_slice, wav_hat)

            if mode == "POP":
                loss = loss_mel
            elif mode == "RSP":
                loss = loss_mel
            elif mode == "ESP":
                loss = loss_mel
            else:
                raise "The protective mode is setting wrong!"

            p_wav.retain_grad = True
            loss.backward()
            opt_noise.step()
            grad = p_wav.grad

            noise = alpha * torch.sign(grad) * -1.
            p_wav = Variable(p_wav.data + noise, requires_grad=True)
            noise = torch.clamp(p_wav.data - wav.data, min=-epsilon, max=epsilon)
            p_wav = Variable(wav.data + noise, requires_grad=True)
            p_wav = Variable(torch.clamp(p_wav, min=-1., max=1.), requires_grad=True)

        return noise, loss, fnames

    def training_step(self, batch, batch_idx):
        device = next(self.net_g.parameters()).device
        # noises, loss, fnames = self.pop_train(self.hps, self.net_g, self.epsilon, self.alpha, self.inner_steps, batch, device, "POP")
        noises, loss, fnames = self.safespeech_train(self.hps, self.net_g, batch, self.epsilon, self.alpha, self.inner_steps,[self.weight_alpha, self.weight_beta], device, "SafeSpeech")
        
        for i, fname in enumerate(fnames):
            noise_i = noises[i].cpu()  # shape [1, T_padded]
            save_path = os.path.join(self.hps.model_dir, f"{fname}.pt")
            torch.save(noise_i, save_path)

        return None

    def configure_optimizers(self):
        # manual update; no optimizer
        return None

    def train_dataloader(self):
        ds = TextAudioSpeakerLoader2(self.hps.train.training_files, self.hps.data)
        return DataLoader(
            ds,
            batch_size=self.hps.train.batch_size,
            shuffle=True,
            collate_fn=TextAudioSpeakerCollate2(),
            pin_memory=True,
            num_workers=4,
            drop_last=False
        )
