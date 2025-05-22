import os, glob, random, torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from vits.model.models import SynthesizerTrn
from vits.utils import utils
from vits.text import symbols
from vits.text import cleaned_text_to_sequence

class SparseAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, beta_l1, beta_norm, set_bias=True):
        super().__init__()
        self.beta_l1 = beta_l1
        self.beta_norm = beta_norm
        self.set_bias = set_bias

        if set_bias:
            self.enc = nn.Linear(input_dim, hidden_dim, bias=True)
            nn.init.xavier_uniform_(self.enc.weight)
            self.enc.bias.data.fill_(0)
            self.dec_bias = nn.Parameter(torch.zeros(input_dim))
        else:
            self.enc = nn.Linear(input_dim, hidden_dim, bias=False)
            nn.init.xavier_uniform_(self.enc.weight)
            
    def forward(self, z):
        h = F.relu(self.enc(z))            # (N, k)
        if self.set_bias:
            z_hat = F.linear(h, self.enc.weight.T, self.dec_bias)  # (N, D)
        else:
            z_hat = F.linear(h, self.enc.weight.T)
        return h, z_hat


    def loss(self, z, h, z_hat):
        """
        Compute combined loss:
          - Reconstruction MSE on latent z
          - L1 sparsity on hidden h
          - Optional latent norm matching between z and z_hat

        returns:
          total_loss: scalar tensor
          recon_loss: scalar tensor
          sparse_loss: scalar tensor
          norm_loss: scalar tensor (0 if beta_norm=0)
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(z_hat, z)
        # Sparsity loss (L1 on hidden activations)
        sparse_loss = self.beta_l1 * h.abs().mean()
        # Latent norm matching loss
        if self.beta_norm > 0:
            norm_z = z.norm(p=2, dim=1).mean()
            norm_z_hat = z_hat.norm(p=2, dim=1).mean()
            norm_loss = self.beta_norm * torch.abs(norm_z - norm_z_hat)
        else:
            norm_loss = torch.tensor(0.0, device=z.device)

        total_loss = recon_loss + sparse_loss + norm_loss
        return total_loss, recon_loss, sparse_loss, norm_loss



class InferDataset(Dataset):
    def __init__(self, txt_path, max_items=20000):
        with open(txt_path, "r") as f:
            lines = f.readlines()
        random.shuffle(lines)
        self.data = lines[:max_items]
        device = 'cuda'
        self.device = device

        # model 관련 설정
        hps = utils.get_hparams_from_file("vits/configs/vits_base.json")
        self.net_g = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model).to(device).eval()
        _ = utils.load_checkpoint("vits/checkpoints/vits_pl_test/G_40000.pth", self.net_g, None)
        for p in self.net_g.parameters():
            p.requires_grad = False


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]
        try:
            uid = line.split("|")[0].split("/")[-1].split(".")[0]
            sid = float(line.split("|")[1])
            txt = line.split("|")[2].strip()

            stn_text = torch.LongTensor(cleaned_text_to_sequence(txt)).to(self.device)
            x_len = torch.LongTensor([stn_text.size(0)]).to(self.device)
            sid = torch.LongTensor([sid]).to(self.device)

            with torch.no_grad():
                z, y_mask, g, _ = self.net_g.infer(stn_text.unsqueeze(0), x_len, sid=sid,
                                                   noise_scale=0, noise_scale_w=0, length_scale=1, return_z=True)
                z = z.squeeze(0).cpu()       # (d, T)
                y_mask = y_mask.squeeze(0).cpu()  # (1, T) → (T)
                g = g.squeeze(0).cpu()       # (d_g)
            return z.T, y_mask, g  # shape: (T, d), (T), (d_g)
        except Exception as e:
            print(f"Error at index {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self))
