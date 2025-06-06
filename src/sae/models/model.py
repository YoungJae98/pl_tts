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
            
    def forward(self, z, steer = False, num_s = 4.0):
        h = F.relu(self.enc(z))            # (N, k)
        if steer:
            p_cnt = 0
            a_cnt = 0
            for h_i in h:
                p_cnt += sum(h_i != 0)
            for h_i in h:
                h_i[h_i < 1.5] = 0
            for h_i in h:
                a_cnt += sum(h_i != 0)
                
            print(p_cnt)
            print(a_cnt)
            if self.set_bias:
                z_hat = F.linear(h, self.enc.weight.T, self.dec_bias)  # (N, D)
            else:
                z_hat = F.linear(h, self.enc.weight.T)
        else:
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
