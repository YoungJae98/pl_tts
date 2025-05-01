#!/usr/bin/env python3
# inference_simple_noise.py

import os
import argparse
import torch
import soundfile as sf
from noise_pop.utils import energy_gate, tile_and_shift
from vits.utils import utils

def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply a universal noise patch with masking to WAV(s)"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to hparams config JSON file")
    parser.add_argument("--noise_path", type=str, required=True,
                        help="Path to trained universal delta (.pt)")
    parser.add_argument("--input", type=str, required=True,
                        help="Input WAV file or directory of WAVs")
    parser.add_argument("--output_dir", type=str, default="perturbed_wavs",
                        help="Directory to save masked perturbed WAVs")
    return parser.parse_args()


def add_masked_noise(wav_tensor, u_delta, hps):
    """
    wav_tensor: [1,1,T]
    u_delta:    [1,1,P]
    returns:    [1,1,T] perturbed by tile&shift + energy mask
    """
    # compute energy gate: [1,1,T]
    gate = energy_gate(wav_tensor, percentile=0.6, beta=0.1)
    # clamp and tile universal patch
    delta = u_delta.clamp(-1e-4, 1e-4)
    noise_full = tile_and_shift(delta, wav_tensor.size(-1))  # [1,1,T]
    # apply mask and add
    # print(torch.sum(noise_full * gate))
    pert = (noise_full * gate + wav_tensor).clamp(-1., 1.)
    return pert


def main():
    args = parse_args()
    # load hparams
    hps = utils.get_hparams_from_file(args.config)
    # prepare output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # load learned delta
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    u_delta = torch.load(args.noise_path, map_location=device)
    u_delta = u_delta.to(device)
    # collect input files
    wav_paths = []
    
    if os.path.isdir(args.input):
        for fn in sorted(os.listdir(args.input)):
            if fn.lower().endswith('.wav'):
                wav_paths.append(os.path.join(args.input, fn))
    else:
        wav_paths = [args.input]

    # process each file
    for path in wav_paths:
        wav, sr = sf.read(path)
        # convert to torch [1,1,T]
        wav_t = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0).to(device)
        # add masked noise
        pert_t = add_masked_noise(wav_t, u_delta, hps)
        # to numpy
        out = pert_t.squeeze().cpu().numpy()
        # save
        out_path = os.path.join(args.output_dir, os.path.basename(path))
        sf.write(out_path, out, sr)
        print(f"Saved perturbed WAV: {out_path}")

if __name__ == '__main__':
    main()
