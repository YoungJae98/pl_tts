#!/usr/bin/env python3
import sys
import soundfile as sf
import torch
import torch.nn.functional as F


def load_mono(path):
    wav, sr = sf.read(path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return torch.from_numpy(wav).float(), sr


def align(a, b):
    # make b same length as a
    L = a.numel()
    if b.numel() >= L:
        return b[:L]
    return torch.cat([b, torch.zeros(L - b.numel())], dim=0)


def cosine_similarity(a, b):
    a = a / (a.norm() + 1e-8)
    b = b / (b.norm() + 1e-8)
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1))


def main(path1, path2):
    wav1, sr1 = load_mono(path1)
    wav2, sr2 = load_mono(path2)
    if sr1 != sr2:
        print(f"Sample rate mismatch: {sr1} vs {sr2}")
        sys.exit(1)
    wav2 = align(wav1, wav2)
    sim = cosine_similarity(wav1, wav2)
    print(f"Cosine similarity between:\n  {path1}\nand\n  {path2}\n= {sim:.4f}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <file1.wav> <file2.wav>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
