#!/usr/bin/env python3
import os
import argparse
import soundfile as sf
from pesq import pesq, PesqError
import csv
import librosa
from tqdm import tqdm

def compare_folders(ref_dir, deg_dir, sr, mode, output_csv):
    results = []
    # filenames = []
    for fn in tqdm(sorted(os.listdir(ref_dir))):
        if not fn.lower().endswith('.wav'):
            continue
        ref_path = os.path.join(ref_dir, fn)
        deg_path = os.path.join(deg_dir, fn)
        if not os.path.isfile(deg_path):
            print(f"[SKIP] Missing degraded file for {fn}")
            continue

        # Load reference signal and resample if needed
        ref, sr_ref = sf.read(ref_path)
        if sr_ref != sr:
            # print(f"[RESAMPLE] {fn}: {sr_ref} -> {sr} Hz")
            ref = librosa.resample(ref.astype(float), orig_sr=sr_ref, target_sr=sr)
        # Load degraded signal and resample if needed
        deg, sr_deg = sf.read(deg_path)
        if sr_deg != sr:
            # print(f"[RESAMPLE] {fn}: {sr_deg} -> {sr} Hz")
            deg = librosa.resample(deg.astype(float), orig_sr=sr_deg, target_sr=sr)

        # Ensure mono
        if ref.ndim > 1: ref = ref[:, 0]
        if deg.ndim > 1: deg = deg[:, 0]

        # Compute PESQ
        try:
            score = pesq(sr, ref, deg, mode)
        except PesqError as e:
            print(f"[ERROR] PESQ failed for {fn}: {e}")
            continue
        # if score < 3:
        #     continue
        # print(f"{fn}: PESQ = {score:.3f}")
        # filenames.append(fn)
        results.append(score)

    # Compute and display average PESQ
    if results:
        avg_score = sum(results) / len(results)
        print(f"\nProcessed {len(results)} files. Average PESQ = {avg_score:.3f}")

    # CSV 저장 (옵션)
    # if output_csv:
    #     with open(output_csv, 'w', newline='') as f:
    #         writer = csv.writer(f)
    #         writer.writerow(['filename', 'pesq_score'])
    #         for fn, score in zip(filenames, results):
    #             writer.writerow([fn, f"{score:.3f}"])
    #     print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch PESQ between two folders of WAVs with auto-resampling")
    parser.add_argument("--ref_dir",    type=str, required=True, help="Directory of reference (clean) WAVs")
    parser.add_argument("--deg_dir",    type=str, required=True, help="Directory of degraded WAVs")
    parser.add_argument("--sr",         type=int, choices=[8000,16000], required=True, help="Sampling rate for PESQ")
    parser.add_argument("--mode",       type=str, choices=["nb","wb"], default="wb", help="'nb' for 8 kHz, 'wb' for 16 kHz")
    parser.add_argument("--output_csv", type=str, default="",   help="Path to CSV output (optional)")
    args = parser.parse_args()

    compare_folders(args.ref_dir, args.deg_dir, args.sr, args.mode, args.output_csv)

#python get_pesq.py --ref_dir /data/dataset/anam/001_jeongwon/wavs --deg_dir /data/gyub/VITS/datasets_모음/datasets_ces01/231004_jeongwon/wavs --sr 16000 --mode wb