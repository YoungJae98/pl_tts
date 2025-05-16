# #!/usr/bin/env python3
# # inference_simple_noise.py

# import os
# import argparse
# import torch
# import soundfile as sf
from noise_pop.utils import energy_gate, tile_and_shift
# from vits.utils import utils

# def parse_args():
#     parser = argparse.ArgumentParser(
#         description="Apply a universal noise patch with masking to WAV(s)"
#     )
#     parser.add_argument("--config", type=str, required=True,
#                         help="Path to hparams config JSON file")
#     parser.add_argument("--noise_path", type=str, required=True,
#                         help="Path to trained universal delta (.pt)")
#     parser.add_argument("--epsilon", type=float, required=True,
#                         help="Path to trained universal delta (.pt)")
#     parser.add_argument("--percentile", type=float, required=True,
#                         help="Path to trained universal delta (.pt)")
#     parser.add_argument("--input", type=str, required=True,
#                         help="Input WAV file or directory of WAVs")
#     parser.add_argument("--output_dir", type=str, default="perturbed_wavs",
#                         help="Directory to save masked perturbed WAVs")
#     return parser.parse_args()


# def add_masked_noise(wav_tensor, u_delta, hps, epsilon, percentile):
#     """
#     wav_tensor: [1,1,T]
#     u_delta:    [1,1,P]
#     returns:    [1,1,T] perturbed by tile&shift + energy mask
#     """
#     # compute energy gate: [1,1,T]
#     gate = energy_gate(wav_tensor, percentile=percentile, beta=0.1)
#     # clamp and tile universal patch    
#     delta = u_delta.clamp(-epsilon, epsilon)
#     noise_full = tile_and_shift(delta, wav_tensor.size(-1))  # [1,1,T]
#     # apply mask and add
#     # print(torch.sum(noise_full * gate))
#     pert = (noise_full * gate + wav_tensor).clamp(-1., 1.)
#     return pert


# def main():
#     args = parse_args()
#     # load hparams
#     hps = utils.get_hparams_from_file(args.config)
#     # prepare output directory
#     os.makedirs(args.output_dir, exist_ok=True)

#     # load learned delta
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     u_delta = torch.load(args.noise_path, map_location=device)
#     u_delta = u_delta.to(device)
#     # collect input files
#     wav_paths = []
    
#     if os.path.isdir(args.input):
#         for fn in sorted(os.listdir(args.input)):
#             if fn.lower().endswith('.wav'):
#                 wav_paths.append(os.path.join(args.input, fn))
#     else:
#         wav_paths = [args.input]
    
#     snr_dbs = []
#     # process each file
#     for path in wav_paths:
#         wav, sr = sf.read(path)
#         # convert to torch [1,1,T]
#         wav_t = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0).to(device)
#         # add masked noise
#         pert_t = add_masked_noise(wav_t, u_delta, hps, args.epsilon, args.percentile)
#         # to numpy
#         out = pert_t.squeeze().cpu().numpy()
#         # save
#         # print(u_delta)
#         # print("torch.max",torch.max(u_delta))
#         # print("torch.min",torch.min(u_delta))
#         # print("torch.mean",torch.mean(u_delta))
#         # break
#         out_path = os.path.join(args.output_dir, os.path.basename(path))
#         sf.write(out_path, out, sr)
#         print(f"Saved perturbed WAV: {out_path}")
#     #     noise_t = pert_t - wav_t
#     #     signal_rms = wav_t.pow(2).mean().sqrt()
#     #     noise_rms  = noise_t.pow(2).mean().sqrt()
#     #     snr_db = 20 * torch.log10(signal_rms / noise_rms)
#     #     snr_dbs.append(snr_db)
#     # print(torch.mean(torch.FloatTensor(snr_dbs)))

# if __name__ == '__main__':
#     main()

#!/usr/bin/env python3
# noise_injection.py

import os
import argparse
import torch
import soundfile as sf

def parse_args():
    parser = argparse.ArgumentParser(
        description="Inject per-file noise patches into WAV(s)"
    )
    parser.add_argument("--noise_dir", type=str, required=True,
                        help="Directory containing per-file noise .pt files")
    parser.add_argument("--input", type=str, required=True,
                        help="Input WAV file or directory of WAVs")
    parser.add_argument("--output_dir", type=str, default="perturbed_wavs",
                        help="Directory to save perturbed WAVs")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    # collect input files
    if os.path.isdir(args.input):
        wav_paths = sorted(
            os.path.join(args.input, fn)
            for fn in os.listdir(args.input)
            if fn.lower().endswith(".wav")
        )
    else:
        wav_paths = [args.input]

    for wav_path in wav_paths:
        # load wav
        wav_np, sr = sf.read(wav_path)
        wav_t = torch.from_numpy(wav_np).float().unsqueeze(0).unsqueeze(0).to(device)  # [1,1,T]

        # derive basename and corresponding noise file
        base = os.path.basename(wav_path)
        noise_file = os.path.join(args.noise_dir, base + ".pt")
        if not os.path.isfile(noise_file):
            print(f"[WARN] noise file not found: {noise_file}, skipping.")
            continue

        # load and crop noise
        noise = torch.load(noise_file, map_location=device)  # assumed [1,1,T_pad]
        # T = wav_t.size(-1)
        # noise = noise[..., :T]

        # add and clamp
        gate = energy_gate(wav_t, percentile=0.8, beta=0.1)
        perturbed = (wav_t + noise * gate).clamp(-1.0, 1.0)

        # save output
        out_np = perturbed.squeeze().cpu().numpy()
        out_path = os.path.join(args.output_dir, base)
        sf.write(out_path, out_np, sr)
        print(f"Saved perturbed WAV: {out_path}")

if __name__ == "__main__":
    main()
