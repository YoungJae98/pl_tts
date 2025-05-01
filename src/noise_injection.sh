python noise_injection.py \
  --config vits/configs/vits_base.json \
  --noise_path noise_pop/checkpoints/univ_noise_vits_ver2/universal_delta_epoch3.pt \
  --input /data/dataset/anam/001_jeongwon/wavs/ \
  --output_dir /data/dataset/anam/001_jeongwon_perturbed/
