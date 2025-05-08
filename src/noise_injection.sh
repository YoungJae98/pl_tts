epsilon=2e-2
percentile=0.8

python noise_injection.py \
  --config vits/configs/vits_base.json \
  --noise_path noise_pop/checkpoints/univ_noise_vits_scale${epsilon}/universal_delta_epoch1.pt \
  --epsilon ${epsilon} \
  --percentile ${percentile} \
  --input /data/dataset/anam/001_jeongwon/wavs/ \
  --output_dir /data/dataset/anam/001_jeongwon_perturbed_${epsilon}_${percentile}/
