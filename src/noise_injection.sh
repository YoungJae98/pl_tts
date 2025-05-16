# epsilon=1.5e-2
# percentile=0.8

# python noise_injection.py \
#   --config vits/configs/vits_base.json \
#   --noise_path noise_pop/checkpoints/univ_noise_vits_scale${epsilon}/universal_delta_epoch1.pt \
#   --epsilon ${epsilon} \
#   --percentile ${percentile} \
#   --input /data/dataset/anam/001_jeongwon/wavs/ \
#   --output_dir /data/dataset/anam/001_jeongwon_perturbed_${epsilon}_${percentile}/

python noise_injection.py \
  --noise_dir noise_pop/checkpoints/safespeech_noise_ep2_0.3 \
  --input /data/dataset/anam/001_jeongwon/wavs/ \
  --output_dir /data/dataset/anam/001_jeongwon_perturbed_ss_ep2_0.3_mask/
