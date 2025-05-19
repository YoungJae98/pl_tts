#!/bin/bash
date

# ── 실험 파라미터 ─────────────────────────────────────

batch_size=1
max_epochs=1                          # 학습 step 수 (= max_steps)
epsilon=2
alpha=1
beta=1

model_name="safespeech_noise_ep${epsilon}_${alpha}_${beta}"           # 저장 폴더·파일명
training_file="data/train_txt/attack_train_origin_pp.txt"

n_speakers=51
pretrained_path="vits/checkpoints/vits_ref_pt/"
config_json="vits/configs/vits_ref.json"

# ── 런타임 설정 ───────────────────────────────────────
FOLDER="noise_pop/checkpoints/${model_name}"
master_port=3129
gpus=6                                     # nproc_per_node

mkdir -p "${FOLDER}"

# ── 학습 실행 ─────────────────────────────────────────
OMP_NUM_THREADS=12 torchrun --nnodes=1 --nproc_per_node=${gpus} \
  --master_port=${master_port} noise_train.py              \
    -m  "${model_name}"        \
    -c  "${config_json}"       \
    -tf "${training_file}"     \
    -me "${max_epochs}"       \
    -bs "${batch_size}"        \
    -p  "${pretrained_path}"   \
    -ep  "${epsilon}"                 \
    -ns "${n_speakers}"        \
    -a  "${alpha}"       