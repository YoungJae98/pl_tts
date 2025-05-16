#!/bin/bash
date

# ── 실험 파라미터 ─────────────────────────────────────
model_name="univ_noise_vits_scale1.5e-2"           # 저장 폴더·파일명
training_file="data/train_txt/noise_train_compact.txt"

batch_size=1
learning_rate=0.01                         # u_noise용 SGD LR
max_epochs=2                          # 학습 step 수 (= max_steps)

snr_db=15
percentile=0.7
beta=0.1
lambda_pen=0.001
epsilon=1.5e-2

n_speakers=51
pretrained_path="vits/checkpoints/vits_ref_pt/"
config_json="vits/configs/vits_ref.json"

# ── 런타임 설정 ───────────────────────────────────────
FOLDER="noise_pop/checkpoints/${model_name}"
master_port=3125
gpus=1                                     # nproc_per_node

mkdir -p "${FOLDER}"

# ── 학습 실행 ─────────────────────────────────────────
OMP_NUM_THREADS=12 torchrun --nnodes=1 --nproc_per_node=${gpus} \
  --master_port=${master_port} pop_train_ver2.py              \
    -m  "${model_name}"        \
    -c  "${config_json}"       \
    -tf "${training_file}"     \
    -lr "${learning_rate}"     \
    -me "${max_epochs}"       \
    -bs "${batch_size}"        \
    -p  "${pretrained_path}"   \
    -ns "${n_speakers}"        \
    -ep "${epsilon}"            \
    --snr_db     "${snr_db}"   \
    --percentile "${percentile}" \
    --beta       "${beta}"     \
    --lambda_pen "${lambda_pen}"