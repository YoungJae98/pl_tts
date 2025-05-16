date

model_name="attack_train_fine_safespeech_ep2_0.3_mask"
training_file="data/train_txt/attack_train_pp.txt"
m_port=3457
FOLDER="vits/checkpoints/${model_name}"
batch_size=16
learning_rate=0.0002
epochs=1000
n_speakers=51


if [ ! -d $FOLDER ]; then
        mkdir $FOLDER
fi

OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nnodes=1 --nproc_per_node=4 --master_port=${m_port} pl_train.py           \
            -m ${model_name} -c vits/configs/vits_ref.json -f ${training_file} -lr ${learning_rate}    \
            -e ${epochs} -bs ${batch_size} -speak ${n_speakers}

