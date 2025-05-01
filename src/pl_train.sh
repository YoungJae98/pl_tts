date

model_name="attack_origin_train_1"
training_file="data/train_txt/attack_train_origin_pp.txt"
m_port=3459
FOLDER="vits/checkpoints/${model_name}"
batch_size=12
learning_rate=0.0002
epochs=2000
n_speakers=51


if [ ! -d $FOLDER ]; then
        mkdir $FOLDER
fi



OMP_NUM_THREADS=12 torchrun --nnodes=1 --nproc_per_node=8 --master_port=${m_port} pl_train.py  \
            -m ${model_name} -c vits/configs/vits_base.json -f ${training_file} -lr ${learning_rate} \
            -e ${epochs} -bs ${batch_size} -speak ${n_speakers}

