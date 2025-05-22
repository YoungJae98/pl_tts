date

model_name="test"
training_file="data/train_txt/model_train2.txt"
m_port=3457
FOLDER="vits/checkpoints/${model_name}"
batch_size=4
learning_rate=0.0002
epochs=1000
n_speakers=61

if [ ! -d $FOLDER ]; then
        mkdir $FOLDER
fi

OMP_NUM_THREADS=12 torchrun --nnodes=1 --nproc_per_node=2 --master_port=${m_port} pl_train.py           \
            -m ${model_name} -c vits/configs/vits_base.json -f ${training_file} -lr ${learning_rate}    \
            -e ${epochs} -bs ${batch_size} -speak ${n_speakers}

