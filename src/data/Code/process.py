f = open("/workspace/pl_tts/src/data/train_txt/ms_model_train_pp.txt", "r")
lines = f.readlines()

n_f = open("/workspace/pl_tts/src/data/train_txt/model_train2.txt", "w")

#/home/jeonyj0612/data
for line in lines:
    new_line = line.replace("/data/dataset/aihub", "/workspace/data")
    n_f.write(new_line)