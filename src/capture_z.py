import os
import torch

from vits.utils import utils
from vits.model.models import SynthesizerTrn
from vits.text.symbols import symbols
from vits.text import cleaned_text_to_sequence
from tqdm import tqdm

import random

device = "cuda:7"
seed = 1234
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# ───── 하이퍼파라미터 및 모델 불러오기 ─────
hps = utils.get_hparams_from_file("vits/configs/vits_base.json")
checkpoint_name = "vits_pl_test"
model_name = "G_40000"

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=61,
    **hps.model).to(device).eval()

_ = utils.load_checkpoint(f"vits/checkpoints/{checkpoint_name}/{model_name}.pth", net_g, None)

for p in net_g.parameters():
    p.requires_grad = False

# ───── 텍스트 파일 불러오기 ─────
txt_path = "/data/youngjae/vits_pl/src/data/train_txt/ms_model_train_pp.txt"
with open(txt_path, "r") as f:
    train_txts = f.readlines()

random.shuffle(train_txts)
train_txts = train_txts[:20000]

# ───── 저장 경로 설정 ─────
save_base = "/data/youngjae/vits_pl/src/sae/vits_zgm/"
os.makedirs(save_base, exist_ok=True)

# ───── inference + 저장 ─────
for line in tqdm(train_txts):
    try:
        uid = line.split("|")[0].split("/")[-1].split(".")[0]
        sid = float(line.split("|")[1])
        txt = line.split("|")[2].strip()

        stn_text = torch.LongTensor(cleaned_text_to_sequence(txt)).to(device)
        x_len = torch.LongTensor([stn_text.size(0)]).to(device)
        sid_tensor = torch.LongTensor([int(sid)]).to(device)

        with torch.no_grad():
            z, y_mask, g, _ = net_g.infer(
                stn_text.unsqueeze(0),
                x_len,
                sid=sid_tensor,
                noise_scale=0, noise_scale_w=0, length_scale=1,
                return_z=True
            )

        # 저장 형태: dict 형태로 저장
        save_data = {
            "z": z.squeeze(0).cpu(),              # (d, T)
            "y_mask": y_mask.squeeze(0).cpu(),    # (1, T) → (T)
            "g": g.squeeze(0).cpu()               # (d_g)
        }
        save_path = os.path.join(save_base, f"{uid}.pt")
        torch.save(save_data, save_path)
    except Exception as e:
        print(f"[ERROR] {uid} 처리 중 오류 발생: {e}")
