# train_sae_vits.py
# ────────────────────────────────────────────────────────────
#  ▶ z_dir        : z 텐서(.pt) 폴더 (shape: (T, d))
#  ▶ 학습 방식     : 파일 단위 배치(batch_files)로 묶음 → 프레임축 concat
#  ▶ 손실          : recon(MSE) + β·L1 + mel loss
#  ▶ 출력          : sae_vits.pth, 체크포인트(epoch) 저장
#  ▶ 로깅          : Weights & Biases(wandb)에 지표 기록
#  ▶ 검증          : 전체 중 500개 파일로 validation 수행
# ────────────────────────────────────────────────────────────
import os, glob, random, torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from sae.models.model import SparseAE, InferDataset
from vits.model.models import SynthesizerTrn
from vits.utils import utils
from vits.text import symbols

from noise_pop.utils import compute_reconstruction_loss
import wandb

# ───── 설정 ─────────────────────────────────────────────────
z_dir            = "sae/vits_zgm"         # 저장된 z 경로
D                = 192                  # z feature dim
gamma            = 16                   # 확장 계수
k                = D * gamma            # sparse dim
batch_files      = 1                    # 파일 단위 배치 크기
num_epochs       = 30
val_size         = 500                  # validation 파일 개수
lr               = 3e-4                 # learning rate
beta_l1          = [1.0, 2.0, 5.0]      # L1 희소성 계수
beta_norm        = [0.01, 0.1, 0.2]
beta_mel         = 0.05
device           = "cuda:1"
batch_shuffle    = True                 # 배치 파일 랜덤 셔플 여부

seed = 1234
# 재현성 설정
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# ───── Dataset 정의 ─────────────────────────────────────────
class ZDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        data = torch.load(self.paths[idx])  # (d,T) or (T,d)
        # print(data)
        z = data["z"]
        y_mask = data["y_mask"]
        g = data["g"]
        if z.ndim == 2 and z.shape[1] == D:
            z = z
        elif z.ndim == 2 and z.shape[0] == D:
            z = z.T
        else:
            raise ValueError(f"Unexpected z shape: {z.shape}")
        return z.float(), y_mask.float(), g.float()

# collate_fn: batch 내 모든 프레임 concat
# def collate_fn(batch):
#     z_cat = torch.cat(batch, dim=0)  # (ΣTi, D)
#     return z_cat


# ───── 파일 목록 & split ───────────────────────────────────────
z_paths = sorted(glob.glob(os.path.join(z_dir, "*.pt")))
random.shuffle(z_paths)
val_paths = z_paths[:val_size]
train_paths = z_paths[val_size:]

# ───── DataLoader 준비 ─────────────────────────────────────────
train_loader = DataLoader(
    ZDataset(train_paths), batch_size=batch_files,
    shuffle=batch_shuffle, #collate_fn=collate_fn,
    num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    ZDataset(val_paths), batch_size=batch_files,
    shuffle=False, #collate_fn=collate_fn,
    num_workers=4, pin_memory=True
)

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

l1 = 2.0
norm = 0.1

# for l1 in beta_l1:
#     for norm in beta_norm:
#         if l1 != 1.0 or norm != 0.1:
#             continue


save_dir         = f"sae/checkpoints/vits_z/{num_epochs}epoch_{l1}_{norm}_{beta_mel}"
os.makedirs(save_dir, exist_ok=True)
# ────────────────────────────────────────────────────────────
seed = 1234
# 재현성 설정
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# ───── Weights & Biases 초기화 ─────────────────────────────────
wandb.init(
    project="vits-sae_mel",
    name=f"{l1}_{norm}_{beta_mel}",
    config={
        "D": D,
        "gamma": gamma,
        "k": k,
        "batch_files": batch_files,
        "num_epochs": num_epochs,
        "val_size": val_size,
        "lr": lr,
        "beta_l1": l1,
        "beta_norm": norm
    }
)
# 모델·옵티마이저·스케줄러 설정
ae = SparseAE(D, k, l1, norm).to(device)
optimizer = torch.optim.Adam(ae.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs, eta_min=1e-5
)
best_recon = float('inf')
global_step = 0
# 학습 루프
for epoch in range(1, num_epochs+1):
    # Train
    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
    epoch_recon = 0.0
    epoch_mel = 0.0

    accum_steps = 8
    accum_counter = 0

    optimizer.zero_grad()
    for z, mask, g in pbar:
        z = z.to(device)
        mask = mask.to(device)
        g = g.to(device)

        h, z_hat = ae(z)

        o = net_g.from_z(z.squeeze(0).T.unsqueeze(0), mask, g, None)
        o_hat = net_g.from_z(z_hat.squeeze(0).T.unsqueeze(0), mask, g, None)

        loss, recon, sparse, norm = ae.loss(z, h, z_hat)

        loss_mel = compute_reconstruction_loss(hps, o, o_hat)

        loss += beta_mel * loss_mel

        loss.backward()

        accum_counter += 1

        if accum_counter % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            epoch_recon += recon.item()
            epoch_mel += loss_mel.item()

            wandb.log({
                "train/loss": loss.item(),
                "train/recon": recon.item(),
                "train/sparse": sparse.item(),
                "train/norm": norm.item(),
                "train/mel":loss_mel.item()
            }, step=global_step)

            pbar.set_postfix({"recon": f"{recon.item():.5f}",
                            "sparse": f"{sparse.item():.5f}",
                            "norm": f"{norm.item():.5f}"})
    # LR 스케줄링

    if accum_counter % accum_steps != 0:
        optimizer.step()
        global_step += 1

    scheduler.step()
    # Validate
    val_recon = 0.0; val_steps = 0
    with torch.no_grad():
        for z, mask, g in val_loader:
            z = z.to(device)
            mask = mask.to(device)
            g = g.to(device)
            
            _, z_hat = ae(z)[0:2]
            recon = F.mse_loss(z_hat, z, reduction='mean')
            val_recon += recon.item(); val_steps += 1
            
    avg_train_recon = epoch_recon / len(train_loader)
    avg_train_mel = epoch_mel / len(train_loader)
    avg_val_recon = val_recon / max(val_steps,1)
    wandb.log({"epoch": epoch, "train/avg_recon": avg_train_recon, "train/avg_mel":avg_train_mel, "val/avg_recon": avg_val_recon}, step=global_step)
    # 체크포인트 저장
    if (epoch + 1) % 5 == 0 or avg_val_recon < best_recon:
        best_recon = avg_val_recon
        ckpt_path = os.path.join(save_dir, f"sae_epoch{epoch}.pth")
        torch.save(ae.state_dict(), ckpt_path); wandb.save(ckpt_path)
# 최종 저장
final_path = os.path.join(save_dir, "sae_final.pth")
torch.save(ae.state_dict(), final_path); wandb.save(final_path)
print(f"✓ 최종 SAE weights saved to {final_path}")
wandb.finish()