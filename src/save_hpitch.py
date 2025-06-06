import os
import numpy as np
import torch
import torch.nn.functional as F
import librosa
import pyworld as pw
from tqdm import tqdm

from torch import nn
from vits.model import commons
from vits.utils import utils
from vits.model.models import SynthesizerTrn
from vits.text.symbols import symbols
from vits.text import text_to_sequence

from sae.models.model import SparseAE

# ─── 설정 ─────────────────────────────────────────────────────────────
# 단일 텍스트
fixed_text = "잘가잘가잘가잘가잘가잘가잘가잘가잘가잘가잘가잘가"

# 저장 디렉터리
output_dir = "/workspace/pl_tts/src/sae/all_speakers_h_and_f0_2"
os.makedirs(output_dir, exist_ok=True)

# 오디오 샘플레이트
sr = 22050

# 처리할 화자 ID 리스트 (0부터 60까지)
speaker_ids = [str(i) for i in range(61)]
# ─────────────────────────────────────────────────────────────────────────

def extract_h_and_avg_f0_for_speaker(
    speaker_id: str,
    text: str,
    vits_model: nn.Module,
    sae_model: nn.Module,
    hps,
    device,
    sr: int
):
    """
    1) VITS inference: 텍스트 → (audio, ..., z, ...)
    2) pyworld를 사용해 waveform에서 프레임별 f0 값을 구한 뒤 평균 f0 계산
    3) SAE를 사용해 z → latent h_all (T_z, D_h)
    반환:
      - h_all: np.ndarray, shape (T_z, D_h)
      - avg_f0: float (무음 제외 평균 피치)
    """
    # 1) 텍스트 토큰으로 변환
    text_seq = text_to_sequence(text, str(0))  # language_code=0
    if hps.data.add_blank:
        text_seq = commons.intersperse(text_seq, 0)
    stn_text = torch.LongTensor(text_seq).to(device)  # (T_text,)
    x_tst = stn_text.unsqueeze(0)                     # (1, T_text)
    x_tst_lengths = torch.LongTensor([stn_text.size(0)]).to(device)

    sid = torch.LongTensor([int(speaker_id)]).to(device)

    # 2) VITS inference: return_z=False 로 호출해서 한 번에 (audio, attn, y_mask, (z, ...)) 얻기
    with torch.no_grad():
        audio_tensor, _, _, (z, *_ ) = vits_model.infer(
            x_tst, x_tst_lengths, y=None, sid=sid,
            noise_scale=0.0, noise_scale_w=0.0, length_scale=1.0,
            return_z=False  # 이 경우에도 z가 마지막 튜플로 반환됨
        )
        # audio_tensor: (1, 1, N) 형태, 범위 [-1, +1]
        wav = audio_tensor.squeeze().cpu().numpy()  # (N,)

    # 3) pyworld로 프레임별 F0 계산 → 무음 제외 평균 f0
    _f0, t = pw.dio(wav.astype(np.float64), sr)
    f0 = pw.stonemask(wav.astype(np.float64), _f0, t, sr)  # (T_wav,)
    f0_nonzero = f0[f0 > 0]
    avg_f0 = float(np.mean(f0_nonzero)) if len(f0_nonzero) > 0 else 0.0

    # 4) SAE로 z → h_all
    #    z: (1, D_z, T_z) tensor → (T_z, D_z) 로 변환
    z_seq = z.squeeze(0).transpose(0, 1)  # (T_z, D_z)
    with torch.no_grad():
        h_all_tensor, _ = sae_model(z_seq)  # h_all_tensor: (T_z, D_h)
    h_all = h_all_tensor.cpu().numpy()      # (T_z, D_h)

    return h_all, avg_f0

# ─── VITS & SAE 모델 로드 ───────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1) VITS 세팅
hps = utils.get_hparams_from_file("vits/configs/vits_base.json")
checkpoint_name = "vits_pl_test"
model_name = "G_40000"

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=61,
    **hps.model
).to(device)
net_g.eval()

utils.load_checkpoint(
    f"vits/checkpoints/{checkpoint_name}/{model_name}.pth",
    net_g, None
)

# 2) SparseAE 세팅
sae = SparseAE(192, 192 * 16, 1.0, 0, True).to(device)
sae_ckpt = "/workspace/pl_tts/src/sae/checkpoints/vits_z/30epoch_5.0_0.1_0.05/sae_final.pth"
sae.load_state_dict(torch.load(sae_ckpt))
sae.eval()
# ─────────────────────────────────────────────────────────────────────────

def main():
    # 각 화자별로 h_all과 평균 f0를 저장
    for spk in tqdm(speaker_ids, desc="Speakers 0-60"):
        h_all, avg_f0 = extract_h_and_avg_f0_for_speaker(
            speaker_id=spk,
            text=fixed_text,
            vits_model=net_g,
            sae_model=sae,
            hps=hps,
            device=device,
            sr=sr
        )
        # 저장 경로
        np.save(os.path.join(output_dir, f"speaker_{spk}_h.npy"), h_all)
        np.save(os.path.join(output_dir, f"speaker_{spk}_avg_f0.npy"), np.array(avg_f0))

    print("\n저장 완료:")
    print(f"→ h: 각 파일마다 shape (T_z, D_h), 경로 = {output_dir}/speaker_<id>_h.npy")
    print(f"→ avg_f0: 각 파일마다 scalar,       경로 = {output_dir}/speaker_<id>_avg_f0.npy")

if __name__ == "__main__":
    main()
