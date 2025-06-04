import librosa
import pyworld as pw
from collections import defaultdict
from tqdm import tqdm
import numpy as np

# ─── 설정 ────────────────────────────────────────────────────────────────
txt_file = '/workspace/pl_tts/src/data/train_txt/model_train2.txt'
sr = 22050

# 원하는 σ 배수를 여기서 설정하세요 (예: 1.5, 2.0)
threshold_factor = 2.0
# ─────────────────────────────────────────────────────────────────────────

def get_f0_mean(wav, sr):
    _f0, t = pw.dio(wav.astype(np.float64), sr)
    f0 = pw.stonemask(wav.astype(np.float64), _f0, t, sr)
    f0_nonzero = f0[f0 > 0]
    return float(np.mean(f0_nonzero)) if len(f0_nonzero) > 0 else 0.0

def get_energy_mean(wav):
    # librosa RMS energy
    energy = librosa.feature.rms(
        y=wav,
        frame_length=2048,
        hop_length=512
    )[0]
    return float(np.mean(energy))

def parse_speaker_wav_dict(txt_path):
    speaker_wav_dict = defaultdict(list)
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            path, speaker_id, _ = line.strip().split('|')
            speaker_wav_dict[speaker_id].append(path)
    return speaker_wav_dict

def compute_speaker_averages(speaker_wav_dict, sr):
    speaker_stats = {}
    for spk, wav_paths in tqdm(speaker_wav_dict.items(), desc="Speakers"):
        f0_list, en_list = [], []
        for path in wav_paths:
            wav, _ = librosa.load(path, sr=sr)
            f0_list.append(get_f0_mean(wav, sr))
            en_list.append(get_energy_mean(wav))
        speaker_stats[spk] = {
            'f0_mean':  np.mean(f0_list),
            'energy_mean': np.mean(en_list),
        }
    return speaker_stats

# 1. 화자별 통계 계산
speaker_wav_dict = parse_speaker_wav_dict(txt_file)
speaker_stats     = compute_speaker_averages(speaker_wav_dict, sr)

# 2. 전체 분포의 평균과 표준편차
pitch_vals  = np.array([v['f0_mean']    for v in speaker_stats.values()])
energy_vals = np.array([v['energy_mean'] for v in speaker_stats.values()])

pitch_mean,  pitch_std  = pitch_vals.mean(),  pitch_vals.std()
energy_mean, energy_std = energy_vals.mean(), energy_vals.std()

# 3. 임계치 계산 (μ ± threshold_factor * σ)
pitch_hi = pitch_mean  + threshold_factor * pitch_std
pitch_lo = pitch_mean  - threshold_factor * pitch_std
energy_hi = energy_mean + threshold_factor * energy_std
energy_lo = energy_mean - threshold_factor * energy_std

# 4. 기준 넘는 화자 추출
high_pitch_speakers  = [spk for spk, v in speaker_stats.items() if v['f0_mean']    > pitch_hi]
low_pitch_speakers   = [spk for spk, v in speaker_stats.items() if v['f0_mean']    < pitch_lo]
high_energy_speakers = [spk for spk, v in speaker_stats.items() if v['energy_mean'] > energy_hi]
low_energy_speakers  = [spk for spk, v in speaker_stats.items() if v['energy_mean'] < energy_lo]

# 5. 결과 출력
print(f"Pitch overall: μ={pitch_mean:.2f} Hz, σ={pitch_std:.2f} Hz")
print(f"Thresholds: >{pitch_hi:.2f} Hz (high), <{pitch_lo:.2f} Hz (low)")
print("High-pitch speakers:", high_pitch_speakers)
print("Low-pitch speakers: ", low_pitch_speakers)
print()
print(f"Energy overall: μ={energy_mean:.5f}, σ={energy_std:.5f}")
print(f"Thresholds: >{energy_hi:.5f} (high), <{energy_lo:.5f} (low)")
print("High-energy speakers:", high_energy_speakers)
print("Low-energy speakers: ", low_energy_speakers)
