import librosa
import pyworld as pw
from collections import defaultdict
from tqdm import tqdm
import numpy as np

def get_f0_mean(wav, sr):
    _f0, t = pw.dio(wav.astype(np.float64), sr)  # raw pitch extraction
    f0 = pw.stonemask(wav.astype(np.float64), _f0, t, sr)  # refinement
    f0_nonzero = f0[f0 > 0]
    return np.mean(f0_nonzero) if len(f0_nonzero) > 0 else 0

def get_energy_mean(wav):
    frame_length = 2048
    hop_length = 512
    energy = librosa.feature.rms(y=wav, frame_length=frame_length, hop_length=hop_length)[0]
    return np.mean(energy)

def compute_speaker_averages(speaker_wav_dict, sr):
    speaker_stats = {}
    for speaker, wav_paths in tqdm(speaker_wav_dict.items()):
        f0_list, energy_list = [], []
        for path in tqdm(wav_paths):
            wav, _ = librosa.load(path, sr=sr)
            f0_list.append(get_f0_mean(wav, sr))
            energy_list.append(get_energy_mean(wav))
        speaker_stats[speaker] = {
            "f0_mean": np.mean(f0_list),
            "energy_mean": np.mean(energy_list),
        }
    return speaker_stats

def parse_speaker_wav_dict(txt_path):
    speaker_wav_dict = defaultdict(list)
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            path, speaker_id, _ = line.strip().split('|')
            speaker_wav_dict[speaker_id].append(path)
    
    return speaker_wav_dict


txt_file = '/workspace/pl_tts/src/data/train_txt/model_train2.txt'  # 예: 경로|speaker_id|text
sr = 22050

# 화자별 wav 경로 리스트 구성
speaker_wav_dict = parse_speaker_wav_dict(txt_file)

# 1. 각 화자 평균
speaker_stats = compute_speaker_averages(speaker_wav_dict, sr)

# 2. 전체 평균
pitch_vals = [v['f0_mean'] for v in speaker_stats.values()]
energy_vals = [v['energy_mean'] for v in speaker_stats.values()]
pitch_global_mean = np.mean(pitch_vals)
energy_global_mean = np.mean(energy_vals)

# 3. 기준을 넘는 화자 탐색
high_pitch_speakers = [spk for spk, v in speaker_stats.items() if v['f0_mean'] > pitch_global_mean + 25]
low_pitch_speakers = [spk for spk, v in speaker_stats.items() if v['f0_mean'] < pitch_global_mean - 25]
high_energy_speakers = [spk for spk, v in speaker_stats.items() if v['energy_mean'] > energy_global_mean * 1.2]
low_energy_speakers = [spk for spk, v in speaker_stats.items() if v['energy_mean'] < energy_global_mean * 0.8]

print(high_pitch_speakers)
print(low_pitch_speakers)
print(high_energy_speakers)
print(low_energy_speakers)