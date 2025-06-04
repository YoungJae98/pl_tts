import torch
from torch import nn

from vits.model import commons
from vits.utils import utils
from vits.model.models import SynthesizerTrn
from vits.text.symbols import symbols
from vits.text import cleaned_text_to_sequence, text_to_sequence, batch_text_to_sequence

from scipy.io.wavfile import write

from sae.models.model import SparseAE
import numpy as np


def get_text(text, hps, language_code):
    text_norm = text_to_sequence(text, str(language_code))
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0) 
    text_norm = torch.LongTensor(text_norm)
    return text_norm

language_code = 0
device = 'cuda'

# inference setting & get h from a,b
hps = utils.get_hparams_from_file("vits/configs/vits_base.json")
checkpoint_name = "vits_pl_test"
model_name = "G_40000"
CUDA_LAUNCH_BLOCKING=1
net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=61,
    **hps.model).to(device)
_ = net_g.eval()

_ = utils.load_checkpoint(f"vits/checkpoints/{checkpoint_name}/{model_name}.pth", net_g, None)

sae = SparseAE(192, 192 * 16, 1.0, 0, True).to(device)
sae.load_state_dict(torch.load('/workspace/pl_tts/src/sae/checkpoints/vits_z/30epoch_5.0_0.1_0.05/sae_final.pth'))
sae.eval()



fixed_texts = ["안녕하세요, 앞으로 잘 부탁드립니다.","오늘 날씨가 참 좋네요.", "이 프로젝트는 정말 기대돼요.", "혹시 시간 괜찮으시면 잠깐 이야기 나눌 수 있을까요?", "도움 주셔서 진심으로 감사드립니다."]
high_pitch_speakers = ['23', '53', '37', '4']
low_pitch_speakers = ['46', '35', '6', '45']


def extract_h_from_text(text, speaker_id, vits_model, sae_model, hps, device):
    stn_text = get_text(text, hps, str(language_code)).unsqueeze(0).to(device)
    x_tst_lengths = torch.LongTensor([stn_text.size(0)]).to(device)

    # 2. speaker id tensor
    sid = torch.LongTensor([int(speaker_id)]).to(device)

    # 3. Inference from VITS (get z latent)
    with torch.no_grad():
        z, y_mask, g, max_len = vits_model.infer(
            stn_text, x_tst_lengths, y=None, sid=sid,
            noise_scale=0, noise_scale_w=0, length_scale=1.0,
            return_z=True
        )
        
        # 4. SAE에 z를 통과시켜 h 획득
        h, _ = sae_model(z.squeeze(0).T)  # z: [1, D, T] → T x D
        h = h.mean(dim=0)
        return h.cpu().numpy()  # shape: [D]
    
def get_avg_h_for_speaker(sid, text_list):
    hs = [extract_h_from_text(text, sid, net_g, sae, hps, device) for text in text_list]
    # print(np.stack(hs).shape)
    return np.mean(np.stack(hs), axis=0)
    
# 각 그룹에서 평균 h 벡터 수집
h_high = [get_avg_h_for_speaker(sid, fixed_texts) for sid in high_pitch_speakers]
h_low  = [get_avg_h_for_speaker(sid, fixed_texts) for sid in low_pitch_speakers]

# # ⛳ 핵심 수정
# h_high_mean = np.mean(np.stack(h_high), axis=0)
# h_low_mean  = np.mean(np.stack(h_low), axis=0)

# diff = np.abs(h_high_mean - h_low_mean)
# top_dims = np.argsort(diff)[-10:][::-1]
# for i in top_dims:
#     print(f"Dim {i}: high={h_high_mean[i]:.4f}, low={h_low_mean[i]:.4f}, diff={diff[i]:.4f}")




# high_pitch  ['23', '53', '37', '4']
# low_pitch   ['46', '35', '1', '28', '38', '6', '18', '45']
# high_energy ['23', '19', '7', '44', '40', '27', '2', '42', '52', '43']
# low_energy  ['46', '36', '10', '20', '22', '38', '37', '55']