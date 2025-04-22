
# VITS-Pytorch Lightning Refactored Training

이 리포지토리는 VITS(Variational Inference with Adversarial Learning 기반 TTS) 모델의 학습 스크립트를 PyTorch Lightning v2.5 기준으로 리팩토링한 코드입니다.

- **PyTorch Lightning 2.x 호환**
  - `Trainer(precision=16)` 를 이용한 자동 Mixed-Precision
  - `DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True)` 로 분산 학습 안정화
- **Manual Optimization 유지**
  - `automatic_optimization=False` 상태에서 `self.manual_backward(loss, optimizer)` 사용
  - Generator / Discriminator를 GAN 방식으로 분리 학습
- **코드 구조 정리**
  - `configure_optimizers` 반환형을 `[optim_g, optim_d], [sched_g, sched_d]` 튜플로 명시
  - `forward`, `training_step`, `on_train_batch_end`, `on_train_epoch_end` 분리
- **기타 개선**
  - `torch.stft(..., return_complex=True)` 리팩토링
  - `librosa.filters.mel` 의 positional → keyword 인자 전환


## 요구사항

- Python >= 3.9
- PyTorch >= 2.0
- PyTorch Lightning >= 2.5
- CUDA Toolkit (8 GPU 분산 학습 시 NCCL 필요)
- torchaudio, librosa, scipy, tqdm, wandb(선택)

bash
pip install -r requirements.txt


## 사용 방법


리팩토링 관련 문의는 PR 혹은 Issue를 통해 남겨주세요! 🎉

