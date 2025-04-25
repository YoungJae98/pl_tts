# train_universal_noise.py
import torch
import datetime
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy

from noise_pop import utils          # argparse → HParams
from noise_pop.noise_trainer import UniversalNoiseModule  # 방금 만든 모듈

if __name__ == "__main__":
    # ── 1. 하이퍼파라미터 & 로거 ─────────────────────────────
    hps    = utils.get_hparams()              # CLI 인자 → HParams
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)

    # ── 2. LightningModule 생성 ────────────────────────────
    model = UniversalNoiseModule(hps, logger)

    # ── 3. DDP 설정 (멀티-GPU 사용 시) ──────────────────────
    gpu_num = torch.cuda.device_count()
    ddp = DDPStrategy(
        process_group_backend="nccl",
        timeout=datetime.timedelta(seconds=5400),
        find_unused_parameters=True,
        gradient_as_bucket_view=True,
    )

    # ── 4. Trainer ─────────────────────────────────────────
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=gpu_num,
        precision=16,                   # fp16
        max_steps=hps.train.total_steps,  
        strategy=ddp,
        enable_checkpointing=False,  
    )

    # ── 5. 학습 시작 ───────────────────────────────────────
    trainer.fit(model)                  # DataLoader는 모듈 내부 제공
