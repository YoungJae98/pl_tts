import torch
import datetime
from vits.trainer import VitsTrainer
from vits.utils import utils

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy

if __name__ == '__main__':
    # 하이퍼파라미터와 로거 준비
    hps    = utils.get_hparams()
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)

    # GPU 개수 확인 후 모델 래핑
    gpu_num  = torch.cuda.device_count()
    pl_model = VitsTrainer(hps, gpu_num, logger)

    # DDP 전략 (accelerator는 Trainer로 옮김)
    ddp = DDPStrategy(
        process_group_backend="nccl",
        timeout=datetime.timedelta(seconds=5400),
        find_unused_parameters=True,
        gradient_as_bucket_view=True,
    )

    # Trainer: 최소 구성
    trainer = pl.Trainer(
        precision=16,
        accelerator="gpu",        # GPU 사용
        devices=gpu_num,          # GPU 개수
        max_epochs=hps.train.epochs,
        strategy=ddp,             # 위에서 정의한 DDP,
        use_distributed_sampler=False,
        enable_checkpointing=False,
    )

    # 학습 시작
    trainer.fit(pl_model)