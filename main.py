import torch
from pytorch_lightning.cli import LightningCLI
from lightning import AutoregressiveLM, DiffusionLM
from lightning.mock import MockData
from lightning.data import ConcatData
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary, TQDMProgressBar

def cli_main():
    # TQDMProgressBar works properly in Colab (updates in place)
    # refresh_rate=1000 means update only every 1000 batches
    progress_bar = TQDMProgressBar(refresh_rate=1000)

    cli = LightningCLI(
        trainer_defaults={
            'accelerator': 'gpu',
            'strategy': 'auto',
            'enable_progress_bar': False, # Disable default, use ours
            'log_every_n_steps': 1000,    # Log less to save disk
            'callbacks': [
                ModelCheckpoint(
                    save_top_k=1,
                    save_last=True,
                    every_n_train_steps=10000,
                    filename='{epoch}-{step}',
                ),
                ModelSummary(max_depth=1),
                progress_bar
            ]
        }
    )


if __name__ == "__main__":
    cli_main()

