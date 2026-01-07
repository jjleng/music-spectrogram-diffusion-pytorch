import torch
from pytorch_lightning.cli import LightningCLI
from lightning import AutoregressiveLM, DiffusionLM
from lightning.mock import MockData
from lightning.data import ConcatData
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary, RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme

def cli_main():
    # Custom theme for a cleaner look
    progress_bar = RichProgressBar(
        theme=RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
        ),
        refresh_rate=500 # Radical reduction: update every 500 batches
    )

    cli = LightningCLI(
        trainer_defaults={
            'accelerator': 'gpu',
            'strategy': 'ddp',
            'enable_progress_bar': False, # Disable default to use RichProgressBar
            'log_every_n_steps': 500, # Match progress bar to save disk logs
            'callbacks': [
                ModelCheckpoint(
                    save_top_k=1, # Keep ONLY the best one to save disk space (~1.6GB each!)
                    save_last=True,
                    every_n_train_steps=10000,
                    filename='{epoch}-{step}',
                ),
                ModelSummary(max_depth=1), # Minimal summary
                progress_bar
            ]
        }
    )


if __name__ == "__main__":
    cli_main()
