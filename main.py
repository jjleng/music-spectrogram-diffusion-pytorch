import torch
from pytorch_lightning.cli import LightningCLI
from lightning import AutoregressiveLM, DiffusionLM
from lightning.mock import MockData
from lightning.data import ConcatData

def cli_main():
    """
    Minimal CLI entry point.
    ALL configuration (callbacks, trainer settings) comes from the YAML config file.
    This avoids any conflicts between code and config.
    """
    cli = LightningCLI()


if __name__ == "__main__":
    cli_main()
