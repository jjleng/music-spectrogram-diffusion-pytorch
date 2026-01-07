import pytorch_lightning as pl
import torch
import pickle
import hashlib
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset, default_collate, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from itertools import chain
from data.musicnet import MusicNet
from data.maestro import Maestro
from data.urmp import URMP
from data.slakh import Slakh2100
from data.guitarset import GuitarSet
from preprocessor.event_codec import Codec


def _get_cache_key(dataset_class: str, path: str, split: str, **kwargs) -> str:
    """Generate a unique cache key based on dataset parameters."""
    key_data = f"{dataset_class}_{path}_{split}_{kwargs}"
    return hashlib.md5(key_data.encode()).hexdigest()[:16]


def _load_cached_dataset(cache_dir: Path, cache_key: str):
    """Load a cached dataset if it exists."""
    cache_file = cache_dir / f"{cache_key}.pkl"
    if cache_file.exists():
        print(f"Loading cached dataset from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None


def _save_cached_dataset(cache_dir: Path, cache_key: str, dataset) -> None:
    """Save a dataset to cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{cache_key}.pkl"
    print(f"Saving dataset cache to {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(dataset, f)


def get_padding_collate_fn(output_size: int):
    def collate_fn(batch):
        """Pad the batch to the longest sequence."""
        seqs = [item[0] for item in batch]
        rest = [item[1:] for item in batch]
        rest = default_collate(rest)
        if output_size is not None:
            seqs = [torch.cat([seq, seq.new_zeros(output_size - len(seq))])
                    if len(seq) < output_size else seq[:output_size] for seq in seqs]
            seqs = torch.stack(seqs, dim=0)
        else:
            seqs = pad_sequence(seqs, batch_first=True, padding_value=0)
        return seqs, *rest
    return collate_fn


class ConcatData(pl.LightningDataModule):
    def __init__(self,
                 batch_size: int,
                 midi_output_size: int = 2048,
                 with_context: bool = False,
                 sample_rate: int = 16000,
                 segment_length: int = 81920,
                 musicnet_path: str = None,
                 maestro_path: str = None,
                 slakh_path: str = None,
                 guitarset_path: str = None,
                 urmp_wav_path: str = None,
                 urmp_midi_path: str = None,
                 sampling_temperature: float = 0.3,
                 cache_dir: str = None,
                 ):
        super().__init__()
        self.save_hyperparameters()
        # Always cache: use provided dir or default to local .dataset_cache
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".dataset_cache")

    def _load_or_create_dataset(self, dataset_class, split: str, factory_kwargs: dict, **dataset_kwargs):
        """Load dataset from cache or create it."""
        # Build cache key from all relevant parameters
        path_str = str(dataset_kwargs.get('path', dataset_kwargs.get('wav_path', '')))
        cache_key = _get_cache_key(
            dataset_class.__name__,
            path_str,
            split,
            sample_rate=self.hparams.sample_rate,
            segment_length=self.hparams.segment_length,
            with_context=self.hparams.with_context,
        )

        # Try loading from cache
        if self.cache_dir:
            cached = _load_cached_dataset(self.cache_dir, cache_key)
            if cached is not None:
                return cached

        # Create dataset
        print(f"Creating {dataset_class.__name__} ({split})...")
        dataset = dataset_class(split=split, **factory_kwargs, **dataset_kwargs)

        # Save to cache
        if self.cache_dir:
            _save_cached_dataset(self.cache_dir, cache_key, dataset)

        return dataset

    def setup(self, stage=None):
        resolution = 100
        segment_length_in_time = self.hparams.segment_length / self.hparams.sample_rate
        codec = Codec(int(segment_length_in_time * resolution + 1))

        factory_kwargs = {
            'codec': codec,
            'resolution': resolution,
            'sample_rate': self.hparams.sample_rate,
            'segment_length': self.hparams.segment_length,
            'with_context': self.hparams.with_context,
        }

        if stage == "fit":
            train_datasets = []
            if self.hparams.musicnet_path is not None:
                train_datasets.append(self._load_or_create_dataset(
                    MusicNet, 'train', factory_kwargs, path=self.hparams.musicnet_path))

            if self.hparams.maestro_path is not None:
                train_datasets.append(self._load_or_create_dataset(
                    Maestro, 'train', factory_kwargs, path=self.hparams.maestro_path))

            if self.hparams.urmp_wav_path is not None and self.hparams.urmp_midi_path is not None:
                train_datasets.append(self._load_or_create_dataset(
                    URMP, 'train', factory_kwargs,
                    wav_path=self.hparams.urmp_wav_path, midi_path=self.hparams.urmp_midi_path))

            if self.hparams.slakh_path is not None:
                train_datasets.append(self._load_or_create_dataset(
                    Slakh2100, 'train', factory_kwargs, path=self.hparams.slakh_path))

            if self.hparams.guitarset_path is not None:
                train_datasets.append(self._load_or_create_dataset(
                    GuitarSet, 'train', factory_kwargs, path=self.hparams.guitarset_path))

            train_num_samples = [len(dataset) for dataset in train_datasets]
            dataset_weights = [
                x ** self.hparams.sampling_temperature for x in train_num_samples]

            print("Train dataset sizes: ", train_num_samples)
            print("Train dataset weights: ", dataset_weights)

            self.sampler_weights = list(
                chain.from_iterable(
                    [dataset_weights[i] / train_num_samples[i]] * train_num_samples[i] for i in range(len(train_num_samples))
                )
            )

            self.train_dataset = ConcatDataset(train_datasets)

        if stage == "validate" or stage == "fit":
            val_datasets = []
            if self.hparams.musicnet_path is not None:
                val_datasets.append(self._load_or_create_dataset(
                    MusicNet, 'val', factory_kwargs, path=self.hparams.musicnet_path))

            if self.hparams.maestro_path is not None:
                val_datasets.append(self._load_or_create_dataset(
                    Maestro, 'val', factory_kwargs, path=self.hparams.maestro_path))

            if self.hparams.urmp_wav_path is not None and self.hparams.urmp_midi_path is not None:
                val_datasets.append(self._load_or_create_dataset(
                    URMP, 'val', factory_kwargs,
                    wav_path=self.hparams.urmp_wav_path, midi_path=self.hparams.urmp_midi_path))

            if self.hparams.slakh_path is not None:
                val_datasets.append(self._load_or_create_dataset(
                    Slakh2100, 'val', factory_kwargs, path=self.hparams.slakh_path))

            if self.hparams.guitarset_path is not None:
                val_datasets.append(self._load_or_create_dataset(
                    GuitarSet, 'val', factory_kwargs, path=self.hparams.guitarset_path))

            self.val_dataset = ConcatDataset(val_datasets)

        if stage == "test":
            test_datasets = []
            if self.hparams.musicnet_path is not None:
                test_datasets.append(self._load_or_create_dataset(
                    MusicNet, 'test', factory_kwargs, path=self.hparams.musicnet_path))

            if self.hparams.maestro_path is not None:
                test_datasets.append(self._load_or_create_dataset(
                    Maestro, 'test', factory_kwargs, path=self.hparams.maestro_path))

            if self.hparams.urmp_wav_path is not None and self.hparams.urmp_midi_path is not None:
                test_datasets.append(self._load_or_create_dataset(
                    URMP, 'test', factory_kwargs,
                    wav_path=self.hparams.urmp_wav_path, midi_path=self.hparams.urmp_midi_path))

            if self.hparams.slakh_path is not None:
                test_datasets.append(self._load_or_create_dataset(
                    Slakh2100, 'test', factory_kwargs, path=self.hparams.slakh_path))

            if self.hparams.guitarset_path is not None:
                test_datasets.append(self._load_or_create_dataset(
                    GuitarSet, 'test', factory_kwargs, path=self.hparams.guitarset_path))

            self.test_dataset = ConcatDataset(test_datasets)

    def train_dataloader(self):
        collate_fn = get_padding_collate_fn(self.hparams.midi_output_size)
        sampler = WeightedRandomSampler(self.sampler_weights, len(
            self.sampler_weights), replacement=True)
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size,
                          sampler=sampler,
                          shuffle=False, num_workers=4, collate_fn=collate_fn)

    def val_dataloader(self):
        collate_fn = get_padding_collate_fn(self.hparams.midi_output_size)
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    def test_dataloader(self):
        collate_fn = get_padding_collate_fn(self.hparams.midi_output_size)
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
