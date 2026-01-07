import torch
import torchaudio
from torch.utils.data import Dataset
import soundfile as sf
from pathlib import Path
import resampy
import numpy as np
from typing import Tuple, Union, Optional, List, Dict, Any
from note_seq import NoteSequence
from tqdm import tqdm

from preprocessor.event_codec import Codec
from preprocessor.preprocessor import preprocess


class Base(Dataset):
    def __init__(self,
                 data_list: List[Tuple[Path, NoteSequence, int, int]],
                 codec: Codec,
                 sample_rate: int = None,
                 segment_length: int = 81920,
                 with_context: bool = False,
                 **kwargs):
        super().__init__()

        boundaries = [0]
        self.data_list = []

        total_num_tokens = 0
        empty_chunks = 0
        print("Preprocessing data...")
        for filename, midi, sr, frames in tqdm(data_list):
            if sample_rate is None:
                segment_length_in_time = segment_length / sr
            else:
                segment_length_in_time = segment_length / sample_rate
            num_chunks = int(frames / (segment_length_in_time * sr))
            tokens, _ = preprocess(
                midi, segment_length=segment_length_in_time, codec=codec, **kwargs)
            num_chunks = min(num_chunks, len(tokens))
            empty_chunks += sum([1 for t in tokens if len(t) <= 1])
            boundaries.append(boundaries[-1] + num_chunks)
            self.data_list.append(
                (filename, tokens, sr, segment_length_in_time))
            total_num_tokens += sum([len(t) for t in tokens])

        print(
            f'Total number of tokens: {total_num_tokens}, total number of chunks: {boundaries[-1]}, empty chunks: {empty_chunks}')
        self.boundaries = np.array(boundaries)
        self.sr = sample_rate
        self.segment_length = segment_length
        self.with_context = with_context

    def __len__(self) -> int:
        return self.boundaries[-1]

    def _get_file_idx_and_chunk_idx(self, index: int) -> Tuple[int, int]:
        bin_pos = np.digitize(index, self.boundaries[1:], right=False)
        chunk_index = index - self.boundaries[bin_pos]
        return bin_pos, chunk_index

    def _get_waveforms(self, index: int, chunk_index: int) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Get waveform without resampling."""
        wav_file, _, sr, length_in_time = self.data_list[index]
        offset = int(chunk_index * length_in_time * sr)
        frames = int(length_in_time * sr)

        if not self.with_context:
            data, _ = sf.read(
                str(wav_file), start=offset, frames=frames, dtype='float32', always_2d=True)
            data = data.mean(axis=1)
            return data

        ctx_offset = offset - frames
        if ctx_offset >= 0:
            ctx, _ = sf.read(str(wav_file), start=ctx_offset,
                             frames=frames * 2, dtype='float32', always_2d=True)
            data = ctx[frames:]
            ctx = ctx[:frames]
        else:
            data, _ = sf.read(
                str(wav_file), start=offset, frames=frames, dtype='float32', always_2d=True)
            ctx = np.zeros_like(data)
        data = data.mean(axis=1)
        ctx = ctx.mean(axis=1)
        return data, ctx

    def __getitem__(self, index: int) -> torch.Tensor:
        file_idx, chunk_idx = self._get_file_idx_and_chunk_idx(index)
        tokens = self.data_list[file_idx][1][chunk_idx]
        source_sr = self.data_list[file_idx][2]  # Source file sample rate

        if not self.with_context:
            data = self._get_waveforms(file_idx, chunk_idx)
            if source_sr != self.sr:
                # Resample using actual sample rates
                data_t = torch.from_numpy(data)
                data_t = torchaudio.functional.resample(data_t, source_sr, self.sr)
                data = data_t.numpy()

            # Ensure exact length (handle rounding differences)
            if data.shape[0] > self.segment_length:
                data = data[:self.segment_length]
            elif data.shape[0] < self.segment_length:
                data = np.pad(data, (0, self.segment_length - data.shape[0]), 'constant')
            return tokens, data

        data, ctx = self._get_waveforms(file_idx, chunk_idx)

        if source_sr != self.sr:
            # Stack for efficiency: (2, T)
            tmp = np.vstack([ctx, data])
            tmp_t = torch.from_numpy(tmp)
            tmp_t = torchaudio.functional.resample(tmp_t, source_sr, self.sr)
            tmp = tmp_t.numpy()
            ctx, data = tmp[0], tmp[1]

        # Ensure exact length
        if data.shape[0] > self.segment_length:
            data = data[:self.segment_length]
            ctx = ctx[:self.segment_length]
        elif data.shape[0] < self.segment_length:
            data = np.pad(data, (0, self.segment_length - data.shape[0]), 'constant')
            ctx = np.pad(ctx, (0, self.segment_length - ctx.shape[0]), 'constant')

        return tokens, data, ctx
