import jams
import os
from tqdm import tqdm
import note_seq
from note_seq.midi_io import midi_to_note_sequence
import pretty_midi
import soundfile as sf

from .common import Base


def get_noteseq(x: jams.JAMS):
    tmp = [i["data"]
           for i in x["annotations"] if i["namespace"] == "note_midi"]

    midi_data = pretty_midi.PrettyMIDI(initial_tempo=120)
    for note_list in tmp:
        inst = pretty_midi.Instrument(
            program=25, is_drum=False, name='acoustic guitar (steel)')
        midi_data.instruments.append(inst)
        for note in note_list:
            inst.notes.append(pretty_midi.Note(
                120, round(note[2]), note[0], note[0] + note[1]))
    noteseq = midi_to_note_sequence(midi_data)
    return noteseq


class GuitarSet(Base):
    def __init__(self,
                 path: str = "/import/c4dm-datasets/GuitarSet",
                 split: str = "train",
                 **kwargs):
        data_list = []
        file_names = os.listdir(f"{path}/annotation")
        if split == "train":
            file_names = [
                file for file in file_names if file.split("-")[0][-1] != "3"]
        elif split == "val" or split == "test":
            file_names = [
                file for file in file_names if file.split("-")[0][-1] == "3"]
        else:
            raise ValueError(f'Invalid split: {split}')

        for file in tqdm(file_names):
            try:
                tmp = jams.load(f"{path}/annotation/{file}")

                # Use JAMS filename as base (more reliable than internal title)
                base_name = file.replace('.jams', '')

                # Try multiple audio file patterns
                candidate_names = [
                    f"{base_name}_mic.wav",
                    f"{base_name}.wav",
                    f"{base_name}_mix.wav",
                ]
                candidate_dirs = ["audio_mono-mic", "audio", "audio_hex-pickup_original", "audio_mono-pickup_original"]
                found_wav = None

                for d in candidate_dirs:
                    for name in candidate_names:
                        p = os.path.join(path, d, name)
                        if os.path.exists(p):
                            found_wav = p
                            break
                    if found_wav:
                        break

                # If not found in expected dirs, try recursive search (slow fallback)
                if not found_wav:
                    for root, _, files in os.walk(path):
                        for file_name in files:
                            if file_name.startswith(base_name) and file_name.endswith(".wav"):
                                found_wav = os.path.join(root, file_name)
                                break
                        if found_wav: break

                if not found_wav:
                    # Audio file missing - skip this annotation
                    print(f"⚠️  Skip {file}: Audio not found for '{base_name}'")
                    continue

                wav_file = found_wav

                info = sf.info(wav_file)
                sr = info.samplerate
                frames = info.frames
                ns = get_noteseq(tmp)
                ns = note_seq.apply_sustain_control_changes(ns)
                data_list.append((wav_file, ns, sr, frames))
            except Exception as e:
                print(f"⚠️  Skip {file}: {str(e)[:60]}")
                continue

        super().__init__(data_list, **kwargs)
