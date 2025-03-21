import librosa
import numpy as np
import os
from tqdm import tqdm
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import datasets

TARGET_SR = 44100

def audio2mel(filepath: str, start: int = 0):
    #TODO load different parts of the audio, not just the beginning
    x, sr = librosa.load(filepath, sr=TARGET_SR, mono=True)
    start, end = 0, 432*512-1

    stft = np.abs(librosa.stft(x[start:end], n_fft=2048, hop_length=512))
    mel = librosa.feature.melspectrogram(sr=sr, S=stft**2, n_mels=128)
    log_mel = librosa.amplitude_to_db(mel)

    return log_mel

def make_dataset(files_dir: str, *, count: int = -1):
    all_mels = {}
    loaded_count = 0
    files = os.listdir(files_dir)
    for file in tqdm(files, desc="Processing files...", total=len(files) if count == -1 else min(count, len(files))):
        if not file.lower().endswith('.mp3') and not file.lower().endswith(".wav"):  # Adds wav support
            continue
        if count > 0 and loaded_count >= count:
            break
        filepath = os.path.join(files_dir, file)
        filename = os.path.basename(filepath)

        # Call the function on the MP3 file
        try:
            mel, sr = audio2mel(filepath)
            if mel.shape == (128, 432):
                all_mels[filename + f'_{sr}'] = mel
                loaded_count += 1
            else:
                print("Skipping shape {}".format(mel.shape))
        except Exception as e:
            print(e)
            pass

    return all_mels

def save_spec(dataset, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    np.savez_compressed(os.path.join(save_dir, "specs.npz"), **ds)

def _check_spec(spec):
    # To keep my sanity
    assert spec.shape == (128, 432), f"Shape is {spec.shape}, expected (128, 432)"
    assert spec.dtype == np.float32, f"Data type is {spec.dtype}, expected np.float32"
    assert np.isfinite(spec).all(), "Data contains non-finite values"
    assert np.abs(spec).max() <= 80, "Data contains values greater than 80 dB"
    assert np.abs(spec).max() > 1, "Empty data, or you probably forget to unnormalize it"

def load_spec(spec_file: str):
    ds = np.load(spec_file)
    dsdict = {}
    for key in ds:
        try:
            _check_spec(ds[key])
        except Exception as e:
            print(f"Error in {key}: {e}")
            continue
        dsdict[key] = ds[key] / 80.0 # Normalize to [-1, 1]
    return dsdict

def mel_to_audio(spec, sr: int, n_iter: int = 32):
    _check_spec(spec)
    mel = librosa.db_to_amplitude(spec * 80.0)

    mel_basis = librosa.filters.mel(sr, n_fft=2048, n_mels=128)
    inv_mel_basis = np.linalg.pinv(mel_basis)
    stft_magnitude = np.dot(inv_mel_basis, mel)

    stft_magnitude_squared = stft_magnitude**2
    audio = librosa.griffinlim(stft_magnitude_squared, hop_length=512, n_iter=n_iter)

    return audio

def convert_ds_to_hf_dataset(ds: dict):
    hfds_dict = {"filename": [], "mel": []}
    for key in ds:
        hfds_dict["filename"].append(key)
        hfds_dict["mel"].append(ds[key])
    hf_ds = datasets.Dataset.from_dict(hfds_dict)
    return hf_ds
