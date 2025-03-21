import os
import librosa
import numpy as np
from tqdm import tqdm
import numpy as np
import librosa.display
import soundfile as sf
import datasets
import huggingface_hub
import dotenv

dotenv.load_dotenv()
# Login to Hugging Face. Comment this line if you don't want to push to the hub
huggingface_hub.login(os.getenv("HF_TOKEN"))

TARGET_SR = 44100

def audio2mel(filepath: str):
    x, sr = librosa.load(filepath, sr=TARGET_SR, mono=True)
    start, end = 0, 432*512-1

    stft = np.abs(librosa.stft(x[start:end], n_fft=2048, hop_length=512))
    mel = librosa.feature.melspectrogram(sr=sr, S=stft**2, n_mels=128)
    log_mel = librosa.amplitude_to_db(mel)

    return log_mel

def audio2mels(filepath: str, start: int = 0):
    x, sr = librosa.load(filepath, sr=TARGET_SR, mono=True)
    log_mels = []
    for start in range(0, len(x), 432 * 512 - 1):
        end = start + 432 * 512 - 1
        if end >= len(x):
            break
        stft = np.abs(librosa.stft(x[start:end], n_fft=2048, hop_length=512))
        mel = librosa.feature.melspectrogram(sr=sr, S=stft**2, n_mels=128)
        log_mel = librosa.amplitude_to_db(mel)
        log_mels.append(log_mel)

    return log_mels

def find_files(directory: str) -> list[str]:
    files_ = []
    ls = os.listdir(directory)
    for root, dirs, files in os.walk(directory):
        for file in files:
            files_.append(os.path.join(root, file))
    return files_

def make_dataset(files_dir: str | list[str], *, count: int = -1):
    # Writing this as a generator for now - could be useful for large datasets in the future
    if isinstance(files_dir, str):
        files_dir = [files_dir]

    # This ensures that we load the same number of files from each directory
    count = -1 if count < 0 else count // len(files_dir) + 1

    for file_dir in files_dir:
        files = [os.path.abspath(f) for f in find_files(file_dir) if f.lower().endswith('.mp3') or f.lower().endswith('.wav')]
        loaded_count = 0
        total_loaded_count = len(files) if count == -1 else None
        for filepath in tqdm(files, desc="Processing files...", total=total_loaded_count):
            if count > 0 and loaded_count >= count:
                break
            filename = os.path.basename(filepath)

            # Call the function on the MP3 file
            try:
                mels = audio2mels(filepath)
                for i, mel in enumerate(mels):
                    if mel.shape == (128, 432):
                        yield f"{file_dir}-{filename}-{i}", mel
                        loaded_count += 1
                        if count > 0 and loaded_count >= count:
                            break
                    else:
                        print("Skipping shape {}".format(mel.shape))
            except Exception as e:
                print(e)
                pass

def save_spec(dataset: dict, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    np.savez_compressed(os.path.join(save_dir, "specs.npz"), **dataset)

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

def mel_to_audio(spec, sr: float, n_iter: int = 32):
    _check_spec(spec)
    mel = librosa.db_to_amplitude(spec * 80.0)

    mel_basis = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=128)
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

def prepare_dataset(files_dir: str | list[str], save_dir: str, count: int = -1):
    should_push = os.getenv("HF_TOKEN") is not None
    dataset = dict(make_dataset(files_dir, count=count))
    save_spec(dataset, save_dir)

    if not should_push:
        print("HF_TOKEN is not set. Dataset will not be pushed to the hub.")
        # bounce if hugging face not logged in
    else:
        hfds = convert_ds_to_hf_dataset(dataset)
        hfds.push_to_hub("comp5421-mel-spectrogram", private=True)

if __name__ == "__main__":
    prepare_dataset([
        "D:/audio-dataset-v3/audio",
        "./fma_small"
    ], "./output", count=50000)
