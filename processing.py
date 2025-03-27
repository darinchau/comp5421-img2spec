import os
import librosa
import numpy as np
from tqdm import tqdm, trange
import numpy as np
import librosa.display
import soundfile as sf
import datasets
import huggingface_hub
import dotenv
from typing import Iterator
from dataclasses import dataclass, asdict
import gc
import time
import torch
from diffusers import DDPMScheduler, UNet2DModel, DDIMScheduler, DiffusionPipeline #type: ignore

SPECTROGRAM_SHAPE = (128, 216)

dotenv.load_dotenv()
# Login to Hugging Face. Comment this line if you don't want to push to the hub
huggingface_hub.login(os.getenv("HF_TOKEN"))

TARGET_SR = 44100

def audio2mel(filepath: str):
    x, sr = librosa.load(filepath, sr=TARGET_SR, mono=True)
    nmel, nframe = SPECTROGRAM_SHAPE
    start, end = 0, nframe * 512 - 1

    stft = np.abs(librosa.stft(x[start:end], n_fft=2048, hop_length=512))
    mel = librosa.feature.melspectrogram(sr=sr, S=stft**2, n_mels=nmel)
    log_mel = librosa.amplitude_to_db(mel)

    return log_mel

def audio2mels(filepath: str, start: int = 0):
    x, sr = librosa.load(filepath, sr=TARGET_SR, mono=True)
    nmel, nframe = SPECTROGRAM_SHAPE
    log_mels = []
    for start in range(0, len(x), nframe * 512 - 1):
        end = start + nframe * 512 - 1
        if end >= len(x):
            break
        stft = np.abs(librosa.stft(x[start:end], n_fft=2048, hop_length=512))
        mel = librosa.feature.melspectrogram(sr=sr, S=stft**2, n_mels=nmel)
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
    files: list[tuple[str, str]] = []

    for file_dir in files_dir:
        files_ = [(file_dir, os.path.abspath(f)) for f in find_files(file_dir) if f.lower().endswith('.mp3') or f.lower().endswith('.wav')]
        files.extend(files_)

    loaded_count = 0
    for file_dir, filepath in tqdm(files, desc="Processing files...", total=len(files) if count == -1 else count):
        if count > 0 and loaded_count >= count:
            break
        filename = os.path.basename(filepath)

        # Call the function on the MP3 file
        try:
            mels = audio2mels(filepath)
            for i, mel in enumerate(mels):
                try:
                    _check_spec(mel)
                    yield {
                        "filename": f"{file_dir}-{filename}-{i}",
                        "mel": mel
                    }
                    loaded_count += 1
                except Exception as e:
                    tqdm.write(f"Error in {filename}: {e}")
        except Exception as e:
            tqdm.write(str(e))
            pass

def convert_ds_stream_to_dict(ds_stream):
    ds_dict = {}
    for ds in ds_stream:
        _check_spec(ds["mel"])
        ds_dict[ds["filename"]] = ds["mel"]
    return ds_dict

def save_spec(dataset, save_dir: str):
    if not isinstance(dataset, dict):
        convert_ds_stream_to_dict(dataset)
    for key in dataset:
        _check_spec(dataset[key])
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "specs.npz")
    np.savez_compressed(path, **dataset)
    return path

def _check_spec(spec):
    # To keep my sanity
    assert isinstance(spec, np.ndarray), f"Data is not a numpy array, but {type(spec)}"
    assert spec.shape == SPECTROGRAM_SHAPE, f"Shape is {spec.shape}, expected {SPECTROGRAM_SHAPE}"
    assert spec.dtype == np.float32, f"Data type is {spec.dtype}, expected np.float32"
    assert np.isfinite(spec).all(), "Data contains non-finite values"
    assert np.abs(spec).max() <= 80, "Data contains values greater than 80 dB"
    assert np.abs(spec).max() > 1, "Empty data, or you probably forget to unnormalize it"

def load_ds(spec_file: str):
    ds = np.load(spec_file)
    for key in ds:
        mel = np.array(ds[key])
        mel = mel.astype(np.float32)
        try:
            _check_spec(mel)
        except Exception as e:
            print(f"Error in {key}: {e}")
            continue
        yield {
            "filename": key,
            "mel": mel
        }

def mel_to_audio(spec, sr: float, n_iter: int = 32):
    _check_spec(spec)
    mel = librosa.db_to_amplitude(spec)
    nmel, nframe = SPECTROGRAM_SHAPE
    mel_basis = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=nmel)
    inv_mel_basis = np.linalg.pinv(mel_basis)
    stft_magnitude = np.dot(inv_mel_basis, mel)

    stft_magnitude_squared = stft_magnitude**2
    audio = librosa.griffinlim(stft_magnitude_squared, hop_length=512, n_iter=n_iter)

    return audio

def convert_ds_to_hf_dataset(ds, batch_size=100):
    # Define features
    features = datasets.Features({
        "filename": datasets.Value("string"),
        "mel": datasets.Array2D(shape=SPECTROGRAM_SHAPE, dtype="float32")
    })

    if isinstance(ds, dict):
        def generator():
            for key in ds.items():
                yield {
                    "filename": key,
                    "mel": ds[key]
                }
        ds_ = generator()
    else:
        ds_ = ds

    # Initialize empty dataset
    hf_ds = datasets.Dataset.from_dict({"filename": [], "mel": []}, features=features)

    while True:
        batch_dict = {
            "filename": [],
            "mel": []
        }
        _stop = False
        for i in range(batch_size):
            try:
                data = next(ds_)
            except StopIteration:
                _stop = True
                break
            batch_dict["filename"].append(data["filename"])
            batch_dict["mel"].append(data["mel"])

        # Create a small dataset from the batch and concatenate it
        batch_ds = datasets.Dataset.from_dict(batch_dict, features=features)
        hf_ds = datasets.concatenate_datasets([hf_ds, batch_ds])
        if _stop:
            break

    return hf_ds

def prepare_dataset(files_dir: str | list[str], hf_hub_dir: str, count: int = -1):
    hfds = convert_ds_to_hf_dataset(make_dataset(files_dir, count=count))
    hfds.push_to_hub(hf_hub_dir, private=True)

def inference(
    model: UNet2DModel | None = None,
    model_repo: str | None = None,
    seed: int | None = None,
    sample_steps: int = 100,
    sample_step_start: int = 1000,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    sample_count: int = 1,
    verbose: bool = False,
    latents: torch.Tensor | None = None,
    return_intermediate_steps: bool = False
):
    """
    Perform inference using the DDIM model

    Args:
        model (UNet2DModel): The model to use for inference
        model_repo (str): The model repository to load the model from
        seed (int): The seed to use for the random number generator
        sample_steps (int): The number of steps to sample
        device (torch.device): The device to use for inference
        sample_count (int): The number of samples to generate
        verbose (bool): Whether to print verbose output

    Return:
        dict: A dictionary containing the latents
        dict['latents'] (np.ndarray): The latents generated (shape: (sample_count, 1, *SPECTROGRAM_SHAPE))
        dict['intermediates'] (np.ndarray): The intermediate steps generated (shape: (sample_count, sample_steps, *SPECTROGRAM_SHAPE)). Only returned if return_intermediate_steps is True
    """
    if model is None and model_repo is None:
        raise ValueError("Either model or model_repo must be provided")

    if model is None and model_repo is not None:
        model = UNet2DModel.from_pretrained(model_repo).to(device) #type: ignore

    steps = sample_steps
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    sampler = DDIMScheduler(
        num_train_timesteps=sample_step_start,
    )

    nmel, nframe = SPECTROGRAM_SHAPE

    if latents is None:
        latents = torch.randn((sample_count, 1, nmel, nframe), device=device, generator=generator, dtype=torch.float32)
        latents = latents * sampler.init_noise_sigma

    sampler.set_timesteps(steps)
    timesteps = sampler.timesteps.to(device)

    extra_step_kwargs = {
        'eta': 0.0,
        'generator': generator
    }

    intermediates = []

    for i, t in enumerate(tqdm(timesteps, desc="DDIM Sampling:", disable=not verbose)):
        model_input = latents

        timestep_tensor = torch.tensor([t], dtype=torch.long, device=device)
        timestep_tensor = timestep_tensor.expand(latents.shape[0])
        with torch.no_grad():
            noise_pred = model(model_input, timestep_tensor)[0] #type: ignore

        if return_intermediate_steps:
            intermediates.append(latents.detach().cpu().numpy())

        latents = sampler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample #type: ignore

    out = {
        "latents":  latents.detach().cpu().numpy() * 80.
    }

    if return_intermediate_steps:
        intermediates = np.concatenate(intermediates, axis=1)
        out["intermediates"] = intermediates * 80.

    return out

if __name__ == "__main__":
    prepare_dataset([
        # "D:/audio-dataset-v3/audio",
        "./fma_small"
    ], "comp5421-mel-spectrogram-fma_small-128x216", count=-1)
