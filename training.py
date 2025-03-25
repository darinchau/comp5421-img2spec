
import os
import torch
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from diffusers import DDPMScheduler, UNet2DModel
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import huggingface_hub
from tqdm import tqdm, trange
from dataclasses import asdict
import wandb
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class COMP5421Config():
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 1e-4
    img_dims: tuple[int, int] = (128, 432)
    dataset_src: str = "darinchau/comp5421-mel-spectrogram"
    training_name: str = "comp5421-project"
    val_size: float = 0.1
    val_step: int = 256  # Validate every n steps
    val_samples: float = 256  # Validate over n samples instead of the whole val set
    save_step: int = 512
    load_model_from: str | None = "darinchau/comp5421-project-sage-lake-20-comp5421-mel-spectrogram-step-2560"


huggingface_hub.login(os.getenv("HF_TOKEN"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate(config, model, loader, noise_scheduler, loss_func):
    val_loss = 0.0
    val_count = 0
    for val_batch in tqdm(loader, desc="Validating...", total=config.val_step):
        val_count += 1
        noise = torch.randn_like(val_batch)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (val_batch.size(0),), device=device, dtype=torch.int64)
        noisy_batch = noise_scheduler.add_noise(val_batch, noise, timesteps)
        noise_pred = model(noisy_batch, timesteps)[0]
        loss = loss_func(noise_pred, noise)
        val_loss += loss
        if val_count >= config.val_step:
            break
    return val_loss / val_count


def main():
    config = COMP5421Config()
    model = UNet2DModel(
        sample_size=config.img_dims,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(32, 64, 64),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D")
    )

    if config.load_model_from is not None:
        model = UNet2DModel.from_pretrained(config.load_model_from)
        step_count = int(config.load_model_from.split("-")[-1])
        print(f"Resuming training from checkpoint: {config.load_model_from} (Step: {step_count})")
    else:
        step_count = 0
        print("Starting training anew...")

    model = model.to(device)

    # Load the dataset
    # If you need this dataset lmk - Darin
    dataset = load_dataset(config.dataset_src)

    def collate_fn(batch):
        mels = [torch.tensor(item['mel']).unsqueeze(0) for item in batch]  # Adding channel dimension
        mels = torch.stack(mels).to(device)  # Shape will be [batch_size, 1, 128, 432]
        mels = mels/80. # Normalize to [-1, 1]
        return mels

    train_test = dataset['train'].train_test_split(test_size=config.val_size)
    train_loader = DataLoader(train_test['train'], batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(train_test['test'], batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_func = torch.nn.MSELoss()

    api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=api_key)

    # Comment out this line if you dont need logging
    wandb.init(
        project=config.training_name,
        config=asdict(config)
    )

    # Training Loop
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.num_epochs}'):
            step_count += 1
            optimizer.zero_grad()

            # Add noise
            noise = torch.randn_like(batch)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch.size(0),), device=device, dtype=torch.int64)
            noisy_batch = noise_scheduler.add_noise(batch, noise, timesteps)

            # Forward pass
            noise_pred = model(noisy_batch, timesteps)[0]

            # Loss
            loss = loss_func(noise_pred, noise)
            loss.backward()
            optimizer.step()

            if step_count > 0 and step_count % config.val_step == 0:
                with torch.no_grad():
                    try:
                        val_loss = validate(config, model, val_loader, noise_scheduler, loss_func)
                        wandb.log({"val_batch_loss": val_loss.item()}, step=step_count)
                    except Exception as e:
                        tqdm.write(f"Failed to perform validation... {e}")

            if step_count > 0 and step_count % config.save_step == 0:
                model.push_to_hub(f"{config.training_name}-{wandb.run.name}-{config.dataset_src.split('/')[1]}-step-{step_count}")

            epoch_loss += loss.item()
            wandb.log({"batch_loss": loss.item()}, step=step_count)

        average_epoch_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch + 1} completed, Average Loss: {average_epoch_loss}')
        wandb.log({"epoch_loss": average_epoch_loss}, step=step_count)

    model.push_to_hub(f"{config.training_name}-{wandb.run.name}-{config.dataset_src.split('/')[1]}")
    wandb.finish()
    print("Training completed.")


if __name__ == "__main__":
    main()
