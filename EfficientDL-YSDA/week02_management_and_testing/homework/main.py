from pathlib import Path

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from modeling.diffusion import DiffusionModel
from modeling.training import generate_samples, train_epoch
from modeling.unet import UnetModel


# choose cuda if it exists and cpu if it does not
def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


# build the optimizer from the config
def build_optimizer(model: DiffusionModel, cfg: DictConfig):
    if cfg.name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.peak_lr,
            betas=tuple(cfg.betas),
            weight_decay=cfg.weight_decay,
        )

    if cfg.name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.peak_lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )

    raise ValueError(f"Unknown optimizer: {cfg.name}")


# hydra builds cfg from files in the conf folder
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # print the final config before doing anything
    print(OmegaConf.to_yaml(cfg))

    # stop here if i only want to check the config
    if not cfg.training.enabled:
        print("Training is disabled. Set training.enabled=true to start it.")
        return

    # fix random values and choose the device
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)

    # build unet and ddpm from the config
    ddpm = DiffusionModel(
        eps_model=UnetModel(
            cfg.model.in_channels,
            cfg.model.out_channels,
            hidden_size=cfg.model.hidden_size,
        ),
        betas=(
            cfg.diffusion.beta_start,
            cfg.diffusion.beta_end,
        ),
        num_timesteps=cfg.diffusion.num_timesteps,
    ).to(device)

    # add random flips only when they are enabled
    train_transforms = []
    if cfg.data.random_flip:
        train_transforms.append(transforms.RandomHorizontalFlip())

    # turn images into tensors from minus one to one
    train_transforms.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5),
            ),
        ]
    )

    # download cifar10 if it is missing
    dataset = CIFAR10(
        cfg.data.root,
        train=True,
        download=True,
        transform=transforms.Compose(train_transforms),
    )

    # take dataloader settings from the config
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        shuffle=True,
    )
    optimizer = build_optimizer(ddpm, cfg.optimizer)

    # start wandb only when it is enabled
    run = None
    if cfg.wandb.enabled:
        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.run_name,
            group=cfg.wandb.group,
            tags=list(cfg.wandb.tags),
            mode=cfg.wandb.mode,
            config=OmegaConf.to_container(cfg, resolve=True),
            save_code=cfg.wandb.save_code,
        )

    # create the folder before saving images
    samples_dir = Path("samples")
    samples_dir.mkdir(parents=True, exist_ok=True)

    try:
        for epoch in range(cfg.training.num_epochs):
            # train for one full epoch
            train_epoch(ddpm, dataloader, optimizer, device)

            # generate images on the selected epochs
            if (epoch + 1) % cfg.sampling.every_n_epochs == 0:
                sample_path = samples_dir / f"{epoch:02d}.png"
                generate_samples(ddpm, device, str(sample_path))

                # send the image to wandb
                if run is not None and (epoch + 1) % cfg.wandb.log_samples_every == 0:
                    run.log(
                        {
                            "epoch": epoch + 1,
                            "samples": wandb.Image(str(sample_path)),
                        }
                    )
            elif run is not None:
                run.log({"epoch": epoch + 1})
    finally:
        # close wandb even if training fails
        if run is not None:
            run.finish()


# run hydra from the terminal
if __name__ == "__main__":
    main()
