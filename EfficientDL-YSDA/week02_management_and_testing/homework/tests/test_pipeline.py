import pytest
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

from modeling.diffusion import DiffusionModel
from modeling.training import generate_samples, train_epoch, train_step
from modeling.unet import UnetModel


@pytest.fixture
def train_dataset():
    transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = CIFAR10(
        "./data",
        train=True,
        download=True,
        transform=transforms,
    )
    return dataset


@pytest.mark.parametrize(["device"], [["cpu"], ["cuda"]])
def test_train_on_one_batch(device, train_dataset):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    # note: you should not need to increase the threshold or change the hyperparameters
    ddpm = DiffusionModel(
        eps_model=UnetModel(3, 3, hidden_size=32),
        betas=(1e-4, 0.02),
        num_timesteps=1000,
    )
    ddpm.to(device)

    optim = torch.optim.Adam(ddpm.parameters(), lr=5e-4)
    dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    x, _ = next(iter(dataloader))
    loss = None
    for i in range(50):
        loss = train_step(ddpm, x, optim, device)
    assert loss < 0.5


@pytest.mark.parametrize(
    "device,learning_rate,should_change",
    [
        # lr=0 is our control case: whole training step works but weights stay the same
        ("cpu", 0.0, False),
        # with normal lr optimizer should actually update at least one weight
        ("cpu", 1e-3, True),
        # same real training case but on gpu when cuda is available
        ("cuda", 1e-3, True),
    ],
)
def test_training(
    device,
    learning_rate,
    should_change,
    train_dataset,
    tmp_path,
):
    # mac has no cuda so i skip this case locally but it will run on gpu
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    # make model initialization noise and timesteps the same between test runs
    torch.manual_seed(42)

    # we need a real ddpm here but a very small one is enough for integration test
    # hidden_size=8 is the smallest valid size for our group norm with 8 groups
    # and 2 timesteps make sampling fast instead of running unet 1000 times
    model = DiffusionModel(
        eps_model=UnetModel(3, 3, hidden_size=8),
        betas=(1e-4, 0.02),
        num_timesteps=2,
    ).to(device)

    # we do not need all 50k cifar images to check that training pipeline works
    # 4 images with batch_size=2 give exactly 2 train steps
    dataset_subset = Subset(train_dataset, range(4))
    dataloader = DataLoader(
        dataset_subset,
        batch_size=2,
        shuffle=False,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    # clone creates a real snapshot of weights before training
    # without clone we would keep links to the same weights and compare them to themselves
    parameters_before = [parameter.detach().clone() for parameter in model.parameters()]

    # this goes through the complete train_epoch -> train_step -> model pipeline
    train_epoch(model, dataloader, optimizer, device)

    parameters_after = list(model.parameters())

    # zip pairs every weight before training with the same weight after training
    # torch.equal returns True if this pair is exactly the same
    # not turns it into True when the weight has changed
    # and any returns True if at least one of all these weight pairs has changed
    parameters_changed = any(
        not torch.equal(before, after)
        for before, after in zip(parameters_before, parameters_after)
    )

    # lr=0 should keep weights unchanged and lr>0 should update them
    assert parameters_changed == should_change

    # tmp_path is a temporary pytest folder so test images do not stay in my repo
    # generate_samples should run sampling and save a real non-empty png file
    output_path = tmp_path / f"samples_{device}_{learning_rate}.png"
    generate_samples(model, device, str(output_path))

    assert output_path.exists()
    assert output_path.stat().st_size > 0
