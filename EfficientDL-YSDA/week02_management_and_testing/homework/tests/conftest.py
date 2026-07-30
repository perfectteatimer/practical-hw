import pytest
import torch


@pytest.fixture(autouse=True)
def set_random_seed():
    # reset torch randomness before every test so every run gets
    # the same model weights noise timesteps and dataloader shuffle
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
