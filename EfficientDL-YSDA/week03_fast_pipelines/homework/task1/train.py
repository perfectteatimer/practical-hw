import torch
from torch import nn
from tqdm.auto import tqdm

from unet import Unet

from dataset import get_train_data


def train_epoch(
    train_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_scale: float,  # param scale
    successful_steps: int,  # counter for tracking nans
    growth_interval: int,  # the step limit after which we can increase the loss scale factor
    growth_factor: float,  # the scale factor for the loss scale
    dynamic: bool,  # boolean to determine static vs dynamic scaling
) -> tuple[float, int]:
    model.train()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)  # to avoid accum of gradients
        with torch.amp.autocast(device.type, dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, labels)
        # TODO: your code for loss scaling here
        scaled_loss = (
            loss * loss_scale
        )  # increase loss in loss scale times to make safe backward
        scaled_loss.backward()

        # when we are done with backward it is time to update params
        # before that we need to divide our gradients to avoid uneeded consequences of big gradients in our optimizer
        found_nonfinite = (
            False  # for dynamic scale: if grad +-inf or nan then we should change scale
        )
        for parametr in model.parameters():
            if (
                parametr.grad is not None
            ):  # in some cases not all gradients are used in backward (detach, several heads, etc)
                parametr.grad.div_(loss_scale)
                if (
                    not torch.isfinite(parametr.grad).all().item()
                ):  # if grad +- inf then do not optimize
                    found_nonfinite = True
        if not dynamic:
            optimizer.step()
        elif not found_nonfinite:  # if grads are okay then we go further
            optimizer.step()
            successful_steps += 1
            if successful_steps >= growth_interval:
                loss_scale *= growth_factor
                successful_steps = 0
        else:  # grads are inf or nan then we decrease loss scale factor and reset to zero counter
            loss_scale *= 0.5
            successful_steps = 0

        accuracy = ((outputs > 0) == labels).float().mean()

        pbar.set_description(
            f"Loss: {round(loss.item(), 4)} "
            f"Accuracy: {round(accuracy.item() * 100, 4)}"
        )
    return loss_scale, successful_steps


def train():
    device = torch.device("cuda:0")
    model = Unet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_loader = get_train_data()
    loss_scale = 2**16  # initial loss scale
    successful_steps = 0  # counter for grads

    growth_interval = 100  # if grads are okay for 100 steps then we can incline them
    growth_factor = 2.0  # incline factor for that matter
    num_epochs = 5
    dynamic = True  # change for false if needed
    for epoch in range(0, num_epochs):
        loss_scale, successful_steps = train_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            device=device,
            loss_scale=loss_scale,
            successful_steps=successful_steps,
            growth_interval=growth_interval,
            growth_factor=growth_factor,
            dynamic=dynamic,
        )
