import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, residual: bool = False):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
        )

        self.is_res = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.main(x)
        if self.is_res:
            x = x + self.conv(x)
            return x / 1.414
        else:
            return self.conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            ConvBlock(in_channels, out_channels), nn.MaxPool2d(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ConvBlock(out_channels, out_channels),
            ConvBlock(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x, skip), 1)
        x = self.layers(x)

        return x


class TimestepEmbedding(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()

        self.lin1 = nn.Linear(1, emb_dim, bias=False)
        self.lin2 = nn.Linear(emb_dim, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 1)
        x = torch.sin(self.lin1(x))
        x = self.lin2(x)
        return x


class UnetModel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_size: int = 256):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.hidden_size = hidden_size

        self.init_conv = ConvBlock(in_channels, hidden_size, residual=True)

        self.down1 = DownBlock(hidden_size, hidden_size)
        self.down2 = DownBlock(hidden_size, 2 * hidden_size)
        self.down3 = DownBlock(2 * hidden_size, 2 * hidden_size)

        self.to_vec = nn.Sequential(nn.AvgPool2d(4), nn.ReLU())

        self.timestep_embedding = TimestepEmbedding(2 * hidden_size)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * hidden_size, 2 * hidden_size, 4, 4),
            nn.GroupNorm(8, 2 * hidden_size),
            nn.ReLU(),
        )

        self.up1 = UpBlock(4 * hidden_size, 2 * hidden_size)
        self.up2 = UpBlock(4 * hidden_size, hidden_size)
        self.up3 = UpBlock(2 * hidden_size, hidden_size)
        self.out = nn.Conv2d(2 * hidden_size, self.out_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)

        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)

        thro = self.to_vec(down3)
        temb = self.timestep_embedding(t)

        # thro stores the lowest dimensional representation of an image
        # it is a 4D tensor [batch size, number of channels, 1, 1] but basically
        # it is a vector of channel features for each image in the batch.
        # temb is a timestep embedding with shape [batch size, number of channels].
        # it tells the model which diffusion step the image is currently at or
        # == how noisy the input image is
        # both thro and temb contain features, and we wanna add the timestep info
        # to the image features so the model knows how to process the current noise level
        # to add them correctly we need matching dimensions so we add two singleton
        # spatial dimensions to temb. they do not add new values they only change
        # the tensor shape from [B, C] to [B, C, 1, 1]

        temb = temb[
            :, :, None, None
        ]  # [keep the same, keep the same, add new empty dim, new add empty dim]

        thro = self.up0(thro + temb)

        up1 = self.up1(thro, down3) + temb
        up2 = self.up2(up1, down2)
        up3 = self.up3(up2, down1)

        out = self.out(torch.cat((up3, x), 1))

        return out
