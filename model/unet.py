import torch
import torch.nn as nn


class DownConvBlock(nn.Module):
    """
    Convolutional block used in the encoder path of U-Net.
    It consists of two 3x3 convolutions followed by ReLU activations.
    """
    def __init__(self, in_channels, out_channels):
        """
        Initialize the downsampling convolutional block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        self.layers = nn.Sequential(
            # [B, in_channels, H, W] -> [B, out_channels, H, W]
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # [B, out_channels, H, W] -> [B, out_channels, H, W]
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Forward pass of the DownConvBlock.

        Args:
            x (torch.Tensor): Input feature map
                [B, in_channels, H, W]

        Returns:
            torch.Tensor: Output feature map
                [B, out_channels, H, W]
        """
        return self.layers(x)


class UpConvBlockSkip(nn.Module):
    """
    Upsampling block with skip connection.
    It performs transposed convolution, concatenation with the skip feature map,
    and a double convolution refinement.
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        """
        Initialize the upsampling block with skip connection.

        Args:
            in_channels (int): Number of input channels from the lower-resolution feature map.
            skip_channels (int): Number of channels from the skip connection.
            out_channels (int): Number of output channels.
        """
        super().__init__()

        # [B, in_channels, H, W] -> [B, out_channels, 2H, 2W]
        self.up = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2
        )

        # After concatenation:
        # [B, out_channels + skip_channels, 2H, 2W] -> [B, out_channels, 2H, 2W]
        self.conv = DownConvBlock(
            in_channels=out_channels + skip_channels,
            out_channels=out_channels
        )

    def forward(self, x, skip):
        """
        Forward pass of the upsampling block.

        Args:
            x (torch.Tensor): Feature map to be upsampled
                [B, in_channels, H, W]
            skip (torch.Tensor): Skip connection feature map
                [B, skip_channels, 2H, 2W]

        Returns:
            torch.Tensor: Output feature map
                [B, out_channels, 2H, 2W]
        """
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class Unet(nn.Module):
    """
    U-Net architecture for semantic segmentation.
    """
    def __init__(self, in_channels=3, hidden_dim=[8, 16, 32, 64], n_channels=1):
        """
        Initialize the U-Net model.

        Args:
            in_channels (int): Number of input image channels (e.g., 3 for RGB).
            hidden_dim (list[int]): Number of feature channels at each level.
            n_channels (int): Number of output segmentation channels.
        """
        super().__init__()

        # =====================
        # Encoder
        # =====================

        # [B, 3, 128, 128] -> [B, 16, 128, 128]
        self.enc1 = DownConvBlock(in_channels, hidden_dim[1])

        # [B, 16, 64, 64] -> [B, 32, 64, 64]
        self.enc2 = DownConvBlock(hidden_dim[1], hidden_dim[2])

        # [B, 32, 32, 32] -> [B, 64, 32, 32]
        self.enc3 = DownConvBlock(hidden_dim[2], hidden_dim[3])

        # [B, C, H, W] -> [B, C, H/2, W/2]
        self.pool = nn.MaxPool2d(2)

        # =====================
        # Decoder
        # =====================

        # [B, 64, 16, 16] + skip [B, 64, 32, 32]
        # -> [B, 32, 32, 32]
        self.dec3 = UpConvBlockSkip(
            in_channels=hidden_dim[3],
            skip_channels=hidden_dim[3],
            out_channels=hidden_dim[2]
        )

        # [B, 32, 32, 32] + skip [B, 32, 64, 64]
        # -> [B, 16, 64, 64]
        self.dec2 = UpConvBlockSkip(
            in_channels=hidden_dim[2],
            skip_channels=hidden_dim[2],
            out_channels=hidden_dim[1]
        )

        # [B, 16, 64, 64] + skip [B, 16, 128, 128]
        # -> [B, 8, 128, 128]
        self.dec1 = UpConvBlockSkip(
            in_channels=hidden_dim[1],
            skip_channels=hidden_dim[1],
            out_channels=hidden_dim[0]
        )

        # [B, 8, 128, 128] -> [B, n_channels, 128, 128]
        self.conv = nn.Conv2d(hidden_dim[0], n_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the U-Net.

        Args:
            x (torch.Tensor): Input image
                [B, in_channels, 128, 128]

        Returns:
            torch.Tensor: Segmentation mask
                [B, n_channels, 128, 128]
        """
        # Encoder
        s1 = self.enc1(x)      # [B, 16, 128, 128]
        x = self.pool(s1)      # [B, 16, 64, 64]

        s2 = self.enc2(x)      # [B, 32, 64, 64]
        x = self.pool(s2)      # [B, 32, 32, 32]

        s3 = self.enc3(x)      # [B, 64, 32, 32]
        x = self.pool(s3)      # [B, 64, 16, 16]

        # Decoder
        x = self.dec3(x, s3)   # [B, 32, 32, 32]
        x = self.dec2(x, s2)   # [B, 16, 64, 64]
        x = self.dec1(x, s1)   # [B, 8, 128, 128]

        return self.conv(x)    # [B, 1, 128, 128]


if __name__ == "__main__":
    x = torch.randn(1, 3, 128, 128)
    model = Unet()
    with torch.no_grad():
        y = model(x)
        print(y.shape)
