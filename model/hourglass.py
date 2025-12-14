import torch.nn as nn
import torch

class DownConvBlock(nn.Module):
    """
        Downsampling convolutional block used in the encoder.

        This block applies two consecutive 2D convolutions with a ReLU activation
        in between. Spatial resolution is preserved (padding=1), while the number
        of feature channels is increased.
    """
    def __init__(self, in_channels, out_channels):
        """
            Initialize the DownConvBlock.

            Args:
                in_channels (int): Number of input feature channels.
                out_channels (int): Number of output feature channels.
        """
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels= in_channels, out_channels= out_channels, kernel_size= 3, padding= 1),
            nn.ReLU(inplace= True),
            nn.Conv2d(in_channels= out_channels, out_channels= out_channels, kernel_size= 3, padding= 1),
        )

    def forward(self, x):
        """
            Forward pass of the downsampling block.

            Args:
                x (torch.Tensor): Input tensor of shape [B, C, H, W].

            Returns:
                torch.Tensor: Output tensor after convolutional layers.
        """
        return self.layers(x) 
    
class UpConvBlock(nn.Module):
    """
        Upsampling convolutional block used in the decoder.

        This block upsamples the feature map using a transposed convolution,
        followed by a ReLU activation and a standard convolution.
    """
    def __init__(self, in_channels, out_channels):
        """
            Initialize the UpConvBlock.

            Args:
                in_channels (int): Number of input feature channels.
                out_channels (int): Number of output feature channels.
        """
        super().__init__()

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels= in_channels, out_channels= out_channels, kernel_size= 2, padding= 0, stride= 2),
            nn.ReLU(inplace= True),
            nn.Conv2d(in_channels= out_channels, out_channels= out_channels, kernel_size= 3, padding= 1),
        )
    
    def forward(self, x):
        """
            Forward pass of the upsampling block.

            Args:
                x (torch.Tensor): Input tensor of shape [B, C, H, W].

            Returns:
                torch.Tensor: Upsampled output tensor.
        """
        return self.layers(x)
    
    
class HourglassNet(nn.Module):
    """
        Simple Hourglass-style convolutional neural network for image segmentation.

        The architecture consists of:
        - An encoder with downsampling blocks and max pooling
        - A latent (bottleneck) stage
        - A decoder with upsampling blocks
        - A final 1x1 convolution to produce segmentation logits    
    """
    def __init__(self, in_channels= 3, hidden_dim = [8, 16, 32, 64], n_channels = 1):
        """
            Initialize the HourglassNet.

            Args:
                in_channels (int, optional): Number of input image channels. Defaults to 3.
                hidden_dim (list[int], optional): Number of feature channels at each level
                    of the network. Defaults to [8, 16, 32, 64].
                n_channels (int, optional): Number of output channels (e.g., number of
                    segmentation classes). Defaults to 1.
        """
        super().__init__()

        self.encoder = nn.Sequential(
            DownConvBlock(in_channels= in_channels, out_channels= hidden_dim[1]), # [B, 3, 128, 128] -> [B, 16, 128, 128]
            nn.MaxPool2d(kernel_size= 2, stride= 2), # [B, 16, 128, 128] -> [B, 16, 64, 64]
            DownConvBlock(in_channels= hidden_dim[1], out_channels= hidden_dim[2]), # [B, 16, 64, 64] -> [B, 32, 64, 64]
            nn.MaxPool2d(kernel_size= 2, stride= 2), # [B, 32, 64, 64] -> [B, 32, 32, 32]
            DownConvBlock(in_channels= hidden_dim[2], out_channels= hidden_dim[3]), # [B, 32, 32, 32] -> [B, 64, 32, 32]
            nn.MaxPool2d(kernel_size= 2, stride= 2) # [B, 64, 32, 32] -> [B, 64, 16, 16]
        )

        self.latent = nn.Identity()

        self.decoder = nn.Sequential(
            UpConvBlock(in_channels= hidden_dim[3], out_channels= hidden_dim[2]), # [B, 64, 16, 16] -> [B, 32, 32, 32]
            UpConvBlock(in_channels= hidden_dim[2], out_channels= hidden_dim[1]), # [B, 32, 32, 32] -> [B, 16, 64, 64]
            UpConvBlock(in_channels= hidden_dim[1], out_channels= hidden_dim[0]) # # [B, 16, 64, 64] -> [B, 8, 128, 128]
        )

        self.conv = nn.Conv2d(in_channels= hidden_dim[0], out_channels= n_channels, kernel_size= 1) # [B, 8, 128, 128] -> [B, 2, 128, 128]

    def forward(self, x):
        """
            Forward pass of the Hourglass network.

            Args:
                x (torch.Tensor): Input image tensor of shape [B, C, H, W].

            Returns:
                torch.Tensor: Output segmentation logits of shape [B, n_channels, H, W].
        """
        x = self.encoder(x)
        x = self.latent(x)
        x = self.decoder(x)
        return self.conv(x) # [B, 2, 128, 128]
    

if __name__ == "__main__":
    x = torch.randn(1, 3, 128, 128)

    model = HourglassNet()
    with torch.no_grad():
        a = model(x)
        print(a.shape)