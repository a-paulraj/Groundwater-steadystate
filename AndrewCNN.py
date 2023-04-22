import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        activation,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.padding = int(kernel_size / 2)
        self.conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.padding
        )
       
    def forward(self, x):
        return self.activation(self.conv(x))

   
class BasicConvNet(nn.Module):
   
    def __init__(
        self,
        input_channels,
        hidden_channels,
        output_channels,
        kernel_size,
        depth,
        activation
    ):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.depth = depth
        self.activation = activation
        self.layers = [
            ConvBlock(
                self.input_channels,
                self.hidden_channels,
                self.kernel_size,
                self.activation
            )
        ]
        for i in range(1, self.depth-1):
            self.layers.append(
                ConvBlock(
                    self.hidden_channels,
                    self.hidden_channels,
                    self.kernel_size,
                    self.activation
                )
            )
        self.layers.append(
            ConvBlock(
                self.hidden_channels,
                self.output_channels,
                self.kernel_size,
                nn.Identity()
            )
        )
        self.layers = nn.ModuleList(self.layers)
       
    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x