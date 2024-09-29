import torch.nn as nn
from mamba_ssm import Mamba

__all__ = ('detrmamba')


class detrmamba(nn.Module):
    def __init__(self, c1=3, w=320, h=320):
        super().__init__()
        self.convb = nn.Sequential(
            nn.Conv2d(in_channels=c1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=c1, kernel_size=3, stride=1, padding=1)
        )
        self.model1 = Mamba(
            d_model=c1,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        self.model2 = Mamba(
            d_model=c1,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        self.model3 = Mamba(
            d_model=w * h,
            d_state=16,
            d_conv=4,
            expand=2,
        )
        self.smooth = nn.Conv2d(in_channels=c1, out_channels=c1, kernel_size=3, stride=1, padding=1)
        self.ln = nn.LayerNorm(normalized_shape=c1)
        self.softmax = nn.Softmax()

    def forward(self, x):
        b, c1, w, h = x.shape
        x = self.convb(x) + x
        x = self.ln(x.reshape(b, -1, c1))
        y = self.model1(x).permute(0, 2, 1)
        z = self.model3(y).permute(0, 2, 1)
        att = self.softmax(self.model2(x))
        result = att * z
        output = result.reshape(b, c1, w, h)
        return self.smooth(output)
