import torch.nn as nn
import  torch
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

def ConvMixer(dim, depth, kernel_size=9, patch_size=7, n_classes=1000):
    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim)
        ) for i in range(depth)],
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes)
    )


# 定义ConvMixer模型参数
input_channels = 3  # 输入图像通道数
depth =  8  # ConvMixer的块数
dim = 3
# 创建ConvMixer模型
model = ConvMixer(dim ,input_channels ,depth)

# 创建输入数据（假设批量大小为1，输入图像尺寸为224x224）
input_data = torch.randn(1, input_channels, 224, 224)

# 进行前向传播
output = model(input_data)

# 输出结果的形状
print(output.shape)
