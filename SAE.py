import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

# 定义稀疏自动编码器类
class SparseAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, sparsity_factor):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)
        self.sparsity_factor = sparsity_factor

    def forward(self, x):
        encoded = torch.sigmoid(self.encoder(x))
        decoded = torch.sigmoid(self.decoder(encoded))
        return encoded, decoded

    def loss_function(self, x, decoded):
        reconstruction_loss = nn.MSELoss()(x, decoded)
        sparsity_loss = torch.mean(torch.abs(encoded - self.sparsity_factor))
        total_loss = reconstruction_loss + sparsity_loss
        return total_loss


# 设置参数
input_size = 784
hidden_size = 128
sparsity_factor = 0.1
learning_rate = 0.01
num_epochs = 10
batch_size = 64

# 加载数据集（假设使用MNIST数据集）
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize((0.1307,), (0.3081,))  # 标准化图像数据
])

train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 创建稀疏自动编码器模型
sae = SparseAutoencoder(input_size, hidden_size, sparsity_factor)

# 定义优化器和学习率调度器
optimizer = optim.Adam(sae.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# 训练模型
for epoch in range(num_epochs):
    total_loss = 0
    for data in train_loader:
        inputs, _ = data
        inputs = inputs.view(inputs.size(0), -1)

        optimizer.zero_grad()
        encoded, decoded = sae(inputs)
        loss = sae.loss_function(inputs, decoded)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

# 保存模型
torch.save(sae.state_dict(), 'sae_model.pt')