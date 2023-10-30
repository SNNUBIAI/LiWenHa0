import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

        self.decoder = nn.Sequential(
            nn.Linear(10, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # [b,1,28,28] ———> [b,784]
        x = self.encoder(x)  # [b,784] ———> [b,10]
        x = self.decoder(x)  # [b,10] ———> [b,784]
        x = x.view(-1, 1, 28, 28)  # 重新将重建图像 reshape 为 [b,1,28,28]
        return x

# 加载数据集
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize((0.1307,), (0.3081,))  # 标准化图像数据
])

train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = MNIST(root='./data', train=False, transform=transform, download=True)

batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 训练准备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AE().to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
best_loss = float('inf')  # 保存最佳损失值

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for x, _ in train_loader:
        x = x.to(device)
        recon = model(x)
        loss = loss_fn(recon, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * x.size(0)

    train_loss /= len(train_loader.dataset)

    # 在每个训练周期结束后，计算验证集上的损失并保存最佳模型
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            recon = model(x)
            loss = loss_fn(recon, x)
            test_loss += loss.item() * x.size(0)

    test_loss /= len(test_loader.dataset)

    # 输出训练过程信息
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    # 如果当前模型的验证集损失更低，则保存该模型
    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(model.state_dict(), 'best_model.pt')

# 加载最佳模型进行重建
best_model = AE().to(device)
best_model.load_state_dict(torch.load('best_model.pt'))

# 可视化重建结果
import matplotlib.pyplot as plt

def plot_reconstructions(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        x, _ = next(iter(test_loader))
        x = x.to(device)
        recon = model(x)

    x = x.cpu().numpy()
    recon = recon.cpu().numpy()

    fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(10, 2))
    for i in range(10):
        axes[0, i].imshow(x[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(recon[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')

    plt.show()

plot_reconstructions(best_model, test_loader, device)
