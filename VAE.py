import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image

#随机初始化
torch.manual_seed(42)
#准备数据
batch_size = 128
transforms = transforms.Compose([
    transforms.ToTensor(),transforms.Normalize((0.1307),(0.3081))
])
train_dataset = MNIST(root='./data', train=True, transform=transforms, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#定义VAE
class VAE(nn.Module):
    def __int__(self, latent_dim):
        super(VAE, self).__int__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(718,512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid(),
        )

    def encode(self, x):
         x = self.encoder(x)
         mu = self.fc_mu(x)
         logvar = self.fc_logvar(x)
         return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mu + epsilon * std
        return z

    def decode(self, z):
        x = self.decoder(z)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
#定义Loss函数
def vae_loss(x, x_recon, mu, logvar):
    reconstruction_loss = F.binary_cross_entropy(x_recon, x.view(-1, 784), reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + kl_divergence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#参数
latent_dim = 20
learning_rate = 1e-3
num_epochs = 20
model = VAE(latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#训练
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)

        recon_batch, mu, logvar = model(data)

        loss = vae_loss(data, recon_batch, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
with torch.no_grad():
    z = torch.randn(64, latent_dim).to(device)
    samples = model.decode(z).cpu()
save_image(samples.view(64, 1, 28, 28), 'vae_samples.png')