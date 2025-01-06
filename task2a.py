
import torch, pdb
from torch.utils.data import DataLoader
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

epochs = 1000
lr = 0.00005  # Adjusted learning rate
bs = 128

device = torch.device('mps')

dataloader = DataLoader(MNIST('.', download=True, transform=transforms.ToTensor()), shuffle=True, batch_size=bs)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 784),
            nn.Sigmoid()  # Output in range [0,1] for MNIST data
        )

    def forward(self, noise):
        return self.gen(noise)

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(784, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),  # No activation function
        )

    def forward(self, image):
        return self.disc(image)

def gen_noise(number):
    return torch.randn(number, 64).to(device)

gen = Generator().to(device)
gen_opt = torch.optim.RMSprop(gen.parameters(), lr=lr)
critic = Critic().to(device)
critic_opt = torch.optim.RMSprop(critic.parameters(), lr=lr)

def visualize_images(images, epoch):
    data = images.detach().cpu().view(-1, 1, 28, 28)
    grid = make_grid(data[:16], nrow=4).permute(1, 2, 0)
    plt.imshow(grid, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.savefig(f'wgan_{epoch}.png')

def calc_gen_loss(gen, critic, number):
    noise = gen_noise(number)
    fake = gen(noise)
    pred = critic(fake)
    gen_loss = -pred.mean()
    return gen_loss

def calc_critic_loss(gen, critic, number, real):
    noise = gen_noise(number)
    fake = gen(noise)
    D_fake = critic(fake.detach())
    D_real = critic(real)
    critic_loss = D_fake.mean() - D_real.mean()
    return critic_loss

# Training the WGAN with weight clipping
n_critic = 5  # Number of critic iterations per generator iteration

for epoch in tqdm(range(epochs)):
    data_iter = iter(dataloader)
    i = 0
    while i < len(dataloader):
        # Train the critic n_critic times
        for t in range(n_critic):
            if i >= len(dataloader):
                break
            real, _ = next(data_iter)
            real = real.view(-1, 784).to(device)
            bs = real.size(0)
            critic_opt.zero_grad()
            critic_loss = calc_critic_loss(gen, critic, bs, real)
            critic_loss.backward()
            critic_opt.step()

            # Weight clipping
            for p in critic.parameters():
                p.data.clamp_(-0.01, 0.01)

            i += 1

        # Train the generator
        gen_opt.zero_grad()
        gen_loss = calc_gen_loss(gen, critic, bs)
        gen_loss.backward()
        gen_opt.step()

    if (epoch + 1) % 5 == 0:
        print(f'Epoch {epoch + 1}/{epochs}')
        noise = gen_noise(16)
        generated_images = gen(noise)
        generated_images = generated_images.reshape(16, 1, 28, 28)
        visualize_images(generated_images, epoch=epoch+1)
