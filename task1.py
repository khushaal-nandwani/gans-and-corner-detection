import torch, pdb
from torch.utils.data import DataLoader
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

epochs = 1000
# 0.0002, 0.0001, 0.000001, 
# 0.001
LEARNING_RATE = 0.00001
GEN_LEARNING_RATE = 0.00002

loss_func = nn.BCEWithLogitsLoss()

BATCH_SIZE = 64
GEN_BATCH_SIZE = 128

print("Starting with Learning Rate, Epochs, Batch_Size: ", LEARNING_RATE, epochs, BATCH_SIZE)

device = torch.device("mps")

print("Apple device available: ", device)

dataloader = DataLoader(MNIST('.', download=True, transform=transforms.ToTensor()),shuffle=True, batch_size=BATCH_SIZE)

class Generator2(nn.Module):
    def __init__(self):
        super().__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(64, 512, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
                        
    def forward(self, noise):
        return self.gen(noise)


class Discriminator2(nn.Module):
    def __init__(self):
        super(Discriminator2, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True), # 1 x 1
        )
        # take 64 dim to 1
        self.fc = nn.Linear(64, 1)
        
    def forward(self, noise):
        output = self.disc(noise)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output
    

# create a new text file and print discriminator and generator models
with open(f"discriminator_{LEARNING_RATE}_{epochs}.txt", "w") as f:
    f.write(str(Discriminator2()))
    
with open(f"generator_{LEARNING_RATE}_{epochs}.txt", "w") as f:
    f.write(str(Generator2()))


def gen_noise(number):
    # 1 dimension vector -> image
    # TODO: try different noise
    return torch.randn(number, 64, 1, 1).to(device)
    
    
# create generator
# TODO: try different learning rates for gen and disc
gen = Generator2().to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=LEARNING_RATE)
# TODO: try different optimizers
disc = Discriminator2().to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=LEARNING_RATE)


def visualize_images(images, save=False, name_end=""):
  data = images.detach().cpu().view(-1,1,*(28,28)) 
  grid = make_grid(data[:16], nrow=4).permute(1,2,0) 
  plt.imshow(grid)
  plt.xticks([]), plt.yticks([])
  if save:
      plt.savefig(f"image_epoch_{name_end}_lr_{LEARNING_RATE}_bs_{BATCH_SIZE}_gbs_{GEN_BATCH_SIZE}.png")
  else:
      plt.show()

noise = gen_noise(16)
generated_images = gen(noise)
print(generated_images.shape)
print("Generated Images Shape: ", generated_images.shape)
# generated_images = generated_images.reshape(16, 1, 28, 28)
visualize_images(generated_images, save=True, name_end="-0")

def calc_gen_loss(loss_func, gen, disc, number):
   noise = gen_noise(number)
   fake = gen(noise)
   pred = disc(fake)
   targets=torch.ones_like(pred)
   gen_loss=loss_func(pred,targets) # <-- attention here (- sign)

   return gen_loss


HARD = False
def calc_disc_loss(loss_func, gen, disc, number, real):
   noise = gen_noise(number)
   fake = gen(noise)
   disc_fake = disc(fake.detach()) # <-- attention here
   if HARD:
       # give 0.1 for fake
        disc_fake_targets=torch.ones_like(disc_fake)*0.1
   else:
       disc_fake_targets=torch.zeros_like(disc_fake)
   disc_fake_loss=loss_func(disc_fake, disc_fake_targets) # <-- attention here (+ sign)

   disc_real = disc(real)
   if HARD:
        # give 0.9 for real
        disc_real_targets=torch.ones_like(disc_real)*0.9  
   else:
       disc_real_targets=torch.ones_like(disc_real)
   disc_real_loss=loss_func(disc_real, disc_real_targets) # <-- attention here

   disc_loss=(disc_fake_loss+disc_real_loss)/2 # <-- attention here

   return disc_loss


# Training
for epoch in tqdm(range(epochs)):
            
    
    for real, _ in dataloader:
        real = real.to(device)
        disc_opt.zero_grad()
        disc_loss = calc_disc_loss(loss_func, gen, disc, BATCH_SIZE, real)
        disc_loss.backward(retain_graph=True)
        disc_opt.step()

        gen_opt.zero_grad()
        gen_loss = calc_gen_loss(loss_func, gen, disc, GEN_BATCH_SIZE)
        gen_loss.backward()
        gen_opt.step()
        
    if epoch % 3 == 0:      # at this point we are seeing, how well our generator is creating the images.
        
        # print gen and disc loss
        print("Gen Loss: ", gen_loss.item(), "  Disc Loss: ", disc_loss.item())
        
        noise = gen_noise(16) 
        generated_images = gen(noise)
        generated_images = generated_images.reshape(16, 1, 28, 28)
        visualize_images(generated_images, save=True, name_end=epoch)
        

print("Training Done")


# visualize the images
noise = gen_noise(16, device)
generated_images = gen(noise)
visualize_images(generated_images)
