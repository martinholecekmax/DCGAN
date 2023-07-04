import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from discriminator import Discriminator
from generator import Generator, initialize_weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("===============================================================")
print("device: ", device)
print("===============================================================")

# Hyperparameters
LEARNING_RATE = 2e-4  # could also use two lrs, one for gen and one for disc
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 1
Z_DIM = 100
NUM_EPOCHS = 5
FEATURES_DISC = 64
FEATURES_GEN = 64

transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
        ),
    ]
)

dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms, download=True)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)

initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(
    gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999)
)  # betas are recommended in the DCGAN paper
opt_disc = optim.Adam(
    disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999)
)  # betas are recommended in the DCGAN paper

criterion = nn.BCELoss()

fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)

# For tensorboard plotting
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")

step = 0

# Set both networks to train mode
gen.train()
disc.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
        fake = gen(noise)

        ## Train Discriminator: max log(D(real)) + log(1 - D(G(z)))
        disc_real = disc(real).reshape(-1)  # reshape to 1D vector of size BATCH_SIZE
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))

        disc_fake = disc(fake).reshape(-1)  # reshape to 1D vector of size BATCH_SIZE
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        loss_disc = (loss_disc_real + loss_disc_fake) / 2

        disc.zero_grad()
        loss_disc.backward(retain_graph=True)
        opt_disc.step()

        ## Train Generator: min log(1 - D(G(z))) <-> Because of saturation of gradients, maximize log(D(G(z)))
        output = disc(gen(noise)).reshape(-1)  # reshape to 1D vector of size BATCH_SIZE
        loss_gen = criterion(output, torch.ones_like(output))

        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Print losses occasionally and print to tensorboard
        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                writer_real.add_image("Real Images", img_grid_real, global_step=step)
                writer_fake.add_image("Fake Images", img_grid_fake, global_step=step)
            step += 1

    # Save model checkpoints
    torch.save(gen.state_dict(), f"weights/gen_{epoch}.pth")
    torch.save(disc.state_dict(), f"weights/disc_{epoch}.pth")
