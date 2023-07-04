import torch
from discriminator import Discriminator
from generator import Generator, initialize_weights


def test():
    N = 8
    in_channels = 3
    H = 64
    W = 64
    z_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    initialize_weights(disc)
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
    gen = Generator(z_dim, in_channels, 8)
    z = torch.randn((N, z_dim, 1, 1))
    initialize_weights(gen)
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"
    print("Success")


test()
