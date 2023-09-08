
import torch as t
import torch.nn as nn
import torch.utils.data as data
import hashlib

import matplotlib.pyplot as plt
from generate_movie import (
    plot_embeddings_movie,
    run_movie_cmd,
    plot_embeddings_movie_unchunked,
)
from itertools import combinations
from utils import get_weight_norm
from sympy import primerange

import math

import copy


from mlp_modular import ModuloAdditionDataset, test_model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MLP_unchunked(nn.Module):
    def __init__(self, embed_dim, vocab_size, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.linear1 = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.linear2 = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.silu = nn.SiLU()
        self.vocab_size = vocab_size

    def forward(self, x1, x2):
        x1 = self.embedding(x1)
        x2 = self.embedding(x2)
        x1 = self.linear1(x1)
        x2 = self.linear1(x2)
        x = x1 + x2
        x = self.silu(x)
        x = self.linear2(x)
        return x

    def get_fourier_modes(self):
        embedding_weights = self.embedding.weight.detach().clone().cpu().to(t.cfloat)
        fourier_matrix = t.zeros((self.vocab_size, self.vocab_size), dtype=t.cfloat)
        for i in range(self.vocab_size):
            for j in range(self.vocab_size):
                theta = t.tensor(2 * t.pi * i * j / (self.vocab_size - 1))
                fourier_matrix[i, j] = t.complex(t.cos(theta), t.sin(theta))
        fourier_matrix = fourier_matrix.to(embedding_weights.device)
        fourier_embedding = t.matmul(fourier_matrix, embedding_weights)
        return fourier_embedding


    def project_to_fourier_mode(self, filename):
        fourier_modes = self.get_fourier_modes()
        n = fourier_modes.shape[0]
        cos = fourier_modes.real
        sin = fourier_modes.imag
        plt.clf()

        # Determine the layout for a roughly square configuration
        n_plots = (n + 1) // 2
        num_cols = int(math.ceil(math.sqrt(n_plots)))
        num_rows = int(math.ceil(n_plots / num_cols))
        plt.figure(figsize=(num_cols * 2, num_rows * 2))

        # Get all embeddings and drop the last one
        embeddings = self.embedding.weight.detach().cpu()[:-1]

        # Compute dot products for all modes and embeddings at once
        dot_products_cos = t.matmul(cos, embeddings.t())
        dot_products_sin = t.matmul(sin, embeddings.t())

        for i in range(n_plots):
            plt.subplot(num_rows, num_cols, i + 1)
            plt.scatter(dot_products_cos[i].numpy(), dot_products_sin[i].numpy())
            plt.title(f"Mode {i}")

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def plot_fourier_modes(self, filename):
        fourier_embedding = self.get_fourier_modes()
        plt.clf()
        # figsize should fit both real and imaginary parts subplots
        plt.figure(figsize=(fourier_embedding.shape[1] * 2, fourier_embedding.shape[0]))
        # plot real and imaginary parts separately
        plt.subplot(1, 2, 1)
        plt.imshow(fourier_embedding.real.cpu().numpy(), cmap="gray")
        plt.subplot(1, 2, 2)
        plt.imshow(fourier_embedding.imag.cpu().numpy(), cmap="gray")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()


def fracs(vocab_size, num_datapoints = 8): 
    return [vocab_size**(-2/num_datapoints) for n in range(1, num_datapoints)]
   
# model.load_state_dict(t.load("modular_addition.ckpt"))

def get_train_test_loaders(train_frac, batch_size, vocab_size, randomize=False, seed=42):
    if randomize:
        dataset = RandomOperationDataset(vocab_size)
    else:
        dataset = ModuloAdditionDataset(vocab_size)
    total_length = len(dataset)
    train_length = int(total_length * train_frac)
    test_length = total_length - train_length
    generator = t.Generator().manual_seed(seed)  # Seed for reproducibility

    train_dataset, test_dataset = data.random_split(
        dataset, [train_length, test_length], generator=generator
    )
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, generator=generator)
    return train_loader, test_loader

def train(model, 
    train_loader,
    test_loader,
    hidden_dim=32,
    embed_dim=16,
    num_lin_epochs = 20000,
    reg=0,
    save_frames = False,
    save_last_frame = True,
    suffix = "",
    freeze_mlp = False,
):
    vocab_size = model.vocab_size
    num_epochs = num_lin_epochs//vocab_size
    print(f"Number of parameters: {count_parameters(model)}")
    # optimizer = t.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    optimizer = t.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)
    scheduler = t.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)
    criterion = nn.CrossEntropyLoss()
    epochs = num_epochs
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    model.to(device)
    step = 0
    frame_n = 0
    for epoch in range(epochs):
        model.train()
        if freeze_mlp:
            model.linear1.weight.requires_grad = False
            model.linear2.weight.requires_grad = False
        train_loss = 0
        for x1, x2, target in train_loader:
            x1, x2, target = x1.to(device), x2.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(x1, x2)
            loss = criterion(output, target) + reg * get_weight_norm(model)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            frame_n += 1
            if save_frames:
                if frame_n % 500 == 0: #for big network might use 10
                    with t.no_grad():
                        step += 1
                        if use_unchunked:
                            plot_embeddings_movie_unchunked(model, step)
                        else:
                            plot_embeddings_movie(model, step)
        model.eval()
        if epoch % 50 == 0:
            val_loss, val_acc = test_model(model, test_loader, device, criterion)
            train_loss, train_acc = test_model(model, train_loader, device, criterion)
            print(
                f"Epoch {epoch}: train loss: {float(train_loss)}, train accuracy: {float(train_acc)}, val loss: {float(val_loss)}, val accuracy: {float(val_acc)}"
            )
        scheduler.step()
    # t.save(model.state_dict(), "modular_addition.ckpt")
    if save_last_frame == True:
        model.project_to_fourier_mode(f"fourier_modes_{suffix}.png")
    return model


def frac_test(vocab_size, Adam = True, sequential = False, graph = False, embed_dim = 6,hidden_dim = 24, quadratic = False, lr = 0.01, reg = 0):
    BATCH_SIZE = 128
    SEED = 42
    train_loader, test_loader = get_train_test_loaders(vocab_size = vocab_size, batch_size = BATCH_SIZE, train_frac = 0.5)
    original_model =  MLP_unchunked(embed_dim=embed_dim, vocab_size=vocab_size, hidden_dim=hidden_dim)
    if quadratic:
        original_model.silu = nn.Lambda(lambda x: x * x)
    original_model = train(original_model, 
        train_loader,
        test_loader,
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        num_lin_epochs = 50000,
        reg=0.00,
        save_last_frame = True,
        suffix = f"vocab_{vocab_size}_e_{embed_dim}_h_{hidden_dim}_main",
    )

    for frac in fracs(vocab_size, num_datapoints = 8):
        train_loader, test_loader = get_train_test_loaders(vocab_size = vocab_size, train_frac = frac, batch_size = BATCH_SIZE, randomize=False, seed=SEED)
        model = copy.deepcopy(original_model)
        model.embedding = nn.Embedding(vocab_size, embed_dim)
        optimizer = t.optim.Adam(model.parameters(), lr=lr)
        train(model, 
            train_loader,
            test_loader,
            hidden_dim=32,
            embed_dim=16,
            num_lin_epochs = 50000,
            reg=reg,
            save_frames = False,
            save_last_frame = True,
            freeze_mlp = True,
            suffix = f"vocab_{vocab_size}_e_{embed_dim}_h_{hidden_dim}_frac_{frac}",
        )
        
if __name__ == "__main__":
    frac_test(114)

