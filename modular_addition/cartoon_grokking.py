import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import Subset
import hashlib
import copy

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from generate_movie import (
    plot_embeddings_movie,
    run_movie_cmd,
    plot_embeddings_movie_unchunked,
)
from grokking_with_freezing import fracs, count_parameters

from itertools import combinations
from utils import get_weight_norm
from sympy import primerange

import os

import math
import random

random.seed(42)


def comp_mult(a, b):
    real_part = a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1]
    imag_part = a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0]
    return t.stack([real_part, imag_part], dim=-1)

# To apply it on 2xn tensors
a = t.tensor([[1.0, 3.0], [2.0, 4.0], [5.0, 6.0]])  # 3x2 tensor
b = t.tensor([[1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]])  # 3x2 tensor

result = comp_mult(a, b)
print("Result for 3x2 tensors:", result)

class ComplexMultiply(nn.Module):
    def __init__(self):
        super(ComplexMultiply, self).__init__()

    def forward(self, a, b):
        real_part = a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1]
        imag_part = a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0]
        return t.stack([real_part, imag_part], dim=-1)

class Rigid_MLP(nn.Module):
    def __init__(
        self,
        num_blocks,
        prime,
        past_embedding = None
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.prime = prime
        self.embedding = nn.Parameter(t.randn(self.prime, self.num_blocks, 2)) if past_embedding == None else past_embedding
        self.complexMultiply = ComplexMultiply()
    def forward(self, x1, x2):
        x1 = self.embedding[x1]
        x2 = self.embedding[x2]
        x = self.complexMultiply(x1, x2)
        return x

class MLP(nn.Module):
    def __init__(
        self,
        embed_dim,
        vocab_size,
        hidden_dim,
        freeze_embed=False,
        use_circular_embeddings=False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_blocks = embed_dim // 2
        if use_circular_embeddings:
            self.embedding = nn.Parameter(make_circle_embeddings(embed_dim, vocab_size))
        else:
            self.embedding = nn.Parameter(t.randn(vocab_size, embed_dim))
        if freeze_embed:
            self.embedding.requires_grad = False
        # self.embedding = nn.Parameter(t.randn(vocab_size, embed_dim))
        self.linear1_weights = nn.Parameter(t.randn(self.n_blocks, 2, hidden_dim))
        self.linear2_weights = nn.Parameter(t.randn(self.n_blocks, hidden_dim, 2))
        # self.unembed = nn.Linear(embed_dim, vocab_size, bias=False)
        self.silu = nn.SiLU()

    def forward(self, x1, x2):
        with t.no_grad():
            self.unembed_data = self.embedding.data.t().conj()
        x1 = self.embedding[x1].view(-1, self.n_blocks, 2)
        x2 = self.embedding[x2].view(-1, self.n_blocks, 2)
        # Apply different linear transformations to each block
        x1 = t.einsum("bnj,njk->bnk", x1, self.linear1_weights)
        x2 = t.einsum("bnj,njk->bnk", x2, self.linear1_weights)
        x = x1 + x2
        x = self.silu(x)
        x = t.einsum("bnk,nkj->bnj", x, self.linear2_weights)
        # Flatten the tensor from shape (batch_size, n_blocks, 2) to (batch_size, n_blocks*2) = (batch_size, embed_dim)
        x = x.reshape(x.size(0), -1)
        return t.mm(x, self.unembed_data)


class MLP_rigid(nn.Module):
    def __init__(
        self,
        embed_dim,
        vocab_size,
        freeze_embed=False,
        use_embedding = False,
        embedding = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_blocks = embed_dim // 2
        if use_embedding:
            self.embedding = embedding 
        else:
            self.embedding = nn.Parameter(t.randn(vocab_size, embed_dim))
        # self.embedding = nn.Parameter(t.randn(vocab_size, embed_dim))
    def forward(self, x1, x2):
        with t.no_grad():
            self.unembed_data = self.embedding.data.t().conj()
        x1 = self.embedding[x1].view(-1, self.n_blocks, 2)
        x2 = self.embedding[x2].view(-1, self.n_blocks, 2)
        x = t.einsum("bnk,nkj->bnj", x, self.linear2_weights)
        # Flatten the tensor from shape (batch_size, n_blocks, 2) to (batch_size, n_blocks*2) = (batch_size, embed_dim)
        x = x.reshape(x.size(0), -1)
        return t.mm(x, self.unembed_data)


class L2Loss(nn.Module):
    def __init__(self, embedding_layer, prime):
        super(CustomLoss, self).__init__()
        self.embedding_layer = embedding_layer
        self.prime = prime

    def forward(self, y_pred, y_true):
        y_true_embed = self.embedding_layer[y_true]
        y_pred_embed = self.embedding_layer[y_pred]
        return torch.norm(y_pred_embed - y_true_embed, p=2)

class EmbEntropyLoss(nn.Module):
    def __init__(self, embedding_layer, prime):
        super(CustomLoss, self).__init__()
        self.embedding_layer = embedding_layer
        self.prime = prime

    def forward(self, y_pred, y_true):
        # Implementing the cross-entropy: -log(softmax(y_pred)[y_true])
        # Numerical stability: add a small constant to log computation
        log_y_pred = torch.log(y_pred.exp() / torch.sum(y_pred.exp(), dim=-1, keepdim=True) + 1e-9)
        neg_log_likelihood = -log_y_pred[range(len(y_true)), y_true]
        return torch.mean(neg_log_likelihood)

        y_true_embed = self.embedding_layer[y_true]
        y_mod_prime = (y_pred + y_true) % self.prime
        y_mod_prime_embed = self.embedding_layer[y_mod_prime]
        return torch.norm(y_mod_prime_embed - y_true_embed, p=2)





def plot_embeddings(model, num_inds, blue_inds = [0,1,2,3,4,5], suffix = ""):
    plt.clf()
    embeddings = model.embedding.detach().cpu() # vocab_size x embed_dim
    chunked = t.chunk(embeddings, embeddings.shape[-1]//2, dim = -1)
    n = embeddings.shape[-1]//2
    # calculate number of rows and columns for subplots
    rows = int(n ** 0.5)
    cols = n // rows
    if rows * cols < n:  # if not enough subplots, add an extra column
        cols += 1
    # visualise each vocab_size x 2 chunk in a subplot
    _, axs = plt.subplots(rows, cols, figsize=(30, 15))
    axs = axs.flatten()  # flatten the array of axes to simplify indexing
    for i, chunk in enumerate(chunked):
        axs[i].scatter(chunk[:, 0], chunk[:, 1])
        for j, word in enumerate(words):
            text_color = blue if j in blue_inds else black
            axs[i].annotate(word, xy=(chunk[j, 0], chunk[j, 1]), color=text_color)
        words = [str(i) for i in range(embeddings.shape[0])]
    plt.tight_layout()  # adjust spacing between subplots
    # make /frames if it does not exist
    if not os.path.exists("model_plots"):
        os.mkdir("model_plots")
    plt.savefig(f"model_plots/embeddings_movie_{suffix}.png")  # change 'i' to your step variable
    plt.close()

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
    lr=0.01,
):
    vocab_size = model.vocab_size
    num_epochs = num_lin_epochs//vocab_size
    print(f"Number of parameters: {count_parameters(model)}")
    # optimizer = t.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    optimizer = t.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
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
            model.linear1_weights.requires_grad = False
            model.linear2_weights.requires_grad = False
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
            # if save_frames:
            #     if frame_n % 500 == 0: #for big network might use 10
            #         with t.no_grad():
            #             step += 1
            #             if use_unchunked:
            #                 plot_embeddings_movie_unchunked(model, step)
            #             else:
            #                 plot_embeddings_movie(model, step)
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
        plot_embeddings(model, suffix)
    return model

def test_model(model, test_loader, device, criterion):
    val_loss = 0
    val_acc = 0
    with t.no_grad():
        for x1, x2, target in test_loader:
            x1, x2, target = x1.to(device), x2.to(device), target.to(device)
            output = model(x1, x2)
            loss = criterion(output, target)
            val_loss += loss.item()
            val_acc += (output.argmax(dim=-1) == target).float().mean()
        val_acc = val_acc / len(test_loader)
        val_loss = val_loss / len(test_loader)
    return val_loss, val_acc

class ModuloAdditionDataset(data.Dataset):
    def __init__(self, shuf, d_vocab=114):
        super().__init__()
        self.d_vocab = d_vocab
        self.eq_token = d_vocab - 1  # assuming '=' is the last token in the vocabulary
        self.shuf = shuf

    def __len__(self):
        return (self.d_vocab - 1) ** 2

    def __getitem__(self, idx):
        a = self.shuf[idx // (self.d_vocab - 1)] 
        b = idx % (self.d_vocab - 1)
        res = (a + b) % (self.d_vocab - 1)
        return a, b, res

def get_train_test_loaders(train_frac, batch_size, vocab_size, randomize=False, seed=42, sequential = False):
    generator = t.Generator().manual_seed(seed)  # Seed for reproducibility
    if randomize:
        dataset = RandomOperationDataset(vocab_size)
    else:
        dataset = ModuloAdditionDataset(vocab_size)
    total_length = len(dataset)
    train_length = int(total_length * train_frac)
    test_length = total_length - train_length

    if sequential:
        train_indices = range(0, train_length)
        test_indices = range(test_length, total_length)
        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)
        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    else:
        train_dataset, test_dataset = data.random_split(
            dataset, [train_length, test_length], generator=generator
        )
        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
        test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, generator=generator)
    return train_loader, test_loader



def frac_test(
    vocab_size, 
    Adam = True, 
    sequential = False, 
    graph = False, 
    embed_dim = 6,
    hidden_dim = 24, 
    quadratic = False, 
    lr = 0.002, 
    reg_orig = 0.0005, 
    reg = 0 , 
    num_datapoints = 12,
    num_lin_epochs = 50000,
):
    BATCH_SIZE = 128
    SEED = 42
    train_loader, test_loader = get_train_test_loaders(vocab_size = vocab_size, batch_size = BATCH_SIZE, train_frac = 0.5)
    original_model =  MLP(embed_dim=embed_dim, vocab_size=vocab_size, hidden_dim=hidden_dim)
    if quadratic:
        original_model.silu = nn.Lambda(lambda x: x * x)
    original_model = train(original_model, 
        train_loader,
        test_loader,
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        num_lin_epochs = num_lin_epochs,
        reg=reg_orig,
        lr = lr,
        save_last_frame = True,
        suffix = f"vocab_{vocab_size}_e_{embed_dim}_h_{hidden_dim}_main",
    )

    for frac in fracs(vocab_size, num_datapoints = num_datapoints):
        print(f"frac {frac}")
        train_loader, test_loader = get_train_test_loaders(vocab_size = vocab_size, train_frac = frac, batch_size = BATCH_SIZE, randomize=False, seed=SEED)
        model = copy.deepcopy(original_model)
        model.embedding = nn.Parameter(t.randn(vocab_size, embed_dim))
        optimizer = t.optim.Adam(model.parameters(), lr=lr)
        train(model, 
            train_loader,
            test_loader,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            num_lin_epochs = num_lin_epochs,
            reg=reg,
            lr = lr,
            save_frames = False,
            save_last_frame = True,
            freeze_mlp = True,
            suffix = f"vocab_{vocab_size}_e_{embed_dim}_h_{hidden_dim}_frac_{frac}",
        )
        
VOCAB_SIZE = 38
SHUF = list(range(VOCAB_SIZE - 1))
# random.shuffle(SHUF)


# if __name__ == "__main__":
#     train_loader, test_loader = get_train_test_loaders(0.5, 4, VOCAB_SIZE, sequential=True)
#     for thing in enumerate(train_loader):
#         print(thing)
#         # if i >= 2:  # stop after printing 3 batches
#         #     break

    # frac_test(114)



