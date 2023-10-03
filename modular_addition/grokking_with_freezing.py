
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
from sympy import primerange, isprime

import math

import copy


from mlp_modular import test_model

from datetime import datetime

def run_movie_cmd(suffix = ""):
    if not os.path.exists("movies"):
        os.mkdir("movies")
    mp4_name = os.path.join('movies', f'movie_{datetime.now().strftime("%Y%m%d_%H%M%S")}{suffix}.mp4')
    os.system(f'ffmpeg -framerate 3 -i frames/embeddings_movie_%06d.png -c:v libx264 -pix_fmt yuv420p {mp4_name}')
    os.system('rm frames/embeddings_movie_*.png')
    print(f'movie saved as {mp4_name}')

def plot_embeddings_movie(model, step):
    plt.clf()
    embeddings = model.embedding.weight.detach().cpu() # vocab_size x embed_dim
    #axs = axs.flatten()  # flatten the array of axes to simplify indexing
    words = [str(i) for i in range(embeddings.shape[0])]
    for j, word in enumerate(words):
        axs[i].annotate(word, xy=(embeddings[j, 0], embeddings[j, 1]))
    # make /frames if it does not exist
    if not os.path.exists("frames"):
        os.mkdir("frames")
    plt.savefig(f"frames/embeddings_movie_{step:06}.png")  # change 'i' to your step variable
    plt.close()



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class ComplexMultiply(nn.Module):
    def __init__(self):
        super(ComplexMultiply, self).__init__()

    def forward(self, a, b):
        real_part = a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1]
        imag_part = a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0]
        return t.stack([real_part, imag_part], dim=-1)


class ModuloAdditionDataset(data.Dataset):
    def __init__(self, d_vocab=114, seed = 42, tiny = False):
        super().__init__()
        self.d_vocab = d_vocab
        self.tiny = tiny
        print(f"tiny is {self.tiny}")
        if not self.tiny:
            self.eq_token = d_vocab - 1  # assuming '=' is the last token in the vocabulary

    def __len__(self):
        if self.tiny:
            return 13
        else:
            return (self.d_vocab - 1) ** 2

    def __getitem__(self, idx):
        a = idx // (self.d_vocab - 1) + 1 % (self.d_vocab - 1)
        b = idx % (self.d_vocab - 1)
        res = (a + b) % (self.d_vocab - 1)
        return a, b, res


class MLP_unchunked(nn.Module):
    def __init__(self, embed_dim, vocab_size, hidden_dim, asymmetric = False, quadratic = False, complex_multiply = False, scale_embed = 1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.linear1 = nn.Linear(embed_dim, hidden_dim, bias=False)
        if asymmetric:
            self.linear1r = nn.Linear(embed_dim, hidden_dim, bias=False)
        else:
            self.linear1r = self.linear1
        self.linear2 = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.unembed = nn.Linear(embed_dim, vocab_size, bias=False)
        self.silu = nn.SiLU()
        self.scale = scale_embed
        self.complex_multiply = complex_multiply
        if quadratic:
            self.silu = Lambda(lambda x: x * x)
        if self.complex_multiply:
            self.cm = ComplexMultiply()
        self.vocab_size = vocab_size

    def forward(self, x1, x2):
        if self.complex_multiply:
            x1 = self.embedding(x1)*scale
            x2 = self.embedding(x2)*scale
            x = self.cm(x1, x2)
            x = self.unembed(x)
            return x
        else:
            x1 = self.embedding(x1)
            x2 = self.embedding(x2)
            x1 = self.linear1(x1)
            x2 = self.linear1r(x2)
            x = x1 + x2
            x = self.silu(x)
            x = self.linear2(x)
            x = self.unembed(x)
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


class MLP_theory(MLP_unchunked):
    def __init__(self, embed_dim, vocab_size, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.linear1 = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.linear2 = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.unembed = nn.Linear(embed_dim, vocab_size, bias=False)
        self.silu = nn.SiLU()
        self.vocab_size = vocab_size



def fracs(vocab_size, num_datapoints = 10): 
    return [vocab_size**(-2*n/num_datapoints) for n in range(1, num_datapoints)]
   
# model.load_state_dict(t.load("modular_addition.ckpt"))

def get_train_test_loaders(train_frac, batch_size, vocab_size, randomize=False, seed=42, sequential = False, tiny = False):
    if randomize:
        dataset = RandomOperationDataset(vocab_size)
    else:
        dataset = ModuloAdditionDataset(vocab_size, tiny = tiny)
    total_length = len(dataset)
    train_length = int(total_length * train_frac)
    test_length = total_length - train_length
    if sequential:
            train_dataset, test_dataset = data.split(
                dataset, [train_length, test_length], generator=generator
            )


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
    record_loss = False,
    suffix = "",
    freeze_mlp = False,
    freeze_first = False,
    sgd = False,
    lr=0.01,
    use_unchunked = False,
):
    vocab_size = model.vocab_size
    num_epochs = num_lin_epochs//vocab_size
    print(f"Number of parameters: {count_parameters(model)}")
    # optimizer = t.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    if sgd:
        optimizer = t.optim.SGD(model.parameters(), lr=lr, weight_decay=reg)
    else:
        optimizer = t.optim.Adam(model.parameters(), lr=lr, weight_decay=reg)        
    scheduler = t.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)
    criterion = nn.CrossEntropyLoss()
    epochs = num_epochs
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    model.to(device)
    step = 0
    frame_n = 0
    final_acc = 0
    for epoch in range(epochs):
        model.train()
        if freeze_mlp:
            model.linear1.weight.requires_grad = False
            model.linear2.weight.requires_grad = False
        if freeze_first:
            model.linear1.weight.requires_grad = False
        train_loss = 0
        for x1, x2, target in train_loader:
            x1, x2, target = x1.to(device), x2.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(x1, x2)
            loss = criterion(output, target) #+ reg * get_weight_norm(model)
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
            final_acc = val_acc
            train_loss, train_acc = test_model(model, train_loader, device, criterion)
            print(
                f"Epoch {epoch}: train loss: {float(train_loss)}, train accuracy: {float(train_acc)}, val loss: {float(val_loss)}, val accuracy: {float(val_acc)}"
            )
        scheduler.step()
    # t.save(model.state_dict(), "modular_addition.ckpt")
    if record_loss == True:
        suffix += f"val_acc_{final_acc:.2f}"
    if save_last_frame == True:
        model.project_to_fourier_mode(f"fourier_modes_{suffix}.png")
    return model


def frac_test(
    vocab_size, 
    Adam = True, 
    sequential = False, 
    graph = False, 
    embed_dim = 6,
    hidden_dim = 48, 
    quadratic = False, 
    lr = 0.005, 
    reg_orig = 0.005, 
    reg = 0.005, 
    num_datapoints = 18,
    num_lin_epochs = 500000,
):
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
        model.embedding = nn.Embedding(vocab_size, embed_dim)
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


VOCAB_SIZE = 114
TRAIN_FRAC = 0.9
BATCH_SIZE = 128
SEED = 42 
TRAIN_LOADER, TEST_LOADER = get_train_test_loaders(vocab_size = VOCAB_SIZE, train_frac = TRAIN_FRAC, batch_size = BATCH_SIZE, randomize=False, seed=SEED)
HIDDEN_DIM=60
EMBED_DIM=6
MODEL =  MLP_unchunked(embed_dim=EMBED_DIM, vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM)
NUM_LIN_EPOCHS = 100000 #300000
QUADRATIC = True
ASYMMETRIC = True

REG=0
save_frames = False
save_last_frame = True
suffix = ""
LR=0.03

def nextprime(x):
    n = int(x)
    if n > 2*x+3:
        return False
    while not isprime(n):
        n += 1
    return n


def quad_exp():
    VOCAB_SIZE = 114
    TRAIN_FRAC = 0.9
    BATCH_SIZE = 128
    SEED = 42 
    TRAIN_LOADER, TEST_LOADER = get_train_test_loaders(vocab_size = VOCAB_SIZE, train_frac = TRAIN_FRAC, batch_size = BATCH_SIZE, randomize=False, seed=SEED)
    HIDDEN_DIM=60
    EMBED_DIM=6
    MODEL =  MLP_unchunked(embed_dim=EMBED_DIM, vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM)
    NUM_LIN_EPOCHS = 100000 #300000
    QUADRATIC = True
    ASYMMETRIC = True    
    for n in range(10,16):
        for i in range(1):
            vocab_size = nextprime(int(10**(1 + n/10)))+1
            model = MLP_unchunked(embed_dim=EMBED_DIM, vocab_size=vocab_size, hidden_dim=HIDDEN_DIM, asymmetric = ASYMMETRIC, quadratic = QUADRATIC)
            frac = vocab_size**(-0.1)
            print(f"frac = {frac}, vocab_size = {vocab_size}")
            TRAIN_LOADER, TEST_LOADER = get_train_test_loaders(vocab_size = vocab_size, train_frac = frac, batch_size = BATCH_SIZE, randomize=False, seed=SEED)
            train(
                model = model, 
                train_loader = TRAIN_LOADER, 
                test_loader = TEST_LOADER,
                hidden_dim = HIDDEN_DIM, 
                embed_dim = EMBED_DIM, 
                num_lin_epochs = NUM_LIN_EPOCHS,
                reg = REG,
                save_last_frame = True,
                save_frames = False,
                suffix = f"frac_{frac}_prime_{vocab_size-1}",
                record_loss = True,
                freeze_first = True,
                lr = LR,
            )

""" 
    discussion of quad_sgd_exp(): stops working around p = 251 or so. Probably need lower lr? Or higher training frac? 
    note that even for small primes, training_frac needs to be high-ish.
"""

def phase_trans_search(lr = 0.03, num_lin_epochs = 300000):
    VOCAB_SIZE = 114
    TRAIN_FRAC = 0.9
    BATCH_SIZE = 128
    SEED = 42 
    TRAIN_LOADER, TEST_LOADER = get_train_test_loaders(vocab_size = VOCAB_SIZE, train_frac = TRAIN_FRAC, batch_size = BATCH_SIZE, randomize=False, seed=SEED)
    HIDDEN_DIM=60
    EMBED_DIM=6
    MODEL =  MLP_unchunked(embed_dim=EMBED_DIM, vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM)
    NUM_LIN_EPOCHS = num_lin_epochs
    QUADRATIC = True
    ASYMMETRIC = True
    REG=0
    save_frames = False
    save_last_frame = True
    suffix = ""
    LR=lr

    for n in range(10,20):
        for i in range(6):
            vocab_size = nextprime(int(10**(1 + n/10)))+1
            model = MLP_unchunked(embed_dim=EMBED_DIM, vocab_size=vocab_size, hidden_dim=HIDDEN_DIM, asymmetric = ASYMMETRIC, quadratic = QUADRATIC)
            frac = vocab_size**(-0.4-0.02*i)
            print(f"frac = {frac}, vocab_size = {vocab_size}")
            TRAIN_LOADER, TEST_LOADER = get_train_test_loaders(vocab_size = vocab_size, train_frac = frac, batch_size = BATCH_SIZE, randomize=False, seed=SEED)
            train(
                model = model, 
                train_loader = TRAIN_LOADER, 
                test_loader = TEST_LOADER,
                hidden_dim = HIDDEN_DIM, 
                embed_dim = EMBED_DIM, 
                num_lin_epochs = NUM_LIN_EPOCHS,
                reg = REG,
                save_last_frame = True,
                save_frames = False,
                suffix = f"frac_{frac}_prime_{vocab_size-1}",
                record_loss = True,
                freeze_first = True,
                lr = LR,
            )
"""
    comments: 
    *clear phase transition behavior around training frac 1/(vocab_size**0.5)
        *however, since I'm only searching along a factor of 10 or so, exponentials can be obscured by log or even constant factors:
            I expect something like this is going on to some extent
        *interesting questions are: 
            *how does this epend on embed_dim?
            *how does this depend on hyperparameters like lr, number of epochs, etc.?
    *weirdly, the model with these parameters works until 631, then starts breaking at 797, then is fully broken at 1009.
        *this could be an issue with lr? Or alternatively a computer issue? 
"""
def phase_search_large_embed():
    VOCAB_SIZE = 114
    TRAIN_FRAC = 0.9
    BATCH_SIZE = 128
    SEED = 42 
    TRAIN_LOADER, TEST_LOADER = get_train_test_loaders(vocab_size = VOCAB_SIZE, train_frac = TRAIN_FRAC, batch_size = BATCH_SIZE, randomize=False, seed=SEED)
    HIDDEN_DIM=100
    EMBED_DIM=50
    MODEL =  MLP_unchunked(embed_dim=EMBED_DIM, vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM)
    NUM_LIN_EPOCHS = 100000 #300000
    QUADRATIC = True
    ASYMMETRIC = True
    REG=0
    save_frames = False
    save_last_frame = True
    suffix = ""
    LR=0.03

    for n in range(10,20):
        for i in range(6):
            vocab_size = nextprime(int(10**(1 + n/10)))+1
            model = MLP_unchunked(embed_dim=EMBED_DIM, vocab_size=vocab_size, hidden_dim=HIDDEN_DIM, asymmetric = ASYMMETRIC, quadratic = QUADRATIC)
            frac = vocab_size**(-0.3-0.04*i)
            print(f"frac = {frac}, vocab_size = {vocab_size}")
            TRAIN_LOADER, TEST_LOADER = get_train_test_loaders(vocab_size = vocab_size, train_frac = frac, batch_size = BATCH_SIZE, randomize=False, seed=SEED)
            train(
                model = model, 
                train_loader = TRAIN_LOADER, 
                test_loader = TEST_LOADER,
                hidden_dim = HIDDEN_DIM, 
                embed_dim = EMBED_DIM, 
                num_lin_epochs = NUM_LIN_EPOCHS,
                reg = REG,
                save_last_frame = True,
                save_frames = False,
                suffix = f"frac_{frac}_prime_{vocab_size-1}",
                record_loss = True,
                freeze_first = True,
                lr = LR,
            )
"""
    comments: 
        *weirdly, seems to be more random than with small embed_dim
        *phase transition still around 1/p^0.5, but more random
        *with these params, clear issues with lr for p = 251 and higher (like, loss sometimes goes up drastically)
        *sometimes works for p = 251 or 317 but fully breaks for high p.
"""


def large_embed():
    VOCAB_SIZE = 114
    TRAIN_FRAC = 0.9
    BATCH_SIZE = 128
    SEED = 42 
    TRAIN_LOADER, TEST_LOADER = get_train_test_loaders(vocab_size = VOCAB_SIZE, train_frac = TRAIN_FRAC, batch_size = BATCH_SIZE, randomize=False, seed=SEED)
    HIDDEN_DIM=100
    EMBED_DIM=50
    MODEL =  MLP_unchunked(embed_dim=EMBED_DIM, vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM)
    NUM_LIN_EPOCHS = 100000 #300000
    QUADRATIC = True
    ASYMMETRIC = True
    REG=0
    save_frames = False
    save_last_frame = True
    suffix = ""
    LR=0.03

    for n in range(10,20):
        for i in range(1):
            vocab_size = nextprime(int(10**(1 + n/10)))+1
            model = MLP_unchunked(embed_dim=EMBED_DIM, vocab_size=vocab_size, hidden_dim=HIDDEN_DIM, asymmetric = ASYMMETRIC, quadratic = QUADRATIC)
            frac = 0.4
            print(f"frac = {frac}, vocab_size = {vocab_size}")
            TRAIN_LOADER, TEST_LOADER = get_train_test_loaders(vocab_size = vocab_size, train_frac = frac, batch_size = BATCH_SIZE, randomize=False, seed=SEED)
            train(
                model = model, 
                train_loader = TRAIN_LOADER, 
                test_loader = TEST_LOADER,
                hidden_dim = HIDDEN_DIM, 
                embed_dim = EMBED_DIM, 
                num_lin_epochs = NUM_LIN_EPOCHS,
                reg = REG,
                save_last_frame = True,
                save_frames = False,
                suffix = f"frac_{frac}_prime_{vocab_size-1}",
                record_loss = True,
                freeze_first = True,
                lr = LR,
            )
"""
    comments: 
    *This mostly works great! Sometimes breaks for large p (order of 200-300), probably for LR reasons.
    *
"""

def sgd_embed_small_lr():
    VOCAB_SIZE = 114
    TRAIN_FRAC = 0.9
    BATCH_SIZE = 128
    SEED = 42 
    TRAIN_LOADER, TEST_LOADER = get_train_test_loaders(vocab_size = VOCAB_SIZE, train_frac = TRAIN_FRAC, batch_size = BATCH_SIZE, randomize=False, seed=SEED)
    HIDDEN_DIM=200
    EMBED_DIM=200
    MODEL =  MLP_unchunked(embed_dim=EMBED_DIM, vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM)
    NUM_LIN_EPOCHS = 1000000 
    QUADRATIC = True
    ASYMMETRIC = True
    REG=0
    save_frames = False
    save_last_frame = True
    suffix = ""
    LR=0.001

    for n in range(10,20):
        for i in range(1):
            vocab_size = nextprime(int(10**(1 + n/10)))+1
            model = MLP_unchunked(embed_dim=EMBED_DIM, vocab_size=vocab_size, hidden_dim=HIDDEN_DIM, asymmetric = ASYMMETRIC, quadratic = QUADRATIC)
            frac = 0.4
            print(f"frac = {frac}, vocab_size = {vocab_size}")
            TRAIN_LOADER, TEST_LOADER = get_train_test_loaders(vocab_size = vocab_size, train_frac = frac, batch_size = BATCH_SIZE, randomize=False, seed=SEED)
            train(
                model = model, 
                train_loader = TRAIN_LOADER, 
                test_loader = TEST_LOADER,
                hidden_dim = HIDDEN_DIM, 
                embed_dim = EMBED_DIM, 
                num_lin_epochs = NUM_LIN_EPOCHS,
                reg = REG,
                save_last_frame = True,
                save_frames = False,
                suffix = f"frac_{frac}_prime_{vocab_size-1}",
                record_loss = True,
                freeze_first = True,
                lr = LR,
            )
"""
    notes:
    *it works well with large LR, and just doesn't work at all with tiny LR. Making me think that bootstrap phenomena are involved.
"""

def check_bootstrap():
    VOCAB_SIZE = 114
    TRAIN_FRAC = 0.9
    BATCH_SIZE = 128
    SEED = 42 
    TRAIN_LOADER, TEST_LOADER = get_train_test_loaders(vocab_size = VOCAB_SIZE, train_frac = TRAIN_FRAC, batch_size = BATCH_SIZE, randomize=False, seed=SEED)
    HIDDEN_DIM=40
    EMBED_DIM=6
    MODEL =  MLP_unchunked(embed_dim=EMBED_DIM, vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM)
    NUM_LIN_EPOCHS = 1000000 
    QUADRATIC = True
    ASYMMETRIC = True
    REG=0
    save_frames = False
    save_last_frame = True
    suffix = ""
    LR=0.00005

    for n in range(10,20):
        for i in range(1):
            vocab_size = nextprime(int(10**(1 + n/10)))+1
            model = MLP_unchunked(embed_dim=EMBED_DIM, vocab_size=vocab_size, hidden_dim=HIDDEN_DIM, asymmetric = ASYMMETRIC, quadratic = QUADRATIC)
            frac = 0.4
            print(f"frac = {frac}, vocab_size = {vocab_size}")
            TRAIN_LOADER, TEST_LOADER = get_train_test_loaders(vocab_size = vocab_size, train_frac = frac, batch_size = BATCH_SIZE, randomize=False, seed=SEED)
            train(
                model = model, 
                train_loader = TRAIN_LOADER, 
                test_loader = TEST_LOADER,
                hidden_dim = HIDDEN_DIM, 
                embed_dim = EMBED_DIM, 
                num_lin_epochs = NUM_LIN_EPOCHS,
                reg = REG,
                save_last_frame = True,
                save_frames = False,
                suffix = f"frac_{frac}_prime_{vocab_size-1}",
                record_loss = True,
                freeze_first = True,
                lr = LR,
            )

"""notes:
    *Here the network is somewhat underparametrized
    *We see pretty definitively that it groks without bootstrap/sgd randomness issues, because the lr is so freakin tiny
"""


def check_bootstrap_and_embed_dim(sgd=False):
    VOCAB_SIZE = 114
    TRAIN_FRAC = 0.9
    BATCH_SIZE = 128
    SEED = 42 
    TRAIN_LOADER, TEST_LOADER = get_train_test_loaders(vocab_size = VOCAB_SIZE, train_frac = TRAIN_FRAC, batch_size = BATCH_SIZE, randomize=False, seed=SEED)
    HIDDEN_DIM=40
    EMBED_DIM=6
    # MODEL =  MLP_unchunked(embed_dim=embed_dim, vocab_size=VOCAB_SIZE, hidden_dim=hidden_dim)
    NUM_LIN_EPOCHS = 300000
    QUADRATIC = True
    ASYMMETRIC = True
    REG=0
    save_frames = False
    save_last_frame = True
    suffix = ""
    LR=0.001
    for i in range(4,30):
        vocab_size = 114
        embed_dim=i
        print(f"embed_dim={embed_dim}")
        hidden_dim = 60
        model = MLP_unchunked(embed_dim=embed_dim, vocab_size=vocab_size, hidden_dim=hidden_dim, asymmetric = ASYMMETRIC, quadratic = QUADRATIC)
        frac = 0.4
        print(f"frac = {frac}, vocab_size = {vocab_size}")
        TRAIN_LOADER, TEST_LOADER = get_train_test_loaders(vocab_size = vocab_size, train_frac = frac, batch_size = BATCH_SIZE, randomize=False, seed=SEED)
        train(
            model = model, 
            train_loader = TRAIN_LOADER, 
            test_loader = TEST_LOADER,
            hidden_dim = hidden_dim, 
            embed_dim = embed_dim, 
            num_lin_epochs = NUM_LIN_EPOCHS,
            reg = REG,
            save_last_frame = True,
            save_frames = False,
            suffix = f"frac_{frac}_prime_{vocab_size-1}_embed_{embed_dim}",
            record_loss = True,
            freeze_first = True,
            lr = LR,
            sgd = sgd
        )

def check_sgd_with_bootstrap_lr():
    VOCAB_SIZE = 114
    TRAIN_FRAC = 0.9
    BATCH_SIZE = 128
    SEED = 42 
    TRAIN_LOADER, TEST_LOADER = get_train_test_loaders(vocab_size = VOCAB_SIZE, train_frac = TRAIN_FRAC, batch_size = BATCH_SIZE, randomize=False, seed=SEED)
    HIDDEN_DIM=50
    EMBED_DIM=6
    MODEL =  MLP_unchunked(embed_dim=EMBED_DIM, vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM)
    NUM_LIN_EPOCHS = 100000 
    QUADRATIC = True
    ASYMMETRIC = True
    REG=0
    SCALE_EMBED = np.sqrt(VOCAB_SIZE - 1)
    save_frames = False
    save_last_frame = True
    suffix = ""
    LR=0.01
    SGD = True

    for n in range(1,12):
        for i in range(1):
            lr = 10**(-n/3)
            vocab_size = 114
            model = MLP_unchunked(embed_dim=EMBED_DIM, vocab_size=vocab_size, hidden_dim=HIDDEN_DIM, asymmetric = ASYMMETRIC, quadratic = QUADRATIC, scale_embed = SCALE_EMBED)
            frac = 0.4
            print(f"frac = {frac}, vocab_size = {vocab_size}")
            TRAIN_LOADER, TEST_LOADER = get_train_test_loaders(vocab_size = vocab_size, train_frac = frac, batch_size = BATCH_SIZE, randomize=False, seed=SEED)
            train(
                model = model, 
                train_loader = TRAIN_LOADER, 
                test_loader = TEST_LOADER,
                hidden_dim = HIDDEN_DIM, 
                embed_dim = EMBED_DIM, 
                num_lin_epochs = NUM_LIN_EPOCHS,
                reg = REG,
                save_last_frame = True,
                save_frames = False,
                suffix = f"frac_{frac}_prime_{vocab_size-1}_lr_{lr}",
                record_loss = True,
                freeze_first = True,
                lr = lr,
                sgd = SGD,
            )


def comp_mult_exp():
    VOCAB_SIZE = None
    TRAIN_FRAC = 0.9
    BATCH_SIZE = 128
    SEED = 42 
    TRAIN_LOADER, TEST_LOADER = get_train_test_loaders(vocab_size = VOCAB_SIZE, train_frac = TRAIN_FRAC, batch_size = BATCH_SIZE, randomize=False, seed=SEED)
    HIDDEN_DIM=0
    EMBED_DIM=2
    MODEL =  MLP_unchunked(embed_dim=EMBED_DIM, vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM)
    NUM_LIN_EPOCHS = 200000 #300000
    QUADRATIC = True
    ASYMMETRIC = True  
    REG=0
    save_frames = False
    save_last_frame = True
    suffix = ""
    LR=0.03
    SGD = False
 

    for n in range(10,20):
        for i in range(14,19): #for adam
            vocab_size = nextprime(int(10**(1 + n/10)))+1
            model = MLP_unchunked(embed_dim=EMBED_DIM, vocab_size=vocab_size, hidden_dim=HIDDEN_DIM, asymmetric = ASYMMETRIC, quadratic = QUADRATIC, complex_multiply = True)
            frac = vocab_size**(-0.4-0.015*i)
            print(f"frac = {frac}, vocab_size = {vocab_size}")
            TRAIN_LOADER, TEST_LOADER = get_train_test_loaders(vocab_size = vocab_size, train_frac = frac, batch_size = BATCH_SIZE, randomize=False, seed=SEED)
            train(
                model = model, 
                train_loader = TRAIN_LOADER, 
                test_loader = TEST_LOADER,
                hidden_dim = HIDDEN_DIM, 
                embed_dim = EMBED_DIM, 
                num_lin_epochs = NUM_LIN_EPOCHS,
                reg = REG,
                save_last_frame = True,
                save_frames = False,
                suffix = f"frac_{frac}_prime_{vocab_size-1}",
                record_loss = True,
                freeze_first = True,
                lr = LR,
                sgd = SGD,
            )
"""
    comments: this works ok though sensitive to LR. I haven't been able to get phase transition below something like frac = p**(-0.65)
"""

def comp_mult_movie():
    VOCAB_SIZE = 114
    TRAIN_FRAC = 0.9
    BATCH_SIZE = 4
    SEED = 42 
    # TRAIN_LOADER, TEST_LOADER = get_train_test_loaders(vocab_size = VOCAB_SIZE, train_frac = TRAIN_FRAC, batch_size = BATCH_SIZE, randomize=False, seed=SEED)
    HIDDEN_DIM=0
    EMBED_DIM=2
    MODEL =  MLP_unchunked(embed_dim=EMBED_DIM, vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM)
    NUM_LIN_EPOCHS = 100000 #300000
    QUADRATIC = True
    ASYMMETRIC = True  
    REG=0
    save_frames = False
    save_last_frame = True
    suffix = ""
    LR=0.001
    SGD = True
    model = MLP_unchunked(embed_dim=EMBED_DIM, vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, asymmetric = ASYMMETRIC, quadratic = QUADRATIC, complex_multiply = True)
    print(f"frac = {TRAIN_FRAC}, vocab_size = {VOCAB_SIZE}")
    TRAIN_LOADER, TEST_LOADER = get_train_test_loaders(vocab_size = VOCAB_SIZE, train_frac = TRAIN_FRAC, batch_size = BATCH_SIZE, randomize=False, seed=SEED, tiny=True)
    train(
        model = model, 
        train_loader = TRAIN_LOADER, 
        test_loader = TEST_LOADER,
        hidden_dim = HIDDEN_DIM, 
        embed_dim = EMBED_DIM, 
        num_lin_epochs = NUM_LIN_EPOCHS,
        reg = REG,
        save_last_frame = False,
        save_frames = True,
        suffix = f"frac_{TRAIN_FRAC}_prime_{VOCAB_SIZE-1}",
        record_loss = True,
        freeze_first = True,
        lr = LR,
        sgd = SGD,
            )



if __name__ == "__main__":
#    comp_mult_movie()
#    comp_mult_exp()
    check_sgd_with_bootstrap_lr()
