import torch as t
import torch.nn as nn
import torch.utils.data as data
import hashlib

from sklearn.decomposition import PCA
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


def all_subsets(s):
    return [set(comb) for i in range(len(s) + 1) for comb in combinations(s, i)]


def hash_with_seed(value, seed = 42):
    m = hashlib.sha256()
    m.update(str(seed).encode('utf-8'))
    m.update(str(value).encode('utf-8'))
    return int(m.hexdigest(), 16)



class RandomOperationDataset(data.Dataset):
    def __init__(self, d_vocab=114):
        super().__init__()
        self.d_vocab = d_vocab
        self.eq_token = d_vocab - 1  # assuming '=' is the last token in the vocabulary

    def __len__(self):
        return (self.d_vocab - 1) ** 2

    def __getitem__(self, idx):
        a = idx // (self.d_vocab - 1)
        b = idx % (self.d_vocab - 1)
        res = hash_with_seed(a + b + a*b*self.d_vocab) % (self.d_vocab - 1)
        return a, b, res

class ModuloAdditionDataset(data.Dataset):
    def __init__(self, d_vocab=114):
        super().__init__()
        self.d_vocab = d_vocab
        self.eq_token = d_vocab - 1  # assuming '=' is the last token in the vocabulary

    def __len__(self):
        return (self.d_vocab - 1) ** 2

    def __getitem__(self, idx):
        a = idx // (self.d_vocab - 1)
        b = idx % (self.d_vocab - 1)
        res = (a + b) % (self.d_vocab - 1)
        return a, b, res


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_circle_embeddings(embed_dim, vocab_size):
    n_blocks = embed_dim // 2
    # k_values = [randint(1, vocab_size -1) for _ in range(n_blocks)]
    k_values = [3, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47][:n_blocks]
    print("Using K values", k_values)
    arange_tensor = t.arange(vocab_size - 1).float()
    div_tensor = 2 * t.pi * arange_tensor / (vocab_size - 1)

    # Create M_k without transposing
    M_k_list = [
        t.stack(
            [
                t.cat((t.cos(k * div_tensor), t.tensor([0]))),
                t.cat((t.sin(k * div_tensor), t.tensor([0]))),
            ],
            dim=1,
        )
        for k in k_values
    ]
    return t.cat(M_k_list, dim=1)



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
        self.linear3 = nn.Linear(embed_dim, vocab_size, bias=False)
        self.silu = nn.SiLU()

    def forward(self, x1, x2):
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
        return self.linear3(x)


def fourier_matrix(n):
    fourier_matrix = t.zeros((n, n), dtype=t.cfloat)
    for i in range(n):
        for j in range(n):
            theta = t.tensor(2 * t.pi * i * j / (n))
            fourier_matrix[i, j] = t.complex(t.cos(theta), t.sin(theta))
    return fourier_matrix

def inv_fourier_matrix(n):
    inv_fourier_matrix = t.zeros((n, n), dtype=t.cfloat)
    for i in range(n):
        for j in range(n):
            theta = t.tensor(2 * t.pi * i * j / (n))
            inv_fourier_matrix[i, j] = t.complex(t.cos(-theta), t.sin(-theta))
    inv_fourier_matrix *= 1/(n)
    return inv_fourier_matrix


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

    def ablate_other_modes(self, modes):
        n = self.vocab_size - 1
        fourier_modes = self.get_fourier_modes()
        inv_fourier_mat = inv_fourier_matrix(n)
        # Create a mask with 1's at specified modes and 0's elsewhere
        mask = t.zeros(n, dtype=t.float32)
        mask.scatter_(0, modes, 1.0)
        # Apply the mask to the Fourier modes
        fourier_modes *= mask.to(t.cfloat)
        # Transform back to the spatial domain using the inverse Fourier matrix
        new_embed = t.matmul(inv_fourier_mat, fourier_modes)
        # Use only the real part of the result to set as the new embedding weights
        self.embedding.weight = nn.Parameter(new_embed.real.to(self.embedding.device))


def add_embedding_noise(model, circuit_nums, device="cpu"):
    embedding_weights = model.embedding.detach().clone().cpu()
    norm_param = embedding_weights[:, :2].norm(p=2).cpu() * 5
    vec = t.zeros(embedding_weights.shape[1])
    for i in range(embedding_weights.shape[1] // 2):
        if i in circuit_nums:
            vec[2 * i] = 1
            vec[2 * i + 1] = 1
    vec = t.randn_like(embedding_weights) * vec.unsqueeze(0)
    vec = vec / t.norm(vec, p=2)
    vec *= norm_param
    embedding_weights += vec
    # set model embedding to new embedding
    model.embedding = nn.Parameter(embedding_weights.to(device))
    return model


def add_embedding_noise_2(model, circuit_nums, frac=0.5, device="cpu"):
    embedding_weights = model.embedding.detach().clone().cpu()
    mask = t.zeros(embedding_weights.shape[1])
    p = frac / len(circuit_nums)
    for i in range(embedding_weights.shape[1] // 2):
        if i in circuit_nums:
            mask[2 * i] = 1
            mask[2 * i + 1] = 1
    mask = t.ones_like(embedding_weights) * mask.unsqueeze(0)
    inv_mask = t.ones_like(embedding_weights) * (1 - mask.unsqueeze(0))
    rand_mask = t.rand_like(embedding_weights) > p
    full_mask = mask * rand_mask + inv_mask
    embedding_weights = (embedding_weights * full_mask).float()[0]
    model.embedding = nn.Parameter(embedding_weights.to(device))
    return model


def plot_embeddings_chunks(model, filename="embeddings_chunks.png"):
    plt.clf()
    embeddings = model.embedding.detach().cpu()  # vocab_size x embed_dim
    chunked = t.chunk(embeddings, embeddings.shape[-1] // 2, dim=-1)
    n = embeddings.shape[-1] // 2

    # calculate number of rows and columns for subplots
    rows = int(n**0.5)
    cols = n // rows
    if rows * cols < n:  # if not enough subplots, add an extra column
        cols += 1

    # visualise each vocab_size x 2 chunk in a subplot
    fig, axs = plt.subplots(rows, cols, figsize=(15, 15))
    axs = axs.flatten()  # flatten the array of axes to simplify indexing
    for i, chunk in enumerate(chunked):
        axs[i].scatter(chunk[:, 0], chunk[:, 1])
        words = [str(i) for i in range(embeddings.shape[0])]
        for j, word in enumerate(words):
            axs[i].annotate(word, xy=(chunk[j, 0], chunk[j, 1]))

    for ax in axs:
        bound = max([abs(x) for x in ax.get_xlim()] + [abs(y) for y in ax.get_ylim()])
        ax.set_xlim([-bound, bound])
        ax.set_ylim([-bound, bound])

    plt.tight_layout()
    plt.savefig(filename)


def plot_embeddings(model, vocab_size):
    plt.clf()
    try:
        embeddings = model.embedding.detach().cpu().numpy()
    except:
        embeddings = model.embedding.weight.detach().cpu().numpy()
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    plt.figure(figsize=(10, 10))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
    words = [str(i) for i in range(vocab_size)]
    for i, word in enumerate(words):
        plt.annotate(word, xy=(embeddings_2d[i, 0], embeddings_2d[i, 1]))
    plt.savefig("embeddings.png")


def get_train_test_loaders(train_frac, batch_size, vocab_size, randomize=False):
    if randomize:
        dataset = RandomOperationDataset(vocab_size)
    else:
        dataset = ModuloAdditionDataset(vocab_size)
    total_length = len(dataset)
    train_length = int(total_length * train_frac)
    test_length = total_length - train_length
    train_dataset, test_dataset = data.random_split(
        dataset, [train_length, test_length]
    )
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train(
    train_loader,
    test_loader,
    vocab_size=114,
    hidden_dim=32,
    embed_dim=16,
    num_epochs = 500,
    save_frames=True,
    reg=0.005,
    use_circular_embeddings=False,
    freeze_embed=False,
    use_unchunked=False,
):
    if use_unchunked:
        model = MLP_unchunked(
            vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim
        )
    else:
        model = MLP(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            freeze_embed=freeze_embed,
            use_circular_embeddings=use_circular_embeddings,
        )
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
        ###disrupt step
        # if epoch == 50:
        #     model.embedding = nn.Parameter(t.randn(vocab_size, embed_dim))
        #     optimizer = t.optim.Adam(model.parameters(), lr=0.01)

    t.save(model.state_dict(), "modular_addition.ckpt")
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


def experiment(model, test_loader, frac=0.5):
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    model = model.to(device)
    orig_embedding = model.embedding.detach().clone()
    model.eval()
    val_loss, val_acc = test_model(model, test_loader, device, nn.CrossEntropyLoss())
    print(f"Original: loss: {float(val_loss)}, accuracy: {float(val_acc)}")
    for subset in all_subsets(list(range(model.embedding.shape[-1] // 2))):
        if len(subset) == 0:
            continue
        model = add_embedding_noise_2(model, subset, frac=frac, device=device)
        val_loss, val_acc = test_model(
            model, test_loader, device, nn.CrossEntropyLoss()
        )
        print(
            f"Embedding noise only on circuits ({subset}): loss: {float(val_loss)}, accuracy: {float(val_acc)}"
        )
        model.embedding = t.nn.Parameter(orig_embedding.to(device))

def train_manyprimes(
        p_min = 17,
        p_max = 217,
        train_frac = 0.7,
        batch_size = 256,
        embed_dim = 8,
        hidden_dim = 24,
        use_unchunked = True,
        randomize = False,
    ):
    primes = list(primerange(p_min, p_max+1))  
    for p in primes:
        vocab_size = p+1
        print(f"prime{p}")
        train_loader, test_loader = get_train_test_loaders(
            train_frac, batch_size, vocab_size, randomize = randomize
        )
        train(
            train_loader,
            test_loader,
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            num_epochs = 2000,
            save_frames=False,
            reg=0,
            use_circular_embeddings=False,
            freeze_embed=False,
            use_unchunked=True,
        )
        run_movie_cmd(suffix = f"p_{p}")
    if use_unchunked:
        model = MLP_unchunked(
            vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim
        )
    else:
        model = MLP(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            freeze_embed=False,
            use_circular_embeddings=False,
        )
    model.load_state_dict(t.load("modular_addition.ckpt"))
    model.eval()
    plot_embeddings(model, vocab_size)
    if not use_unchunked:
        plot_embeddings_chunks(model)
        experiment(model, test_loader, frac=0.5)


def train_manyfractions(
        p = 23,     
        batch_size = 256,
        embed_dim = 6,
        hidden_dim = 18,
        use_unchunked = True,
        randomize = False,
        save_frames = False
    ):
    vocab_size = p+1
    fractions = [n/10 for n in range(1, 10)]
    for fraction in fractions:
        train_frac = fraction
        print(f"fraction {fraction}")
        train_loader, test_loader = get_train_test_loaders(
            train_frac, batch_size, vocab_size, randomize = randomize
        )
        train(
            train_loader,
            test_loader,
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            num_epochs = 3500,
            save_frames=save_frames,
            reg=0,
            use_circular_embeddings=False,
            freeze_embed=False,
            use_unchunked=use_unchunked,
        )
        run_movie_cmd(suffix = f"p_{p}_frac_{fraction}")
    if use_unchunked:
        model = MLP_unchunked(
            vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim
        )
    else:
        model = MLP(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            freeze_embed=False,
            use_circular_embeddings=False,
        )
    model.load_state_dict(t.load("modular_addition.ckpt"))
    model.eval()
    plot_embeddings(model, vocab_size)
    if not use_unchunked:
        plot_embeddings_chunks(model)
        experiment(model, test_loader, frac=0.5)


if __name__ == "__main__":
    train_manyfractions(randomize = False, save_frames = True)

    # p = 229
    # train_frac = 0.9
    # batch_size = 256
    # vocab_size = p+1
    # embed_dim = 8
    # hidden_dim = 30
    # use_unchunked = True
    # randomize = False
    # train_loader, test_loader = get_train_test_loaders(
    #     train_frac, batch_size, vocab_size, randomize = randomize
    # )
    # train(
    #     train_loader,
    #     test_loader,
    #     vocab_size=vocab_size,
    #     hidden_dim=hidden_dim,
    #     embed_dim=embed_dim,
    #     num_epochs = 50,
    #     save_frames=True,
    #     reg=0,
    #     use_circular_embeddings=False,
    #     freeze_embed=False,
    #     use_unchunked=True,
    # )
    # run_movie_cmd




    # train_frac = 0.7
    # batch_size = 256
    # vocab_size = 38
    # embed_dim = 14
    # hidden_dim = 32
    # use_unchunked = True
    # train_loader, test_loader = get_train_test_loaders(
    #     train_frac, batch_size, vocab_size
    # )
    # train(
    #     train_loader,
    #     test_loader,
    #     vocab_size=vocab_size,
    #     embed_dim=embed_dim,
    #     hidden_dim=hidden_dim,
    #     save_frames=True,
    #     use_circular_embeddings=False,
    #     reg=0.001,
    #     freeze_embed=False,
    #     use_unchunked=use_unchunked,
    # )

