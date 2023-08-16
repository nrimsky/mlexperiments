import torch as t
import torch.nn as nn
import torch.utils.data as data
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from modular_addition.generate_movie import plot_embeddings_movie, run_movie_cmd
from random import randint

def get_module_parameters(model):
    return model.fc.parameters()


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
    k_values = [randint(1, vocab_size -1) for _ in range(n_blocks)]
    print("Using K values", k_values)
    arange_tensor = t.arange(vocab_size - 1).float()
    div_tensor = 2 * t.pi * arange_tensor / (vocab_size - 1)

    # Create M_k without transposing
    M_k_list = [
        t.stack([
            t.cat((t.cos(k * div_tensor), t.tensor([0]))),
            t.cat((t.sin(k * div_tensor), t.tensor([0])))
        ], dim=1)
        for k in k_values
    ]
    embeddings = t.cat(M_k_list, dim=1)
    embeddings *= 0.7
    return embeddings

def make_random_embeddings(embed_dim, vocab_size):
    return t.randn(vocab_size, embed_dim)
    
class MLP(nn.Module):
    def __init__(self, embed_dim, vocab_size, hidden_dim):
        super().__init__()
        self.n_blocks = embed_dim // 2
        self.embedding = nn.Parameter()
        self.embedding = nn.Parameter(make_circle_embeddings(embed_dim, vocab_size))
        self.linear1_weights = nn.Parameter(t.randn(self.n_blocks, 2, hidden_dim))
        self.linear2_weights = nn.Parameter(t.randn(self.n_blocks, hidden_dim, 2))
        self.linear3 = nn.Linear(embed_dim, vocab_size, bias=False)
        self.silu = nn.SiLU()
        self.embedding.requires_grad = False


    def forward(self, x1, x2):
        x1 = self.embedding[x1].view(-1, self.n_blocks, 2)
        x2 = self.embedding[x2].view(-1, self.n_blocks, 2)
        # Apply different linear transformations to each block
        x1 = t.einsum('bnj,njk->bnk', x1, self.linear1_weights)
        x2 = t.einsum('bnj,njk->bnk', x2, self.linear1_weights)
    
        x = x1 + x2
        # x = x ** 2
        x = self.silu(x)
        x = t.einsum('bnk,nkj->bnj', x, self.linear2_weights)
        # Flatten the tensor from shape (batch_size, n_blocks, 2) to (batch_size, n_blocks*2) = (batch_size, embed_dim)
        x = x.reshape(x.size(0), -1)
        return self.linear3(x)


def plot_embeddings_chunks(model):
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
    fig, axs = plt.subplots(rows, cols, figsize=(15, 15))
    axs = axs.flatten()  # flatten the array of axes to simplify indexing
    for i, chunk in enumerate(chunked):
        axs[i].scatter(chunk[:, 0], chunk[:, 1])
        words = [str(i) for i in range(embeddings.shape[0])]
        for j, word in enumerate(words):
            axs[i].annotate(word, xy=(chunk[j, 0], chunk[j, 1]))
    plt.tight_layout()  # adjust spacing between subplots
    plt.savefig("embeddings_chunks.png")


def plot_embeddings(model, vocab_size):
    plt.clf()
    embeddings = model.embedding.detach().cpu().numpy()
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    plt.figure(figsize=(10, 10))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
    words = [str(i) for i in range(vocab_size)] 
    for i, word in enumerate(words):
        plt.annotate(word, xy=(embeddings_2d[i, 0], embeddings_2d[i, 1]))
    plt.savefig("embeddings.png")

def get_train_test_loaders(train_frac, batch_size, vocab_size):
    dataset = ModuloAdditionDataset(vocab_size)
    total_length = len(dataset)
    train_length = int(total_length * train_frac)
    test_length = total_length - train_length
    train_dataset, test_dataset = data.random_split(dataset, [train_length, test_length])
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def get_weight_norm(model):
    return sum(t.norm(p, p=2) for p in model.parameters() if p.requires_grad)

def train(vocab_size = 114, train_frac = 0.3, hidden_dim = 32, embed_dim = 16, save_frames = True, epochs = 30000):
    model = MLP(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim)
    print(f"Number of parameters: {count_parameters(model)}")
    batch_size = 256
    train_loader, test_loader = get_train_test_loaders(train_frac, batch_size, vocab_size)
    optimizer = t.optim.SGD(model.parameters(), lr=0.02)
    criterion = nn.CrossEntropyLoss()
    # epochs = 10000
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    model.to(device)
    old_acc = 0
    step = 0
    for epoch in range(epochs):
        if epoch == epochs//2:
            model.embedding.requires_grad = True
            optimizer = t.optim.SGD(model.parameters(), lr=0.005)
        model.train()
        train_loss = 0
        reg_loss = 0
        for x1, x2, target in train_loader:
            x1, x2, target = x1.to(device), x2.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(x1, x2)
            reg = get_weight_norm(model) * 0.02
            loss = criterion(output, target) + reg
            train_loss += loss.item()
            reg_loss += reg.item()
            loss.backward()
            optimizer.step()
        model.eval()
        if save_frames:
            if epoch % 500 == 0: 
                with t.no_grad():
                    step += 1
                    plot_embeddings_movie(model, step)
        if epoch % 200 == 0:
            val_loss, val_acc = test_model(model, test_loader, device, criterion)
            print(f"Epoch {epoch}: train loss {train_loss}; reg loss {reg_loss}; test loss {val_loss}; test acc {val_acc}; old acc {old_acc}")
            old_acc = val_acc
    t.save(model.state_dict(), "modular_addition.ckpt")


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


if __name__ == "__main__":
    train(vocab_size = 38, train_frac = 0.8, embed_dim = 6, hidden_dim = 8, save_frames = True, epochs = 40000)
    model = MLP(vocab_size=38, embed_dim=12, hidden_dim=8)
    model.load_state_dict(t.load("modular_addition.ckpt"))
    model.eval()
    plot_embeddings(model, 38)
    plot_embeddings_chunks(model)
    run_movie_cmd()