import torch as t
import torch.nn as nn
import torch.utils.data as data
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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

class MLP(nn.Module):
    def __init__(self, embed_dim, vocab_size, hidden_dim):
        super().__init__()
        self.embedding = nn.Parameter(t.randn(vocab_size, embed_dim))
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, vocab_size)
        self.silu = nn.SiLU()

    def forward(self, x1, x2):
        x1 = self.embedding[x1] 
        x2 = self.embedding[x2]
        x = x1 + x2
        x = self.linear1(x)
        x = x ** 2
        return self.linear2(x)
    

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


def train(vocab_size = 114, train_frac = 0.3, hidden_dim = 32, embed_dim = 16):
    model = MLP(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim)
    print(f"Number of parameters: {count_parameters(model)}")
    dataset = ModuloAdditionDataset(vocab_size)

    total_length = len(dataset)
    train_length = int(total_length * train_frac)
    test_length = total_length - train_length
    train_dataset, test_dataset = data.random_split(dataset, [train_length, test_length])

    batch_size = 256
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    optimizer = t.optim.AdamW(model.parameters(), lr=2e-2, weight_decay=.5)
    criterion = nn.CrossEntropyLoss()
    epochs = 5000
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x1, x2, target in train_loader:
            x1, x2, target = x1.to(device), x2.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(x1, x2)
            loss = criterion(output, target)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        val_acc = 0
        if epoch % 50 == 0:
            with t.no_grad():
                for x1, x2, target in test_loader:
                    x1, x2, target = x1.to(device), x2.to(device), target.to(device)
                    output = model(x1, x2)
                    loss = criterion(output, target)
                    val_loss += loss.item()
                    val_acc += (output.argmax(dim=-1) == target).float().mean()
            print(f"Epoch {epoch}: train loss {train_loss / len(train_loader)}; test loss {val_loss / len(test_loader)}; test acc {val_acc / len(test_loader)}")
    t.save(model.state_dict(), "modular_addition.ckpt")


if __name__ == "__main__":
    train(vocab_size = 114, train_frac = 0.3, embed_dim = 12, hidden_dim = 32) #best so far embed_dim = 12, hidden_dim = 32
    model = MLP(vocab_size=114, embed_dim=12, hidden_dim=32)
    model.load_state_dict(t.load("modular_addition.ckpt"))
    model.eval()
    plot_embeddings(model, 114)