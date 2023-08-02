import torch
from torch import nn
from torch.optim import AdamW
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class ModuloAdditionDataset(Dataset):
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
        return torch.tensor([a, b, self.eq_token]), res


class Transformer(nn.Module):
    def __init__(
        self, d_residual=128, n_heads=4, d_hidden_mlp=512, d_vocab=114, input_len=3
    ):
        super().__init__()
        self.d_model = d_residual
        self.n_heads = n_heads
        self.d_head = d_residual // n_heads
        self.input_len = input_len

        self.positional_embeddings = nn.Embedding(input_len, d_residual)
        self.input_embeddings = nn.Embedding(d_vocab, d_residual)

        self.q = nn.Linear(d_residual, d_residual)
        self.k = nn.Linear(d_residual, d_residual)
        self.v = nn.Linear(d_residual, d_residual)
        self.o = nn.Linear(d_residual, d_residual)

        self.mlp = nn.Sequential(
            nn.Linear(d_residual, d_hidden_mlp),
            nn.GELU(),
            nn.Linear(d_hidden_mlp, d_residual),
        )

        self.unembed = nn.Linear(d_residual, d_vocab, bias=False)

    def forward(self, input):
        batch_size, seq_length = input.shape 
        assert seq_length == self.input_len

        # compute positional embeddings
        position = torch.arange(0, seq_length, device=input.device).unsqueeze(0)
        position = self.positional_embeddings(position)

        # compute input embeddings
        input = self.input_embeddings(input)

        embeddings = input + position

        Q = (
            self.q(embeddings)
            .view(batch_size, seq_length, self.n_heads, -1)
            .transpose(1, 2)
        ) 
        K = (
            self.k(embeddings)
            .view(batch_size, seq_length, self.n_heads, -1)
            .transpose(1, 2)
        ) 
        V = (
            self.v(embeddings)
            .view(batch_size, seq_length, self.n_heads, -1)
            .transpose(1, 2)
        )

        # compute dot-product attention
        attn = Q @ K.transpose(-2, -1) / torch.sqrt(torch.tensor(self.d_head))
        attn = attn.softmax(dim=-1)
        attn_output = attn @ V

        # apply o 
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, -1) 
        attn_output = self.o(attn_output)

        # pass through MLP
        output = self.mlp(attn_output)

        # unembed
        output = self.unembed(output)

        return output


def train(model, epochs=50000, vocab_size=114, train_frac=0.3):
    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ModuloAdditionDataset(vocab_size)

    total_length = len(dataset)
    train_length = int(total_length * train_frac)
    test_length = total_length - train_length
    train_dataset, test_dataset = random_split(dataset, [train_length, test_length])

    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.to(device)
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            last_token_logits = output[:, -1, :]
            loss = criterion(last_token_logits, target)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        val_acc = 0
        if epoch % 100 == 0:
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    last_token_logits = output[:, -1, :]
                    loss = criterion(last_token_logits, target)
                    val_loss += loss.item()
                    val_acc += (last_token_logits.argmax(dim=-1) == target).float().mean()

            print(
                "Epoch: {} - Train Loss: {:.4f} - Test Loss: {:.4f} - Test Acc: {:.4f}".format(
                    epoch + 1, train_loss / len(train_loader), val_loss / len(test_loader), val_acc / len(test_loader)
                )
            )

    # Save model
    torch.save(model.state_dict(), "modular_addition.ckpt")


if __name__ == "__main__":
    model = Transformer(d_vocab=38)
    train(model, vocab_size=38, train_frac=0.3)
