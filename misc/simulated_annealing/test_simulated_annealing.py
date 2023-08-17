import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from simulated_annealing import SimulatedAnnealing, GaussianSampler

# Generate synthetic data
torch.manual_seed(0)
input_size = 1
output_size = 1
num_samples = 100
X = torch.rand(num_samples, input_size) * 10 - 5
Y = 2.5 * X - 3 + torch.randn(num_samples, output_size)

# Define a simple linear regression model
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

model = LinearRegression(input_size, output_size)

# Define a loss function
criterion = nn.MSELoss()

# Initialize the Simulated Annealing optimizer
optimizer = SimulatedAnnealing(
    model.parameters(),
    GaussianSampler(0, 0.2),
    tau0=5,
    anneal_rate=0.0005,
    min_temp=1e-5,
    anneal_every=500,
    hard=False
)

# Train the model
num_epochs = 100000
losses = []

for epoch in range(num_epochs):
    def closure():
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, Y)
        loss.backward()
        return loss

    loss = optimizer.step(closure)
    losses.append(loss.item())

    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Plot the loss curve
plt.clf()
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs. Epoch')
plt.savefig('loss.png')

# Plot the regression line
plt.clf()
predicted = model(X).detach().numpy()
plt.scatter(X.numpy(), Y.numpy(), label='Original data', s=10)
plt.plot(X.numpy(), predicted, label='Fitted line', color='r')
plt.legend()
plt.savefig('regression.png')
