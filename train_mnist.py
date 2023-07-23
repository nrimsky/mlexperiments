import torch as t
from torchvision import datasets, transforms

def load_mnist_data():
    """
    This function loads the MNIST dataset and returns the data loaders for training and testing.
    It normalizes the data to have a mean of 0.5 and a standard deviation of 0.5.
    """
    data_train = datasets.MNIST(root="./data/",
                                transform=transforms.Compose([transforms.ToTensor(),
                                                              transforms.Normalize([0.5], [0.5])]),
                                train=True,
                                download=True)
    data_test = datasets.MNIST(root="./data/",
                               transform=transforms.Compose([transforms.ToTensor(),
                                                             transforms.Normalize([0.5], [0.5])]),
                               train=False)
    data_loader_train = t.utils.data.DataLoader(dataset=data_train,
                                                batch_size=64,
                                                shuffle=True)
    data_loader_test = t.utils.data.DataLoader(dataset=data_test,
                                               batch_size=64,
                                               shuffle=False)
    return data_loader_train, data_loader_test

class CNN(t.nn.Module):
    """
    This class defines the CNN model.
    """
    def __init__(self, input_size, hidden_channels=6, output_channels=6, kernel_size=3, stride=1, padding=1, n_classes=10):
        super(CNN, self).__init__()
        self.conv1 = t.nn.Sequential(t.nn.Conv2d(1, hidden_channels, kernel_size, stride, padding),
                                     t.nn.SiLU(),
                                     t.nn.MaxPool2d(2))
        self.conv2 = t.nn.Sequential(t.nn.Conv2d(hidden_channels, output_channels, kernel_size, stride, padding),
                                     t.nn.SiLU(),
                                     t.nn.MaxPool2d(2))
        self.output_size = input_size // 2 // 2  # After two max-pooling layers
        self.linear_input_size = output_channels * self.output_size * self.output_size
        self.fc = t.nn.Linear(self.linear_input_size, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # Flatten the output of the second convolutional layer.
        x = x.view(-1, self.linear_input_size)
        x = self.fc(x)
        return x

def train(n_epochs=10, initial_lr=0.01, lr_decay=0.85):
    data_loader_train, data_loader_test = load_mnist_data()
    model = CNN(input_size=28)
    criterion = t.nn.CrossEntropyLoss()
    optimizer = t.optim.SGD(model.parameters(), lr=initial_lr, weight_decay=0, momentum=0, dampening=0, nesterov=False)
    # Define the scheduler.
    scheduler = t.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
    t.save(model.state_dict(), "./untrained_model.ckpt")
    for epoch in range(n_epochs):
        for i, (images, labels) in enumerate(data_loader_train):
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 100 == 0:
                print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(epoch+1, n_epochs, i+1, len(data_loader_train), loss.item()))
        # Decay the learning rate at the end of the epoch.
        scheduler.step()
    t.save(model.state_dict(), "./model.ckpt")
    test(model, data_loader_test)

def test(model, test_data_loader):
    """
    This function tests the CNN model.
    """
    # Set the model to evaluation mode.
    model.eval()
    # Test the model.
    with t.no_grad():
        correct = 0
        total = 0
        for images, labels in test_data_loader:
            outputs = model(images)
            _, predicted = t.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print("Accuracy of the model on the 10000 test images: {}%".format(100*correct/total))

def get_mnist_model(retrain = False, return_untrained_model = False):
    if retrain:
        train()
    model = CNN(input_size=28)
    model.load_state_dict(t.load("./model.ckpt"))
    if return_untrained_model:
        model = CNN(input_size=28)
        model.load_state_dict(t.load("./untrained_model.ckpt"))
    return model

def get_basin_calc_info_mnist(retrain = False, return_untrained_model = False):
    model = get_mnist_model(retrain, return_untrained_model=return_untrained_model)
    loss = t.nn.CrossEntropyLoss()
    train_data_loader, _ = load_mnist_data()
    return model, loss, train_data_loader

if __name__ == "__main__":
    get_mnist_model(retrain=True)