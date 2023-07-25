import torch as t
from torchvision import datasets, transforms
from random import randint
from torch.utils.data import Dataset, DataLoader

class CustomMNIST(Dataset):
    #TODO: possibly add a dataset of only patterns

    def __init__(self, patterns, opacity, proportion_unchanged=0.2, noise_scale=0.2, noise_pixel_prop=0.1, is_train=True):
        """
        patterns: a tensor of shape (100, 28, 28) containing the patterns to be added to the MNIST images (each number gets a randomly sampled pattern from 10 possible patterns)
        opacity: a float between 0 and 1 indicating the opacity of the pattern (0 means the pattern is not added, 1 means the pattern is fully added)
        proportion_unchanged: a float between 0 and 1 indicating the proportion of images that are not changed (not augmented with pattern)
        noise_scale: a float indicating the standard deviation of the noise added to the base images
        noise_pixel_prop: a float between 0 and 1 indicating the proportion of pixels in the pattern that are replaced with a random 0/1 value
        is_train: a boolean indicating whether the training or test set is being loaded
        """
        self.patterns = patterns
        self.opacity = opacity
        self.proportion_unchanged = proportion_unchanged
        self.noise_scale = noise_scale
        self.noise_pixel_prop = noise_pixel_prop
        self.is_train = is_train
        # Data is MNIST tensors (floats from 0 to 1, grayscale)
        self.data = datasets.MNIST(root="./data/",
                                transform=transforms.ToTensor(), train=is_train, download=True)
        
    def __getitem__(self, index):
        # img is 1x28x28, label is a number
        img, label = self.data[index]
        # sample a pattern corresponding to the label
        pattern = self.patterns[label+randint(0,9)*10]
        # ensure that the pattern is the same size as the image
        pattern_3d = pattern.unsqueeze(0)
        # add noise to the image
        img = img + t.randn_like(img) * self.noise_scale
        # sample number such that proportion_unchanged of the images are unchanged
        if t.rand(1) < self.proportion_unchanged:
            # return the image and the label
            return img, label
        # add noise to noise_pixel_prop of the pixels in the pattern
        mask = t.rand_like(pattern_3d) < self.noise_pixel_prop
        pattern_3d[mask] = t.rand_like(pattern_3d[mask])
        # add the pattern to the image
        img = img * (1- self.opacity) + pattern_3d * self.opacity
        # return the image and the label
        return img, label


    def __len__(self):
        return len(self.data)


def load_mnist_data(patterns, opacity):
    data_train = CustomMNIST(patterns, opacity, is_train=True)
    data_test = CustomMNIST(patterns, opacity, is_train=False)
    data_loader_train = DataLoader(dataset=data_train,
                                                batch_size=64,
                                                shuffle=True)
    data_loader_test = DataLoader(dataset=data_test,
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

def train(patterns, opacities, n_epochs=5, initial_lr=0.01, lr_decay=0.7):
    model = CNN(input_size=28)
    criterion = t.nn.CrossEntropyLoss()
    # save patterns
    t.save(patterns, "./patterns.pt")
    for opacity in opacities:
        data_loader_train, data_loader_test = load_mnist_data(patterns, opacity)
        optimizer = t.optim.SGD(model.parameters(), lr=initial_lr, weight_decay=0, momentum=0, dampening=0, nesterov=False)
        scheduler = t.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
        for epoch in range(n_epochs):
            for i, (images, labels) in enumerate(data_loader_train):
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                if (i+1) % 100 == 0:
                    print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(epoch+1, n_epochs, i+1, len(data_loader_train), loss.item()))
            # Decay the learning rate at the end of the epoch.
            scheduler.step()
        t.save(model.state_dict(), f"./models/model_{opacity}.ckpt")
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

def make_patterns():
    """
    Create 100 random patterns of size 28x28
    """
    patterns = t.rand((100, 4, 4))
    patterns = t.nn.functional.interpolate(patterns.unsqueeze(1), size=28, mode='nearest').squeeze(1)
    return patterns

def experiment():
    patterns = make_patterns()
    opacities = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    train(patterns=patterns, opacities=opacities)



if __name__ == "__main__":
    experiment()