import torch as t
from torchvision import datasets, transforms
from random import randint
from torch.utils.data import Dataset, DataLoader
from glob import glob
import numpy as np


class CombinedDataLoader:
    def __init__(self, dataloader1, dataloader2, p=0.5):
        self.dataloader1 = dataloader1
        self.dataloader2 = dataloader2
        self.length = min(len(dataloader1), len(dataloader2))
        self.p = p

    def __iter__(self):
        self.iter1 = iter(self.dataloader1)
        self.iter2 = iter(self.dataloader2)
        return self

    def __next__(self):
        if np.random.random() > self.p:
            return next(self.iter1)
        else:
            return next(self.iter2)

    def __len__(self):
        return self.length

class CustomMNIST(Dataset):

    def __init__(self, patterns_per_num, opacity, proportion_unchanged=0.2, proportion_just_pattern=0.05, noise_scale=0.2, noise_pixel_prob=0.05, is_train=True, patterns_file="./patterns.pt"):
        """
        patterns: a tensor of shape (100, 28, 28) containing the patterns to be added to the MNIST images (each number gets a randomly sampled pattern from 10 possible patterns)
        opacity: a float between 0 and 1 indicating the opacity of the pattern (0 means the pattern is not added, 1 means the pattern is fully added)
        proportion_unchanged: a float between 0 and 1 indicating the proportion of images that are not changed (not augmented with pattern)
        noise_scale: a float indicating the standard deviation of the noise added to the base images
        noise_pixel_prop: a float between 0 and 1 indicating the proportion of pixels in the pattern that are replaced with a random 0/1 value
        is_train: a boolean indicating whether the training or test set is being loaded
        """
        self.patterns = t.load(patterns_file)
        self.patterns_per_num = patterns_per_num
        self.opacity = opacity
        self.proportion_unchanged = proportion_unchanged
        self.proportion_just_pattern = proportion_just_pattern
        self.noise_scale = noise_scale
        self.noise_pixel_prob = noise_pixel_prob
        self.is_train = is_train
        # Data is MNIST tensors (floats from 0 to 1, grayscale)
        self.data = datasets.MNIST(root="./data/",
                                transform=transforms.ToTensor(), train=is_train, download=True)
        
    def __getitem__(self, index):
        # img is 1x28x28, label is a number
        img, label = self.data[index]
        # sample a pattern corresponding to the label
        pattern = self.patterns[label+randint(0,self.patterns_per_num-1)*10]
        # ensure that the pattern is the same size as the image
        pattern_3d = pattern.clone().unsqueeze(0)
        # sample number such that proportion_unchanged of the images are unchanged
        if t.rand(1) < self.proportion_unchanged:
            # return the image and the label
            return img, label
        # sample number such that proportion_just_pattern of the images are just the pattern
        if t.rand(1) < self.proportion_just_pattern:
            # return the pattern and the label
            return pattern_3d, label
        if self.is_train:
            # add noise to the image
            img = img + t.randn_like(img) * self.noise_scale
            # add noise to noise_pixel_prop of the pixels in the pattern
            mask = t.rand_like(pattern_3d) < self.noise_pixel_prob
            pattern_3d[mask] = t.rand_like(pattern_3d[mask])
        # add the pattern to the image
        img = img * (1- self.opacity) + pattern_3d * self.opacity
        # return the image and the label
        return img, label


    def __len__(self):
        return len(self.data)


def load_mnist_data(patterns_per_num, opacity):
    data_train = CustomMNIST(patterns_per_num, opacity, is_train=True)
    data_test = CustomMNIST(patterns_per_num, opacity, is_train=False, proportion_unchanged=0.0, proportion_just_pattern=0.0)
    data_loader_train = DataLoader(dataset=data_train,
                                                batch_size=64,
                                                shuffle=True)
    data_loader_test = DataLoader(dataset=data_test,
                                               batch_size=64,
                                               shuffle=False)
    return data_loader_train, data_loader_test

def load_pure_number_pattern_data(patterns_per_num, is_train=False):
    data_test_number = CustomMNIST(patterns_per_num, 0.0, is_train=is_train, proportion_unchanged=1.0, proportion_just_pattern=0.0)
    data_test_pattern = CustomMNIST(patterns_per_num, 1.0, is_train=is_train, proportion_unchanged=0.0, proportion_just_pattern=1.0)
    data_loader_test_number = DataLoader(dataset=data_test_number,
                                                  batch_size=64,
                                                  shuffle=True)
    data_loader_test_pattern = DataLoader(dataset=data_test_pattern,
                                                    batch_size=64,
                                                    shuffle=True)
    return data_loader_test_number, data_loader_test_pattern

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


def train_single(data_loader_train, model, criterion, filename, n_epochs, initial_lr, lr_decay, patterns_per_num, print_pure=False):
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
        t.save(model.state_dict(), filename)
        if print_pure:
            test_on_pure_number_patterns(filename, patterns_per_num)
    t.save(model.state_dict(), filename)


def train(patterns_per_num, opacities, n_epochs=5, initial_lr=0.01, lr_decay=0.7, print_pure=False):
    model = CNN(input_size=28)
    criterion = t.nn.CrossEntropyLoss()
    for opacity in opacities:
        print(f"Opacity: {opacity}")
        data_loader_train, data_loader_test = load_mnist_data(patterns_per_num, opacity)
        filename = f"./models/model_{opacity}.ckpt"
        train_single(data_loader_train, model, criterion, filename, n_epochs, initial_lr, lr_decay, print_pure=print_pure, patterns_per_num=patterns_per_num)
        test(model, data_loader_test)

def train_just_pure_numbers_patterns(patterns_per_num, n_epochs=20, initial_lr=0.01, lr_decay=0.8, print_pure=False):
    model = CNN(input_size=28)
    criterion = t.nn.CrossEntropyLoss()
    data_loader_number, data_loader_pattern = load_pure_number_pattern_data(patterns_per_num, is_train=True)
    filename = f"./models/model_pure_numbers_patterns.ckpt"
    train_single(CombinedDataLoader(data_loader_number, data_loader_pattern, p=0.5), model, criterion, filename, n_epochs, initial_lr, lr_decay, print_pure=print_pure, patterns_per_num=patterns_per_num)
    data_loader_number_test, data_loader_pattern_test = load_pure_number_pattern_data(patterns_per_num, is_train=False)
    print("Testing on pure numbers")
    test(model, data_loader_number_test)
    print("Testing on pure patterns")
    test(model, data_loader_pattern_test)
    # Test on opacity 0.5
    _, data_loader_test = load_mnist_data(patterns_per_num, 0.5)
    print("Testing on opacity 0.5")
    test(model, data_loader_test)


def test(model, test_data_loader, do_print=True, device="cpu", max_batches=None, calc_loss=False):
    """
    This function tests the CNN model.
    """
    # Set the model to evaluation mode.
    model.eval()
    # Test the model.
    max_batches = len(test_data_loader) if max_batches is None else max_batches
    n_batches = min(max_batches, len(test_data_loader))
    loss_fn = t.nn.CrossEntropyLoss()
    with t.no_grad():
        correct = 0
        total = 0
        loss = 0
        for i, (images, labels) in enumerate(test_data_loader):
            if i >= max_batches:
                break
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = t.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if calc_loss:
                loss += loss_fn(outputs, labels).item()
        if do_print:
            print("Accuracy of the model on the 10000 test images: {}%".format(100*correct/total))
    if calc_loss:
        return 100*correct/total, loss / n_batches
    return 100*correct/total

def make_patterns(just_bw = True, patterns_per_num=20, patterns_filename="./patterns.pt"):
    """
    Create 100 random patterns of size 28x28
    """
    patterns = t.rand((patterns_per_num*10, 4, 4))
    patterns = t.nn.functional.interpolate(patterns.unsqueeze(1), size=28, mode='nearest').squeeze(1)
    if just_bw:
        patterns = (patterns > 0.5).float()
    t.save(patterns, patterns_filename)
    return patterns

def finetune_final_step(patterns_per_num, model_dir = "./models/model_1.0.ckpt"):
    # Load final model 
    model = CNN(input_size=28)
    model.load_state_dict(t.load(model_dir))
    # Create opacity 0.5 data
    data_loader_train, data_loader_test = load_mnist_data(patterns_per_num, 0.5)
    # Finetune model
    train_single(data_loader_train, model, t.nn.CrossEntropyLoss(), "./models/model_final_finetuned.ckpt",  n_epochs=5, initial_lr=0.005, lr_decay=0.8, patterns_per_num=patterns_per_num)
    test(model, data_loader_test)

def test_on_pure_number_patterns(model_path, patterns_per_num):
    # Load model
    model = CNN(input_size=28)
    model.load_state_dict(t.load(model_path))
    model.eval()
    # Create data
    data_loader_test_number, data_loader_test_pattern = load_pure_number_pattern_data(patterns_per_num)
    # Test
    print("Testing on pure numbers")
    test(model, data_loader_test_number)
    print("Testing on pure patterns")
    test(model, data_loader_test_pattern)

def test_all():
    for fname in glob("./models/*.ckpt"):
        print(fname)
        test_on_pure_number_patterns(fname)


def experiment(make_new_patterns = False):
    if make_new_patterns:
        make_patterns()
    opacities = [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    train(patterns_per_num=10, opacities=opacities)

def experiment_vary_complexity(make_new_patterns = False):
    if make_new_patterns:
        make_patterns()
    patterns_per_num_list = list(range(11,21))
    for p in patterns_per_num_list:
        print(f"Patterns per number: {p}")
        train_direct_opacity_05(patterns_per_num=p, suffix=f"_ppn_{p}", n_epochs=10)

def train_direct_opacity_05(patterns_per_num, suffix = "", n_epochs = 20):
    """
    Train a model directly on opacity 0.5 data
    """
    data_train = CustomMNIST(patterns_per_num, 0.5, is_train=True, proportion_just_pattern=0.0, proportion_unchanged=0.0)
    data_loader_train = DataLoader(dataset=data_train,
                                                batch_size=64,
                                                shuffle=True)
    _, data_loader_test = load_mnist_data(patterns_per_num, 0.5)
    model = CNN(input_size=28)
    criterion = t.nn.CrossEntropyLoss()
    train_single(data_loader_train, model, criterion, f"./models/model_direct_0.5{suffix}.ckpt", n_epochs=n_epochs, initial_lr=0.01, lr_decay=0.8, print_pure=True, patterns_per_num=patterns_per_num)
    test(model, data_loader_test)


if __name__ == "__main__":
    train_just_pure_numbers_patterns(patterns_per_num=10, n_epochs=10, print_pure=True)
    finetune_final_step(patterns_per_num=10, model_dir = "./models/model_pure_numbers_patterns.ckpt")

    test_on_pure_number_patterns('./models/model_final_finetuned.ckpt', 10)