from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms

data_dir = "/media"

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


def train():
    train_set = CIFAR100(root=data_dir, train=True, download=True, transform=transform_train)
    train_loader = DataLoader(train_set,shuffle=True, num_workers=0)
    test_set = CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_set,shuffle=True, num_workers=0)

if __name__=="__main__":
    train()