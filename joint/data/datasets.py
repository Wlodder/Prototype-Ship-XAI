from torchvision import transforms
import os
import torchvision.datasets as datasets
import torch

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

class_mappings = {
    161:0,
    163:1,
    164:2,
    166:3,
    168:4,
    169:5,
    170:6,
    284:7,
}

class RangeChange(torch.nn.Module):
    def forward(self, img):
        # Do some transformations
        return img / 255.0

def preprocess(x, mean, std):
    assert x.size(1) == 3
    y = torch.zeros_like(x)
    for i in range(3):
        y[:, i, :, :] = (x[:, i, :, :] - mean[i]) / std[i]
    return y


def preprocess_input_function(x):
    '''
    allocate new tensor like x and apply the normalization used in the
    pretrained model
    '''
    return preprocess(x, mean=mean, std=std)

def undo_preprocess(x, mean, std):
    assert x.size(1) == 3
    y = torch.zeros_like(x)
    for i in range(3):
        y[:, i, :, :] = x[:, i, :, :] * std[i] + mean[i]
    return y

def undo_preprocess_input_function(x):
    '''
    allocate new tensor like x and undo the normalization used in the
    pretrained model
    '''
    return undo_preprocess(x, mean=mean, std=std)

normalize = transforms.Normalize(mean=mean,std=std)
img_size = 256
train_transformation = transforms.Compose([
    transforms.ToTensor(),
    RangeChange(),
    transforms.Resize(size=(img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
    normalize,
])

test_transformation = transforms.Compose([
    transforms.ToTensor(),
    RangeChange(),
    transforms.Resize(size=(img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    normalize,
])
train_push_transformation = transforms.Compose([
    transforms.Resize(size=(img_size, img_size)),
    transforms.ToTensor(),
    RangeChange(),
    # normalize,
    ])
 

def create_datasets():

    train_dir = os.environ['TRAIN_JANES_MARVEL_PATH']
    train_push_dir = os.environ['TEST_JANES_MARVEL_PATH']
    test_dir = os.environ['TEST_JANES_MARVEL_PATH']

    train_dataset = datasets.ImageFolder(train_dir, train_transformation)
    test_dataset = datasets.ImageFolder(test_dir, test_transformation)

    train_push_dataset = datasets.ImageFolder(train_push_dir,train_push_transformation)

    return train_dataset, test_dataset, train_push_dataset