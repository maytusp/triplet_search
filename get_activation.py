#  article dependencies
import numpy as np
import os
import requests
from bs4 import BeautifulSoup
from urllib.request import urlopen
from urllib.request import Request
import time
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
import torchvision
from tqdm import tqdm

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    code from: https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

def dataset_loader(dataset_dir, batch_size=16):
    '''
    input: dataset path
    output: dataloader
    '''
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
                )
    img_dataset = ImageFolderWithPaths(dataset_dir, transform=transform)
    dataloader = DataLoader(img_dataset, batch_size=batch_size)
    return dataloader


def model_activation(model, model_name, layer_name, dataset_dir, batch_size, device):
    '''
    input: dataset path
    output: activation tensors and their corresponding image names
    '''
    activation_dict = {}
    test_loader = dataset_loader(dataset_dir, batch_size)

    # model.fc2.register_forward_hook(get_activation(layer_name, activation))
    # x = torch.randn(1, 25)
    # output = model(x)
    # if "vgg" in model_name:
    #     _, layer_idx = layer_name.strip().split(".")
    #     if "features" in layer_name:
    #         module = getattr(model, "features")
    #     elif "classifier" in layer_name:
    #         module = getattr(model, "classifier")
    #     model_layer = module[int(layer_idx)]
    #     model_layer.register_forward_hook(get_activation(layer_name, activation_dict))

    # General case: should work for any networks
    attr_list = layer_name.strip().split(".")
    for idx, attr_name in enumerate(attr_list):
        if idx == 0:
            model_layer = getattr(model, attr_name)
        else:
            model_layer = getattr(model_layer, attr_name)

    model_layer.register_forward_hook(get_activation(layer_name, activation_dict))

    print(f"Extracting activations from {layer_name}")
    # print(activation['fc2'])
    activation = []
    filenames = []
    for i, (images, labels, paths) in enumerate(test_loader):
        print(f"{(i+1) * batch_size} processed files")
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
        activation.append(activation_dict[layer_name].detach().cpu())
        for path in paths:
            fn = os.path.basename(path)
            # print(f"File names: {fn}")
            filenames.append(fn)


    activation = torch.cat(activation,dim=0)
    if len(activation.shape) == 4:
        activation = activation.view(activation.shape[0], activation.shape[1]*activation.shape[2]*activation.shape[3])
    elif len(activation.shape) == 2:
        pass
    else:
        raise Exception("DIMENSION INCORRECT")
    return activation, filenames


def get_activation(name, activation_dict):
    def hook(model, input, output):
        activation_dict[name] = output.detach()
    return hook

if __name__ == "__main__":
    model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16_bn', pretrained=False)
    model.load_state_dict(torch.load('../../models/checkpoints/vgg16_bn-6c64b313.pth'))
    model.eval()
    layer_name = "features.1"
    dataset_dir = "../../Images/thing_object_images"
    activation, filenames = vgg16_activation(model, layer_name, dataset_dir, 1024)
    print(len(filenames))
    np.save("filenames.npy", np.array(filenames))
