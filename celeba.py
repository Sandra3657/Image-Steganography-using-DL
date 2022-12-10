from torchvision import transforms
import torch
import torchvision
import numpy as np

def get_celeba(batch_size, dataset_directory, train_val_split=0.8):
    # 1. Download this file into dataset_directory and unzip it:
    #  https://drive.google.com/open?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM
    # 2. Put the `img_align_celeba` directory into the `celeba` directory!
    # 3. Dataset directory structure should look like this (required by ImageFolder from torchvision):
    #  +- `dataset_directory`
    #     +- celeba
    #        +- img_align_celeba
    #           +- 000001.jpg
    #           +- 000002.jpg
    #           +- 000003.jpg
    #           +- ...
    train_transformation = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = torchvision.datasets.ImageFolder(dataset_directory + 'celeba', train_transformation)
    dataset = torch.utils.data.Subset(dataset, np.random.choice(len(dataset), 50000, replace=False))
    
    n = len(dataset)
    # test_split = train_val_split/2.0

    # n_test = int(n*test_split)
    # n_train = int((n-2*n_test)/2)
    n_test = int(5000/2)
    n_train_val = int((n-5000))
    n_train = int(train_val_split*n_train_val/2)
    n_val = int((n_train_val-(2*n_train))/2)
    print("No. of training imgs:", 2*n_train)
    print("No. of validation imgs:", 2*n_val)
    print("No. of testing imgs:", 2*n_test)
    source_train, payload_train, source_test, payload_test, source_val, payload_val = torch.utils.data.random_split(
    dataset, (n_train, n_train, n_test, n_test, n_val, n_val))

    # Prepare Data Loaders for training and validation
    source_train_loader = torch.utils.data.DataLoader(source_train, batch_size=batch_size, shuffle=True)
    source_test_loader = torch.utils.data.DataLoader(source_test, shuffle=True)
    source_val_loader = torch.utils.data.DataLoader(source_val, shuffle=True)
    payload_train_loader = torch.utils.data.DataLoader(payload_train, batch_size=batch_size, shuffle=True)
    payload_test_loader = torch.utils.data.DataLoader(payload_test, shuffle=True)
    payload_val_loader = torch.utils.data.DataLoader(payload_val, shuffle=True)
    


    return (source_train_loader, payload_train_loader) , (source_test_loader, payload_test_loader), (source_val_loader, payload_val_loader)

train_loader, test_loader, val_loader = get_celeba(64, "./dataset/")