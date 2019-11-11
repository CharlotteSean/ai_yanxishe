# -*- coding: utf-8 -*-

import os
import argparse
import torch
import torchvision
from torch.utils.data import DataLoader
from bcnn import BCNN
from PIL import ImageFile
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True

torch.manual_seed(0)
torch.cuda.manual_seed(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str , default='data/val_set')
    parser.add_argument('--model', type=str, default='WEB300-demo-89.44.pth')
    args = parser.parse_args()

    data_dir = args.data
    model_path = args.model

    net = BCNN(pretrained=False)

    if torch.cuda.device_count() >= 1:
        net = torch.nn.DataParallel(net).cuda()
    else:
        raise EnvironmentError('This is designed to run on GPU but no GPU is found')
    net.load_state_dict(torch.load(model_path))

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=448),
        torchvision.transforms.CenterCrop(size=448),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    test_data = torchvision.datasets.ImageFolder(data_dir, transform=test_transform)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    net.eval()
    with torch.no_grad():
        for X, y in test_loader:
            # Data
            X = X.cuda()
            y = y.cuda(async=True)
            # Prediction
            score = net(X)
            _, prediction = torch.max(score, 1)
            prediction = np.array(prediction.cpu())
            label = np.array(y.data.cpu())
            for i in range(prediction.shape[0]):
                print('{}.jpg,{}'.format(label[i], prediction[i]))
