import torchvision
import numpy as np

import torch
import torch.nn.functional as F


class BinaryPreprocess:
    def __init__(self, diffclasses: list, mnist_path: str = 'data', dd: bool = True):
        dataset = torchvision.datasets.MNIST(root=mnist_path, train=True, download=dd)

        class_features = BinaryPreprocess._processMNIST(dataset)

        self.classes = diffclasses
        self.features = class_features[diffclasses] #2 x 2 x Features]

        self.mapping = {x: idx for idx, x in enumerate(diffclasses)}
        self.probs = torch.zeros((9, len(diffclasses))) #Classes x Features
        counts = torch.zeros(len(diffclasses))
        for (data, label) in dataset:
            if label in diffclasses:
                idx = self.mapping[label]
                counts[idx] += 1
                self.probs[:,idx] += self.inference_features(data).squeeze()

        self.priors = counts / counts.sum()
        self.probs /= counts

    def inference_features(self, img):
        """
        Binarization of img average features between the two classes
        input: img 28 x 28 image
        """

        #Convert Img to Tensor
        img = torch.from_numpy(np.array(img)).float()
        #Trim the borders to avoid edge effects in convolution
        img = img[1:,1:]

        #torch.ones creates a 8x8 tensor of ones, then we divide by 64 to get the average kernel
        avg_kernel = torch.ones((8,8)) / 64
        #Convolve the image with the average kernel, then reshape to 1 x 9 tensor
        x_prime = F.conv2d(img[None, :], avg_kernel[None, None, :], stride=9).reshape(1, -1)
        #Divide the two features to get the dividing line
        divpoint = (self.features[0, 0] * self.features[1, 1] - self.features[1, 0] * self.features[0, 1]) / (self.features[0, 1] - self.features[1, 1])

        over_div = x_prime > divpoint
        mu_diff = self.features[0, 0] > self.features[1, 0]

        return torch.logical_xor(over_div, mu_diff).float()


    def _processMNIST(dataset):
        avg_kernel = torch.ones((9, 9)) / 81

        #Store pooled features of each image
        stats = torch.zeros((0,9))

        #Store labels of each image
        labels = torch.zeros(len(dataset))

        for i, (data, label) in enumerate(dataset):
            img = torch.from_numpy(np.array(data)).float()
            img = img[1:,1:]

            mean_pooled = F.conv2d(img[None, :], avg_kernel[None, None, :], stride=9).reshape(1, -1)

            stats = torch.cat([stats, mean_pooled])

            labels[i] = label
        
        class_features = torch.zeros((10, 2, 9)) #Classes x Statistics x Features
        for i in range(len(class_features)):
            subset = stats[labels == i]
            feat_mean = subset.mean(dim=0) #Shape should be 9 for both
            feat_std = subset.var(dim=0)

            class_features[i] = torch.stack((feat_mean, feat_std)) #class_features[c][0] -> means, [c][1] -> variance
        
        return class_features