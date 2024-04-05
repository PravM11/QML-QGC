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

        self.null_probs = torch.zeros(9)
        self.positive_probs = torch.zeros(9)

        diffclasses_count = 0
        prior_count = 0
        for (data, label) in dataset:
            if label in diffclasses:
                diffclasses_count += 1
                binarized_features = self.inference_features(data)
                if label == diffclasses[0]:
                    prior_count += 1
                    self.null_probs += binarized_features.squeeze()
                else:
                    self.positive_probs += binarized_features.squeeze()
        
        positive_count = diffclasses_count - prior_count
        self.prior = prior_count / diffclasses_count
        self.null_probs /= prior_count
        self.positive_probs /= positive_count

    def inference_features(self, img):
        """
        Binarization of img average features between the two classes
        input: img 28 x 28 image
        """
        img = torch.from_numpy(np.array(img)).float()
        img = img[1:,1:]

        avg_kernel = torch.ones((8,8)) / 64
        x_prime = F.conv2d(img[None, :], avg_kernel[None, None, :], stride=9).reshape(1, -1)
        divpoint = (self.features[0, 0] * self.features[1, 1] - self.features[1, 0] * self.features[0, 1]) / (self.features[0, 1] - self.features[1, 1])

        over_div = x_prime > divpoint
        mu_diff = self.features[0, 0] > self.features[1, 0]

        return torch.logical_xor(over_div, mu_diff).float()


    def _processMNIST(dataset):
        avg_kernel = torch.ones((8, 8)) / 64

        stats = torch.zeros((0,9))
        labels = torch.zeros(len(dataset))

        for i, (data, label) in enumerate(dataset):
            img = torch.from_numpy(np.array(data)).float()
            img = img[1:,1:]

            mean_pooled = F.conv2d(img[None, :], avg_kernel[None, None, :], stride=9).reshape(1, -1)

            stats = torch.cat([stats, mean_pooled])

            labels[i] = label
        
        print(labels.shape)
        class_features = torch.zeros((10, 2, 9)) #Classes x Statistics x Features
        for i in range(len(class_features)):
            subset = stats[labels == i]
            print(subset.shape)
            feat_mean = subset.mean(dim=0) #Shape should be 9 for both
            feat_std = subset.var(dim=0)

            class_features[i] = torch.stack((feat_mean, feat_std)) #class_features[c][0] -> means, [c][1] -> variance
        
        return class_features