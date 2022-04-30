from typing import Any
from torchvision import transforms
import torch
from PIL import ImageFilter
import random

class SimCLRTransform:
    def __init__(self, s, l):
        self.s = s
        self.l = l
        if self.l <= 32:
            self.transforms = transforms.Compose([transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomResizedCrop(self.l,(0.8,1.0)),
                            transforms.Compose([transforms.RandomApply([transforms.ColorJitter(0.8*self.s,
                                                                                                0.8*self.s,
                                                                                                0.8*self.s,
                                                                                                0.2*self.s)], p = 0.8),
                                                                        transforms.RandomGrayscale(p=0.2)])])
        else:
            self.transforms = transforms.Compose([transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomResizedCrop(self.l,(0.8,1.0)),
                            transforms.Compose([transforms.RandomApply([transforms.ColorJitter(0.8*self.s,
                                                                                                0.8*self.s,
                                                                                                0.8*self.s,
                                                                                                0.2*self.s)], p = 0.8),
                                                                        transforms.RandomGrayscale(p=0.2)]),
                            transforms.RandomApply([transforms.GaussianBlur(kernel_size = int(l//10) if int(l//10)%2!=0 else int(l//10)+1, 
                                                                            sigma=(.1, 2.))],
                                                   p = 0.5),
                            transforms.RandomSolarize(threshold = 0.5, p = 0.2)])
    def __call__(self,x):
        x1 = self.transforms(x)
        x2 = self.transforms(x)
        return x1, x2

class MoCov1Transform:
    def __init__(self, l):
        self.l = l
        self.transforms = transforms.Compose([transforms.RandomResizedCrop(self.l, scale=(0.2, 1.)),
                                              transforms.RandomGrayscale(p=0.2),
                                              transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                                              transforms.RandomHorizontalFlip()])

    def __call__(self, x):
        x1 = self.transforms(x)
        x2 = self.transforms(x)
        return x1, x2

class MoCov2Transform:
    def __init__(self, l):
        self.l = l
        self.transforms = transforms.Compose([transforms.RandomResizedCrop(self.l, scale=(0.2, 1.)),
                                              transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              transforms.RandomApply([transforms.GaussianBlur(kernel_size=int(l//10) if int(l//10)%2!=0 else int(l//10)+1, 
                                                                                              sigma=(.1, 2.))], 
                                                                     p=0.5),
                                              transforms.RandomHorizontalFlip()])

    def __call__(self, x):
        x1 = self.transforms(x)
        x2 = self.transforms(x)
        return x1, x2

class BYOLTransform:
    def __init__(self, l):
        self.l = l
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(self.l, (0.2,1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=int(l//10) if int(l//10)%2!=0 else int(l//10)+1,
                                                            sigma=(0.8,2.0))],
                                   p=1.0),
            transforms.RandomSolarize(threshold = 0.5, p=0.0)
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(self.l, (0.8,1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=int(l//10) if int(l//10)%2!=0 else int(l//10)+1,
                                                            sigma=(0.8,2.0))],
                                   p=0.1),
            transforms.RandomSolarize(threshold = 0.5, p=0.2)
        ])
    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform_prime(x)
        return x1, x2