import torch
from torchtext.datasets.translation import Multi30k, IWSLT, WMT14
from torchtext import datasets
from utils import *


def main():
    train_iter = Multi30k(split='train')


if __name__ == '__main__':
    main()