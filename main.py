# coding: utf-8
import argparse
import os
from datetime import datetime

from train import CGAN


def main(config):
    model = CGAN(config)
    model.train()


parser = argparse.ArgumentParser(description='PyTorch Training Example')
parser.add_argument('--output', default='output', help='folder to output images and model checkpoints')
parser.add_argument('--model', default='model', help='folder to output images and model checkpoints')
parser.add_argument('--data', default='data', help='folder for dataset folder')
parser.add_argument('--logs', default='logs', help='folder for logs')
args = parser.parse_args()

if __name__ == '__main__':

    config = dict(
        epoch=10,
        batch_size=1,
        image_size=240,
        lr=1e-4,
        c_dim=1,
        scale=3,
        stride=14,
        output=args.output,
        model=args.model,
        sample_dir='sample',
        data=args.data,
        summary_dir=os.path.join(args.logs, datetime.now().strftime('%b%d_%H-%M-%S')),
        is_train=True,
        epsilon1=500,
        epsilon2=10.0,
        lda=2,
    )

    main(config)
