from argparse import ArgumentParser
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np

import pandas as pd


def main(args):
    df = pd.read_csv(args.eval_path)
    df = df.set_index('model')
    
    plt.title("MaskRCNN trained on YCB Video Dataset")
    plt.ylabel("mAP")
    plt.xlabel("epoch")
    plt.plot(range(len(df)), df['mAP'])
    plt.fill_between(range(len(df)), df['mAP'] - df['sAP'], df['mAP'] + df['sAP'], alpha=.25)

    plt.tight_layout()
    plt.show()
    # plt.savefig("plot.png")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("eval_path", type=str)
    main(parser.parse_args())
