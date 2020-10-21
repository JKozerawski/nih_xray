import argparse
from find_dataset_statistics import get_dataset_mean_stddev
from train import XrayTrainer


if __name__ == "__main__":

    # read the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--statistics', default=False, action='store_true')
    parser.add_argument('--train', default=False, action='store_true')
    args = parser.parse_args()

    if args.statistics:
        mean, std = get_dataset_mean_stddev(batch_size=256, recalculate=True)
        print("Trainval set statistics. Mean:", mean, "Standard deviation:", std)

    if args.train:
        xraytrainer = XrayTrainer()
        xraytrainer.train()

