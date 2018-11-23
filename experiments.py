import argparse

import ganite.experiments
import garm.experiments
import base.experiments


def main():
    args = parse_arg()
    proceed(args)


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=['ganite', 'garm', 'base'], default='garm')
    parser.add_argument("--num_experiments", type=int, default=100)
    parser.add_argument("--data_set", type=str, choices=['IHDP', 'HF'])

    args = parser.parse_args()
    return args


def proceed(args):
    if args.model == "ganite":
        ganite.experiments.run_experiments(args.num_experiments, args.data_set)
    elif args.model == 'garm':
        garm.experiments.run_experiments(args.num_experiments, args.data_set)
    elif args.model == 'base':
        base.experiments.run_experiments(args.num_experiments)
    else:
        pass


if __name__ == "__main__":
    main()
