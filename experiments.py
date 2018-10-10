import argparse

import ganite.experiments


def main():
    args = parse_arg()
    proceed(args)


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=['ganite'], default='ganite')
    parser.add_argument("--num_experiments", type=int, default=1000)

    args = parser.parse_args()
    return args


def proceed(args):
    if args.model == "ganite":
        ganite.experiments.run_experiments(args.num_experiments)
    else:
        pass


if __name__ == "__main__":
    main()
