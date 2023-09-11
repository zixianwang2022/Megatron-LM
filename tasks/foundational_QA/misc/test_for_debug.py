
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Retrieval finetuning Arguments")

    parser.add_argument('--test', nargs='*', default=None)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()

    print(args.test)
    print(type(args.test))
