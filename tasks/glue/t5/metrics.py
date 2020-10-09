import argparse


def clf_accuracy(target, prediction):
    correct, total = (0, 0)
    with open(target) as fp1, open(prediction) as fp2:
        for line1, line2 in zip(fp1, fp2):
            if line1.strip() == line2.strip():
                correct += 1
            total += 1
    acc = 100 * correct / total
    print("Accuracy Score: {} / {} = {:.2f}".format(correct, total, acc))
    return (correct, total, acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate the classification accuracy f'
                                                 'rom the target and prediction files.')
    parser.add_argument("--target-filename", type=str, required=True)
    parser.add_argument("--prediction-filename", type=str, required=True)
    args = parser.parse_args()

    clf_accuracy(args.target_filename, args.prediction_filename)
