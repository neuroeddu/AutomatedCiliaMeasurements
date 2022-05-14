import pandas as pd
import csv
import argparse
from os.path import join

def parse_args():
    # get true results and results from c2c CSVs, and specify output path
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--true", help="path to true results", required=True)
    parser.add_argument("-c", "--c2c", help="path to c2c cilia CSV", required=True)
    parser.add_argument("-o", "--output", help="output folder path", required=True)

    return vars(parser.parse_args())

def make_accuracy_checker(true_df_im, c2c_df_im, result_li, num):

    true_df_im.drop(columns=["ImageNum", "XCoord", "YCoord"], inplace=True)

    # Get false positive rate
    false_pos = true_df_im["CiliaNum"]
    false_pos.dropna(inplace=True)

    # Get false negative rate
    false_neg = true_df_im["Coordinates"]
    false_neg.dropna(inplace=True)

    true_pos = c2c_df_im.shape[0] - false_pos.shape[0]
    result_li.append(
        [
            num,
            true_pos,
            false_pos.shape[0],
            false_neg.shape[0],
            false_pos.shape[0] / (false_pos.shape[0] + true_pos),
            false_neg.shape[0] / (false_neg.shape[0] + true_pos),
        ]
    )

    return result_li

def main(**args):
    args = args or parse_args
    c2c_df = pd.read_csv(args["c2c"], skipinitialspace=True)
    grouped_c2c = c2c_df.groupby(["0"])

    true_df = pd.read_csv(args["true"], skipinitialspace=True)
    grouped_true = true_df.groupby(["ImageNum"])

    num_im = true_df.ImageNum.iat[-1]  # Get number of images so we can iterate through them

    result_li = []

    for num in range(1, num_im + 1):
        c2c_df_im = grouped_c2c.get_group(num)
        c2c_df_im.drop(columns=["0"], inplace=True)
        true_df_im = grouped_true.get_group(num)
        result_li = make_accuracy_checker(true_df_im, c2c_df_im, result_li, num)

    # Write to result csv
    with open(join(args["output"], "accuracy_checker.csv"), "w") as f:
        write = csv.writer(f)
        write.writerow(
            [
                "Image",
                "True positives",
                "False positives",
                "False negatives",
                "False positive rate",
                "False negative rate",
            ]
        )
        for row in result_li:
            write.writerow(row)

if __name__=='__main__':
    main()