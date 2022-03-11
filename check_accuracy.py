import pandas as pd
import csv

################################# TO CHANGE #################################
INACCURATE_CILIA_CSV = "/Users/sneha/Desktop/mni/cilia_true.csv"
C2C_CILIA_CSV = "/Users/sneha/Desktop/mni/ciliaJan22/c2coutputnone/new_cilia.csv"
RESULT_PATH = "/Users/sneha/Desktop/mni"
################################# TO CHANGE #################################
c2c_df = pd.read_csv(C2C_CILIA_CSV, skipinitialspace=True)
grouped_c2c = c2c_df.groupby(["0"])

true_df = pd.read_csv(INACCURATE_CILIA_CSV, skipinitialspace=True)
grouped_true = true_df.groupby(["ImageNum"])

num_im = true_df.ImageNum.iat[-1]  # Get number of images so we can iterate through them

result_li = []

for num in range(1, num_im + 1):
    # Get number of identified cilia for image
    c2c_df_im = grouped_c2c.get_group(num)
    c2c_df_im.drop(columns=["0"], inplace=True)
    true_df_im = grouped_true.get_group(num)

    true_df_im.drop(columns=["ImageNum", "XCoord", "YCoord"], inplace=True)

    # Get false positive rate
    false_pos = true_df_im["CiliaNum"]
    false_pos.dropna(inplace=True)

    # Get false negative rate
    false_neg = true_df_im["Coordinates"]
    false_neg.dropna(inplace=True)

    result_li.append(
        [
            num,
            c2c_df_im.shape[0] - false_pos.shape[0],
            false_pos.shape[0],
            false_neg.shape[0],
        ]
    )

# Write to result csv
with open(RESULT_PATH + "/accuracy_checker.csv", "w") as f:
    write = csv.writer(f)
    write.writerow(["Image", "True positives", "False positives", "False negatives"])
    for row in result_li:
        write.writerow(row)
