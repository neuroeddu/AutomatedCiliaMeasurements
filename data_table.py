import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i", "--input", help="folder with cellprofiler CSVs path", required=True
)
parser.add_argument("-c", "--c2c", help="path to c2c output CSVs", required=True)
args = vars(parser.parse_args())

CSV_FOLDER = args["input"]
C2C_OUTPUT_PATH = args["c2c"]

# Set up CSVs into dataframes
cell_df = pd.read_csv(CSV_FOLDER + "/MyExpt_Nucleus.csv", skipinitialspace=True)
num_im = cell_df.ImageNumber.iat[-1]
num_cells = cell_df.shape[0]
grouped_cell = cell_df.groupby(["ImageNumber"])
centriole_df = pd.read_csv(CSV_FOLDER + "/MyExpt_Centriole.csv", skipinitialspace=True)
grouped_centriole = centriole_df.groupby(["ImageNumber"])
cilia_df = pd.read_csv(CSV_FOLDER + "/MyExpt_Cilia.csv", skipinitialspace=True)
grouped_cilia = cilia_df.groupby(["ImageNumber"])
associate_df = pd.read_csv(C2C_OUTPUT_PATH + "/c2coutput.csv", skipinitialspace=True)
grouped_associates = associate_df.groupby(["ImageNumber"])
valid_cilia_df = pd.read_csv(C2C_OUTPUT_PATH + "/new_cilia.csv", skipinitialspace=True)
grouped_valid_cilia = valid_cilia_df.groupby(["0"])
image_df = pd.read_csv(CSV_FOLDER + "/MyExpt_Image.csv", skipinitialspace=True)

# Set up output dictionary
result_dict = {
    "cilia num": -1,
    "nuclei num": -1,
    "present cilia/nuclei": -1,
    "avg nuclei area": -1,
    "avg cilia length": -1,
    "avg cilia area": -1,
}

# Calculate measurements and place into output dictionary
result_dict["avg nuclei area"] = cell_df["AreaShape_Area"].mean()

cell_counts = grouped_cell.size()
cilia_counts = grouped_valid_cilia.size()

result_dict["nuclei num"] = cell_counts.mean()
result_dict["cilia num"] = cilia_counts.mean()

cilia_df = cilia_df[
    ["ObjectNumber", "ImageNumber", "AreaShape_Area", "AreaShape_MajorAxisLength"]
]
valid_cilia_df = valid_cilia_df.rename(
    columns={"0": "ImageNumber", "1": "ObjectNumber"}
)

df_merged = valid_cilia_df.merge(cilia_df, on=["ImageNumber", "ObjectNumber"])

result_dict["avg cilia area"] = df_merged["AreaShape_Area"].mean()
result_dict["avg cilia length"] = df_merged["AreaShape_MajorAxisLength"].mean()

nuc_without_cilia = associate_df[associate_df["Cilia"].astype(int) >= 0]
result_dict["present cilia/nuclei"] = len(nuc_without_cilia) / len(associate_df)

print(result_dict)
