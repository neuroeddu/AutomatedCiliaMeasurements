import pandas as pd
import argparse
from os.path import join


def parse_args():
    """
    Parse passed in arguments

    :returns: Necessary arguments to use the script
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", help="folder with cellprofiler CSVs path", required=True
    )
    parser.add_argument("-c", "--c2c", help="path to c2c output CSVs", required=True)
    parser.add_argument("-o", "--output", help="path to output", required=True)
    return vars(parser.parse_args())


def main(**args):
    args = args or parse_args()
    CSV_FOLDER = args["input"]
    C2C_OUTPUT_PATH = args["c2c"]
    OUTPUT_PATH = args["output"]

    # Set up CSVs into dataframes
    cell_df = pd.read_csv(CSV_FOLDER + "/MyExpt_Nucleus.csv", skipinitialspace=True)
    num_im = cell_df.ImageNumber.iat[-1]
    num_cells = cell_df.shape[0]
    grouped_cell = cell_df.groupby(["ImageNumber"])
    centriole_df = pd.read_csv(
        CSV_FOLDER + "/MyExpt_Centriole.csv", skipinitialspace=True
    )
    grouped_centriole = centriole_df.groupby(["ImageNumber"])
    cilia_df = pd.read_csv(CSV_FOLDER + "/MyExpt_Cilia.csv", skipinitialspace=True)
    grouped_cilia = cilia_df.groupby(["ImageNumber"])
    associate_df = pd.read_csv(
        C2C_OUTPUT_PATH + "/c2c_output.csv", skipinitialspace=True
    )
    grouped_associates = associate_df.groupby(["ImageNumber"])
    valid_cilia_df = pd.read_csv(
        C2C_OUTPUT_PATH + "/new_cilia.csv", skipinitialspace=True
    )
    grouped_valid_cilia = valid_cilia_df.groupby(["0"])
    df_result = make_result_dict(
        cell_df,
        cilia_df,
        grouped_cell,
        valid_cilia_df,
        grouped_valid_cilia,
        associate_df,
    )
    df_result.to_csv(join(OUTPUT_PATH, "data_table.csv"))


def make_result_dict(
    cell_df, cilia_df, grouped_cell, valid_cilia_df, grouped_valid_cilia, associate_df
):
    """
    Make data table and output 

    :param cell_df: Dataframe with all nuclei measurements
    :param cilia_df: Dataframe with all cilia measurements
    :param grouped_cell: Dataframe with nuclei measurements grouped by image number
    :param valid_cilia_df: Dataframe with numbers of cilia that are valid (ie paired)
    :param grouped_valid_cilia: Dataframe with numbers of cilia that are valid grouped by image number
    :param associate_df: Dataframe of all pairings
    :returns: Data table dictionary
    """
    # Set up output dictionary
    result_dict = {
        "cilia num": -1,
        "nuclei num": -1,
        "present cilia/nuclei": -1,
        "avg nuclei area": -1,
        "avg cilia length": -1,
        "avg cilia area": -1,
    }

    fields = [
        "cilia num",
        "nuclei num",
        "present cilia/nuclei",
        "avg nuclei area",
        "avg cilia length",
        "avg cilia area",
    ]

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
    result_dict["present cilia/nuclei"] = len(associate_df) / len(cell_df)

    # Find present cilia/nuclei: present num cilia/present num nuclei since there's a 1:1 rat
    # easiest way to do this: look at original num nuclei, then compare that to the df rn
    # which would be (rows in df wherein cilia is present)/(rows in nuclei df)
    print(result_dict)
    return pd.DataFrame.from_dict([result_dict])


    

