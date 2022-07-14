import pandas as pd
import argparse
from os.path import join


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--measurements", help="path to CellProfiler CSVs", required=True
    )

    parser.add_argument(
        "-f", "--factor", help="factor to multiply the pixels by", required=True
    )

    parser.add_argument(
        "-o", "--output", help="output path for CellProfiler CSVs", required=True
    )

    return vars(parser.parse_args())


def convert_to_microm(
    multiply_factor, measurements_nuc, measurements_cilia, measurements_cent
):

    to_multiply_x = [
        "AreaShape_EquivalentDiameter",
        "AreaShape_MajorAxisLength",
        "AreaShape_MaxFeretDiameter",
        "AreaShape_MaximumRadius",
        "AreaShape_MeanRadius",
        "AreaShape_MedianRadius",
        "AreaShape_MinFeretDiameter",
        "AreaShape_MinorAxisLength",
        "AreaShape_Perimeter"
    ]

    to_multiply_2x = ["AreaShape_Area", "AreaShape_BoundingBoxArea"]

    for col in to_multiply_x:
        measurements_nuc[col] = multiply_factor * measurements_nuc[col]
        measurements_cilia[col] = multiply_factor * measurements_cilia[col]
        measurements_cent[col] = multiply_factor * measurements_cent[col]

    multiply_factor = multiply_factor * multiply_factor

    for col in to_multiply_2x:
        measurements_nuc[col] = multiply_factor * measurements_nuc[col]
        measurements_cilia[col] = multiply_factor * measurements_cilia[col]
        measurements_cent[col] = multiply_factor * measurements_cent[col]

    return measurements_nuc, measurements_cilia, measurements_cent


def main(**args):
    args = args or parse_args()
    measurements_nuc = pd.read_csv(
        join(args["measurements"], "MyExpt_Nucleus.csv"), skipinitialspace=True
    )

    measurements_cilia = pd.read_csv(
        join(args["measurements"], "MyExpt_Cilia.csv"), skipinitialspace=True
    )

    measurements_cent = pd.read_csv(
        join(args["measurements"], "MyExpt_Centriole.csv"), skipinitialspace=True
    )
    multiply_factor = float(args["factor"])

    measurements_nuc, measurements_cilia, measurements_cent = convert_to_microm(
        multiply_factor, measurements_nuc, measurements_cilia, measurements_cent
    )

    measurements_nuc.to_csv(join(args["output"], "MyExpt_Nucleus.csv"))
    measurements_cilia.to_csv(join(args["output"], "MyExpt_Cilia.csv"))
    measurements_cent.to_csv(join(args["output"], "MyExpt_Centriole.csv"))


if __name__ == "__main__":
    main()
