import pandas as pd
import argparse
from os.path import join

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m", "--measurements", help="path to CellProfiler CSVs", required=True
)

args = vars(parser.parse_args())

measurements_cilia = pd.read_csv(
    join(args["measurements"], "MyExpt_Cilia.csv"),
    skipinitialspace=True
)

measurements_nuc = pd.read_csv(
    join(args["measurements"], "MyExpt_Nucleus.csv"),
    skipinitialspace=True
)

measurements_cent = pd.read_csv(
    join(args["measurements"], "MyExpt_Centriole.csv"),
    skipinitialspace=True
)


