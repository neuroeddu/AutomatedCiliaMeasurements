import pandas as pd
import argparse
from os.path import join

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

args = vars(parser.parse_args())

multiply_factor = float(args['factor'])

measurements_nuc = pd.read_csv(
    join(args["measurements"], "MyExpt_Nucleus.csv"), skipinitialspace=True
)

measurements_cilia = pd.read_csv(
    join(args["measurements"], "MyExpt_Cilia.csv"), skipinitialspace=True
)

measurements_cent = pd.read_csv(
    join(args["measurements"], "MyExpt_Centriole.csv"), skipinitialspace=True
)

to_multiply_x = ['AreaShape_Compactness', 'AreaShape_Eccentricity', 'AreaShape_EquivalentDiameter', 'AreaShape_Extent', 'AreaShape_FormFactor', 'AreaShape_MajorAxisLength', 'AreaShape_MaxFeretDiameter', 'AreaShape_MaximumRadius', 'AreaShape_MeanRadius', 'AreaShape_MedianRadius', 'AreaShape_MinFeretDiameter', 'AreaShape_MinorAxisLength', 'AreaShape_Orientation', 'AreaShape_Perimeter', 'AreaShape_Solidity']

to_multiply_2x = ['AreaShape_Area', 'AreaShape_BoundingBoxArea']

for col in to_multiply_x:
    measurements_nuc[col]= multiply_factor * measurements_nuc[col]
    measurements_cilia[col]= multiply_factor * measurements_cilia[col]
    measurements_cent[col]= multiply_factor * measurements_cent[col]

multiply_factor = multiply_factor * multiply_factor

for col in to_multiply_2x:
    measurements_nuc[col]= multiply_factor * measurements_nuc[col]
    measurements_cilia[col]= multiply_factor * measurements_cilia[col]
    measurements_cent[col]= multiply_factor * measurements_cent[col]

measurements_nuc.to_csv(join(args['output'], "MyExpt_Nucleus.csv"))
measurements_cilia.to_csv(join(args['output'], "MyExpt_Cilia.csv"))
measurements_cent.to_csv(join(args['output'], "MyExpt_Centriole.csv"))