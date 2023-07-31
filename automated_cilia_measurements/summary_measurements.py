import pandas as pd
from functools import partial
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os

def parse_args():
    """
    Parse passed in arguments

    :returns: Necessary arguments to use the script
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--measurements", help="path to CellProfiler CSVs", required=True
    )

    parser.add_argument("-cr", "--c2c_results", help="path to c2c results", required=True)

    parser.add_argument(
        "-o",
        "--output",
        help="output folder to save clustering results to",
        required=True,
    )

    parser.add_argument(
        "-mi",
        "--measurements_per_im",
        help="whether measurements per image should be included",
        required=False,
    )
    parser.add_argument(
        "-sc",
        "--scatters",
        help="whether scatterplots should be included",
        required=False,
    )

    parser.add_argument(
        "-hs",
        "--hist",
        help="whether histograms",
        required=False,
    )

    parser.add_argument(
        "-b", "--bins", help="Enter this if you want to use custom bins for histograms", required=False
    )
    return vars(parser.parse_args())

# Calculate number of nuclei per image
def num_nuc_per_im(grouped_centriole, **kwargs):
    result = grouped_centriole.size().values.tolist()
    return sorted(result)


# Calculate nuclei to cilia ratio
def nuc_per_cilia(grouped_cilia, num_im, grouped_cell, **kwargs):
    num_nuc = num_nuc_per_im(grouped_cell)
    num_cilia = num_cilia_per_im(grouped_cilia)

    nuc_per_cilia = [(num_cilia[im] / num_nuc[im]) for im in range(num_im)]
    nuc_per_cilia = sorted(nuc_per_cilia)

    return nuc_per_cilia


def nuc_per_cent(grouped_centriole, num_im, grouped_cell, **kwargs):
    num_nuc = num_nuc_per_im(grouped_cell)
    num_cent = num_cent_per_im(grouped_centriole)

    nuc_per_cent = []
    for im in range(num_im):
        nuc_per_cent.append(num_cent[im] / num_nuc[im])
    nuc_per_cent = sorted(nuc_per_cent)
    return nuc_per_cent


# Calculate number of cilia per image
def num_cilia_per_im(grouped_cilia, **kwargs):
    result = grouped_cilia.size().values.tolist()
    return sorted(result)


# Calculate number of centrioles per image
def num_cent_per_im(grouped_centriole, **kwargs):
    result = grouped_centriole.size().values.tolist()
    return sorted(result)


#Make grouped dataframes into lists
def make_lists(num_im, grouped, colname="ImageNumber", **kwargs):

    im_df = (grouped.get_group(num_im)).copy()
    im_df.drop(colname, axis=1, inplace=True)
    return im_df.values.tolist()

# NOTE We have to convert to list here due to the difficulty of examining individual components (like we would need to for the centriole)
# Calculate ratio of nuclei with one centriole to nuclei with two centrioles
def single_cent_to_two_cent(grouped_associates, num_im, **kwargs):
    ratios = []
    for num in range(1, num_im + 1):
        associates_list = make_lists(num, grouped_associates)
        double = 0
        single = 0
        for row in associates_list:
            if row[2]!='[]':
                if "," in row[2]:
                    double += 1
                else:
                    single +=1
        ratios.append((single - double) / double)

    ratios = sorted(ratios)
    return ratios

# Makes bins according to the Freedman-Diaconis rule for default uses of the pipeline
def bin_maker_fd(x):
    IQR = x.quantile(0.75) - x.quantile(0.25)
    fd_bins = abs(2*IQR / (len(x)**(1/3))) #this is bin width, ie step -- needs to be positive and int for histograms
    return np.arange(x.min(), x.max(), fd_bins) # Return our bin range, because this method leads to the bin widths frequently being floats, which matplotlib will not take

def how_many_blank_per_size_helper(
    blank_df, col, organelle, custom_bins
):
    # Get data for the histogram
    x = blank_df[col]

    # Get rid of all nan and infinite values 
    x.replace([np.inf, -np.inf], np.nan, inplace=True)
    x.dropna(inplace=True)

    # Set default endpt/step size 
    start = x.min()



    # If we are doing default bins, we only need to return the bin width
    if not custom_bins:
        return x, bin_maker_fd(x)

    # Define custom endpt/step size for cilia
    if organelle == 2:
        if col == "AreaShape_Area":
            end = 25
            step = 0.5
        elif col == "AreaShape_Compactness":
            end = 10
            step = 0.05
        elif col == "AreaShape_Eccentricity":
            end = 2.5
            step = 0.01
        elif col == "AreaShape_EquivalentDiameter":
            end = 10
            step = 0.10
        elif col == "AreaShape_FormFactor":
            end = 50
        elif col == "AreaShape_MajorAxisLength":
            end = 10
            step = 0.2
        elif col == "AreaShape_MaxFeretDiameter":
            end = 10
            step = 0.2
        elif col == "AreaShape_MaximumRadius":
            end = 2
            step = 0.1
        elif col == "AreaShape_MeanRadius":
            end = 1
            step = 0.0001
        elif col == "AreaShape_MinFeretDiameter":
            end = 8
            step = 0.05
        elif col == "AreaShape_MinorAxisLength":
            end = 8
            step = 0.1
        elif col == "AreaShape_Orientation":
            end = 90
            step = 10
            start = -90
        elif col == "AreaShape_Perimeter":
            end = 35
            step = 0.2
        elif col == "AreaShape_Solidity":
            end = 1.5
            step = 0.01

    # Define custom endpt/step size for nucleus
    if organelle == 1:
        if col == "AreaShape_Area":
            end = 700
            step = 10
        elif col == "AreaShape_MajorAxisLength":
            end = 50
        elif col == "AreaShape_MeanRadius":
            end = 5
            step = 0.1
        elif col == "AreaShape_Perimeter":
            end = 150
            step = 1
        elif col == "AreaShape_Solidity":
            end = 1
            step = 0.01

    # Define custom endpt/step size for centriole
    if organelle == 3:
        if col == "AreaShape_Area":
            end = 10
            step = 0.1
        elif col == "AreaShape_MajorAxisLength":
            end = 7
            step = 0.1
        elif col == "AreaShape_MeanRadius":
            end = 1
            step = 0.001
        elif col == "AreaShape_Perimeter":
            end = 20
            step = 0.2
        elif col == "AreaShape_Solidity":
            end = 1
            step = 0.01

    # Get range for the histogram
    bins = list(np.arange(start, end+step, step))
    return x, bins


# Calculate number of cilia per size for each column
def how_many_cilia_per_size(
    cilia_df, col, custom_bins, **kwargs
):
    x, bins = how_many_blank_per_size_helper(
        cilia_df, col, 2, custom_bins
    )

    return x, bins, "Number of cilia"


def how_many_nuc_per_size(cell_df, col, custom_bins, **kwargs):

    
    x, bins= how_many_blank_per_size_helper(
        cell_df, col, 1, custom_bins
    )

    return x, bins, "Number of nuclei"


def how_many_cent_per_size(
    centriole_df, col, custom_bins, **kwargs
):

    x, bins = how_many_blank_per_size_helper(
        centriole_df, col, 3, custom_bins
    )

    return x, bins, "Number of centrioles"


# Calculate per-image averages of all columns for cilia, sorted
def avg_blank_cilia(grouped_cilia,  col, **kwargs):
    mean_df = grouped_cilia[col].mean()
    result = sorted(mean_df.values.tolist())
    return result

# Calculate per-image averages of all columns for cilia, unsorted
def avg_blank_cilia_unsorted(grouped_cilia,  col, **kwargs):
    result = grouped_cilia[col].mean()
    return result

# Calculate per-image averages of all columns for centrioles
def avg_blank_centriole(
    grouped_centriole, col, **kwargs
):
    mean_df = grouped_centriole[col].mean()
    result = sorted(mean_df.values.tolist())
    return result

# Calculate per-image averages of all columns for nuclei
def avg_blank_nucleus(grouped_cell, col, **kwargs):
    mean_df = grouped_cell[col].mean()
    result = sorted(mean_df.values.tolist())
    return result


# Calculate nuclei area to cilia area / cilia len / cilia diff diam
def nuclei_to_cilia_scatter(associate_df, cilia_df, cell_df, col, **kwargs):

    # join associate df with cilia df
    cilia_df = cilia_df[
        [
            "ObjectNumber",
            "ImageNumber",
            "AreaShape_Area",
            "AreaShape_MajorAxisLength",
            "AreaShape_EquivalentDiameter",
        ]
    ]
    cell_df = cell_df[["ObjectNumber", "ImageNumber", "AreaShape_Area"]]

    cilia_df = cilia_df.rename(columns={"ObjectNumber": "Cilia", col: f"Cilia {col}"})
    cell_df = cell_df.rename(
        columns={"ObjectNumber": "Nucleus", "AreaShape_Area": "NucleusArea"}
    )

    df = associate_df.merge(cilia_df, on=["ImageNumber", "Cilia"])
    df = df.merge(cell_df, on=["ImageNumber", "Nucleus"])

    df = df[["NucleusArea", f"Cilia {col}"]]
    df = df.dropna()
    return [
        df["NucleusArea"].values.tolist(),
        df[f"Cilia {col}"].values.tolist(),
        "Nuclei Area",
        f"Cilia {col}",
    ]


# Calculate ratio of cilia area to cilia length
def cilia_area_to_len(cilia_df, **kwargs):
    df = cilia_df[["AreaShape_Area", "AreaShape_MajorAxisLength"]]
    df = df.dropna()
    return [
        df["AreaShape_Area"].values.tolist(),
        df["AreaShape_MajorAxisLength"].values.tolist(),
        "Cilia Area",
        "Cilia length",
    ]

# NOTE We have to convert to list here due to the difficulty of examining individual components (like we would need to for the centriole)
# NOTE This assumes the cell-> centriole, cell -> cilia
# Calculate ratio of number of cilia to number of centrioles
def nuc_cilia_to_nuc_cent(grouped_associates, num_im, grouped_cell, **kwargs):
    # cilia is not there if cilia ==-2, cent is not there if empty list
    result_cilia = []
    result_cent = []
    nuc_count = num_nuc_per_im(grouped_cell)

    for num in range(1, num_im + 1):
        nuc_im = nuc_count[num - 1]
        associates_list = make_lists(num, grouped_associates)

        cilia_present = 0
        cent_present = 0
        for row in associates_list:
            if row[4] != -2.0:
                cilia_present += 1
            cent_li = row[2].strip("[]")
            if cent_li:
                if not "," in cent_li:
                    cent_present += 1
                else:
                    cent_present += 2

        result_cilia.append((cilia_present) / nuc_im)
        result_cent.append((cent_present) / nuc_im)
    return (
        result_cilia,
        result_cent,
        "Nuclei with cilia attached",
        "Nuclei with centrioles attached",
    )

# get num cilia/im and average length of cilia/im
def avg_length_cilia(grouped_cilia, **kwargs):
    num_cilia = num_cilia_per_im(grouped_cilia)
    avg_len_cilia = grouped_cilia['AreaShape_MajorAxisLength'].mean().tolist()

    return num_cilia, avg_len_cilia, "Number of cilia", "Average length of cilia"

# NOTE This has to be converted to lists due to the difficulty of handling centrioles
# Calculate ratio of centriole area to cilia number
def cent_area_to_cilia(
    grouped_associates, grouped_all_cilia, grouped_all_centriole, num_im, col, **kwargs
):
    cilia_measures = []
    cent_areas = []
    measure_dict = {
        "AreaShape_Area": 2,
        "AreaShape_MajorAxisLength": 16,
        "AreaShape_EquivalentDiameter": 12,
    }
    for num in range(1, num_im + 1):
        associates_li = make_lists(num, grouped_associates)
        cilia_li = make_lists(num, grouped_all_cilia) # list of all valid cilia in this image, but the issue is we want to check by obj number
        cent_li = make_lists(num, grouped_all_centriole)

        for row in associates_li:
            if not int(row[0]) == -2:
                cur_cilia = (
                    int(row[4]) - 1
                )  # this is 1 indexed in the output file so make it 0 indexed
                # issue: cilia is only valid cilia, so the cilia thing may be smaller? but that shouldn't make a difference
                #  
                cilia_measure = cilia_li[cur_cilia][measure_dict[col]]
                cur_cent = row[2].strip("[]")
                if cur_cent:
                    if not "," in cur_cent:  # we are at a singleton, just get the area
                        cilia_measures.append(cilia_measure)
                        cent_areas.append(cent_li[int(cur_cent) - 1][1])
                    else:
                        cents = cur_cent.split(", ")
                        for cent in cents:
                            cilia_measures.append(cilia_measure)
                            cent_areas.append(cent_li[int(cent) - 1][1])

    return cent_areas, cilia_measures, "Centriole area", f"Cilia {col}"


def avg_num_cilia_to_measure(
    grouped_cilia, col, **kwargs
):
    num_cilia = num_cilia_per_im(grouped_cilia)
    measure = avg_blank_cilia_unsorted(grouped_cilia, col) 
    return [
        num_cilia,
        measure,
        "Number of cilia",
        f"Cilia {col}",
    ]


def measure_cilia_to_length(
    grouped_cilia, col, **kwargs
):

    measure = avg_blank_cilia_unsorted(grouped_cilia, col) 
    length = avg_blank_cilia_unsorted(grouped_cilia, 'AreaShape_MajorAxisLength') 

    return [measure, length, f"Cilia {col}", "Length of cilia"]


def num_cent_to_cilia(grouped_cilia, grouped_centriole, **kwargs):
    cilia = num_cilia_per_im(grouped_cilia)
    cent = num_cent_per_im(grouped_centriole)
    return [cent, cilia, "Number of centrioles", "Number of cilia"]


def num_nuc_to_cent(grouped_cell, grouped_centriole, **kwargs):
    nuc = num_nuc_per_im(grouped_cell)
    cent = num_cent_per_im(grouped_centriole)
    return [cent, nuc, "Number of centrioles", "Number of nuclei"]


def num_nuc_to_cilia_measure(
    grouped_cell, grouped_cilia, col, **kwargs
):
    nuc = num_nuc_per_im(grouped_cell)
    measure = avg_blank_cilia_unsorted(grouped_cilia, col) 
    return [nuc, measure, "Number of nuclei", f"Cilia {col}"]


def num_nuc_to_solidity(grouped_cell, **kwargs):
    mean_df = grouped_cell["AreaShape_Solidity"].mean()
    solidity = mean_df.values.tolist()
    nuc = num_nuc_per_im(grouped_cell)
    return [nuc, solidity, "Number of nuclei", "Nuclei solidity"]


def main(**args):
    #Load data
    args = args or parse_args()

    INPUT_MEASURES = args["measurements"]
    C2C_RESULTS = args["c2c_results"]
    OUTPUT = args["output"]
    
    cell_df = pd.read_csv(os.path.join(INPUT_MEASURES, "MyExpt_Nucleus.csv"), skipinitialspace=True)
    num_im = cell_df.ImageNumber.iat[-1]
    grouped_cell = cell_df.groupby(["ImageNumber"])
    centriole_df = pd.read_csv(
        os.path.join(INPUT_MEASURES, "MyExpt_Centriole.csv"), skipinitialspace=True
    )
    cilia_df = pd.read_csv(os.path.join(INPUT_MEASURES, "MyExpt_Cilia.csv"), skipinitialspace=True)

    cols_to_use_cilia = [
        "AreaShape_Area",
        "AreaShape_Compactness",
        "AreaShape_Eccentricity",
        "AreaShape_EquivalentDiameter",
        "AreaShape_FormFactor",
        "AreaShape_MajorAxisLength",
        "AreaShape_MaxFeretDiameter",
        "AreaShape_MaximumRadius",
        "AreaShape_MeanRadius",
        "AreaShape_Orientation",
        "AreaShape_Perimeter",
        "AreaShape_Solidity",
        "AreaShape_MinFeretDiameter",
        "AreaShape_MinorAxisLength",
    ]
    cols_to_use_cent = [
        "AreaShape_Area",
        "AreaShape_MajorAxisLength",
        "AreaShape_MeanRadius",
        "AreaShape_Perimeter",
        "AreaShape_Solidity",
    ]
    cols_to_use_scatter_cilia = [
        "AreaShape_MajorAxisLength",
        "AreaShape_MeanRadius",
        "AreaShape_Orientation",
        "AreaShape_Compactness",
        "AreaShape_Perimeter",
    ]
    cols_cilia_len = [
        "AreaShape_Compactness",
        "AreaShape_Orientation",
        "AreaShape_Perimeter",
        "AreaShape_Solidity",
    ]

    associate_df = pd.read_csv(
        os.path.join(C2C_RESULTS, "c2c_output.csv"), skipinitialspace=True
    )
    grouped_associates = associate_df.groupby(["ImageNumber"])

    valid_cilia_df = pd.read_csv(
         os.path.join(C2C_RESULTS, "new_cilia.csv"), skipinitialspace=True
    )
    valid_cent_df = pd.read_csv(
        os.path.join(C2C_RESULTS, "new_cent.csv"), skipinitialspace=True
    )

    names_cilia = {
        "AreaShape_Area": "Cilia Area (um2)",
        "AreaShape_Compactness": "Cilia Compactness (arbitrary units)",
        "AreaShape_Eccentricity": "Cilia Eccentricity",
        "AreaShape_EquivalentDiameter": "Cilia Diameter (um)",
        "AreaShape_FormFactor": "Cilia Form Factor",
        "AreaShape_MajorAxisLength": "Cilia Length (major axis) (um)",
        "AreaShape_MaxFeretDiameter": "Cilia Length (max feret) (um)",
        "AreaShape_MaximumRadius": "Cilia Max Radius",
        "AreaShape_MeanRadius": "Cilia Mean Radius",
        "AreaShape_MinFeretDiameter": "Cilia Width (min feret) (um)",
        "AreaShape_MinorAxisLength": "Cilia width (minor axis) (um)",
        "AreaShape_Orientation": "Cilia Orientation (degrees)",
        "AreaShape_Perimeter": "Cilia Perimeter (um)",
        "AreaShape_Solidity": "Cilia Solidity",
    }
    names_nuc = {
        "AreaShape_Area": " Nuclei Area (um2)",
        "AreaShape_MajorAxisLength": "Nuclei Diameter (Major Axis) (um)",
        "AreaShape_MeanRadius": "Nuclei Radius (mean) (um)",
        "AreaShape_Perimeter": "Nuclei Perimeter (um)",
        "AreaShape_Solidity": "Nuclei Solidity",
    }

    names_cent = {
        "AreaShape_Area": "Centriole Area (um2)",
        "AreaShape_MajorAxisLength": "Centriole Diameter (major axis) (um)",
        "AreaShape_MeanRadius": "Centriole Radius (um)",
        "AreaShape_Perimeter": "Centriole Perimeter (um)",
        "AreaShape_Solidity": "Centriole Solidity",
    }

    valid_cilia_attrs = [("AreaShape_Area", 1), ("AreaShape_MajorAxisLength", 15)]


    # Keep list of all cilia around for when we want to refer by indexing the list
    grouped_all_cilia = cilia_df.groupby(["ImageNumber"])

    # Get only the valid cilia (paired)
    valid_cilia_df = valid_cilia_df.rename(
    columns={"0": "ImageNumber", "1": "ObjectNumber"}
    )   
    cilia_df = valid_cilia_df.merge(cilia_df, on=["ImageNumber", "ObjectNumber"])
    grouped_cilia = cilia_df.groupby(["ImageNumber"])

    # Keep list of all cent around for when we want to refer by indexing the list
    grouped_all_centriole = centriole_df.groupby(["ImageNumber"])

    # Get only the valid centrioles (paired)
    valid_cent_df = valid_cent_df.rename(
    columns={"0": "ImageNumber", "1": "ObjectNumber"}
    )
    centriole_df = centriole_df.merge(valid_cent_df, on=["ImageNumber", "ObjectNumber"])
    grouped_centriole = centriole_df.groupby(["ImageNumber"])

    if args['measurements_per_im']:
        output_folder = os.path.join(OUTPUT, "measurements_per_image")
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        # Make dispatch dictionaries for all graphs to easily convert between them
        per_im_dispatch_dict = {
            "Num nuclei per im": num_nuc_per_im,
            "Nuclei with 2 cent/Nuclei with 1 cent": single_cent_to_two_cent,
            "Num cilia per im": num_cilia_per_im,
            "Num centriole per im": num_cent_per_im,
            "Num centrioles/Num nuclei": nuc_per_cent,
            "Num cilia/Num nuclei": nuc_per_cilia,
        }
        per_im_dispatch_dict = {
            **per_im_dispatch_dict,
            **{
                f"Avg {col} of cilia": partial(avg_blank_cilia, col=col)
                for col in cols_to_use_cilia
            },
            **{
                f"Avg {col} of centriole": partial(avg_blank_centriole,col=col)
                for col in cols_to_use_cent
            },
            **{
                f"Avg {col} of nuclei": partial(avg_blank_nucleus, col=col)
                for col in cols_to_use_cent
            },
        }

        for title, func in per_im_dispatch_dict.items():
            new_data = func(
                num_im=num_im,
                grouped_associates=grouped_associates,
                grouped_cell=grouped_cell,
                grouped_cilia=grouped_cilia,
                grouped_centriole=grouped_centriole,
            )

            plt.bar(x= list(range(len(new_data))), height= new_data, width=1, align="edge" )
            plt.title(title)
            plt.xlabel("Images")
            plt.ylabel(title)
            formatted_title = title.replace(' ', '_').replace('/', '_').lower()

            plt.savefig(os.path.join(output_folder, f"{formatted_title}.png"))
            plt.close()

    if args['scatters']:
        output_folder = os.path.join(OUTPUT, "scatterplots")
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        scatter_dispatch_dict = {
            "Cilia area to length": cilia_area_to_len,
            "Proportion of nuclei with cilia to proportion of nuclei with centrioles": nuc_cilia_to_nuc_cent,
            "Number of cilia/avg len of cilia": avg_length_cilia,
            "Number of centrioles/number of cilia": num_cent_to_cilia,
            "Number of nuclei/number of centrioles": num_nuc_to_cent,
            "Number of nuclei/solidity": num_nuc_to_solidity,
        }
        scatter_dispatch_dict = {
            **scatter_dispatch_dict,
            **{
                f"Nucleus area to cilia {col}": partial(nuclei_to_cilia_scatter, col=col)
                for (col, _) in valid_cilia_attrs
            },
            **{
                f"Centriole area to cilia {col}": partial(cent_area_to_cilia, col=col)
                for (col, _) in valid_cilia_attrs
            },
            **{
                f"Cilia number to cilia {col}": partial(
                    avg_num_cilia_to_measure, col=col
                )
                for col in cols_to_use_scatter_cilia
            },
            **{
                f"Cilia {col} to cilia length": partial(
                    measure_cilia_to_length, col=col
                )
                for col in cols_cilia_len
            },
            **{
                f"Nuclei number to cilia {col}": partial(
                    num_nuc_to_cilia_measure, col_idx=idx, col=col
                )
                for (col, idx) in valid_cilia_attrs
            },
        }

        for title, func in scatter_dispatch_dict.items():
            new_x, new_y, x_label, y_label = func(
                associate_df=associate_df,
                cilia_df=cilia_df,
                cell_df=cell_df,
                centriole_df=centriole_df,
                valid_cilia_df=valid_cilia_df,
                grouped_associates=grouped_associates,
                num_im=num_im,
                grouped_cilia=grouped_cilia,
                grouped_cell=grouped_cell,
                grouped_centriole=grouped_centriole,
                grouped_all_cilia=grouped_all_cilia,
                grouped_all_centriole=grouped_all_centriole,
            )

            plt.scatter(x=new_x, y=new_y)
            plt.title(title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            formatted_title = title.replace(' ', '_').replace('/', '_').lower()
            
            plt.savefig(os.path.join(output_folder, f"{formatted_title}.png"))
            plt.close()

    if args['hist']:

        output_folder = os.path.join(OUTPUT, "histograms")
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        histogram_dispatch_dict = {}

        histogram_dispatch_dict = {
            **histogram_dispatch_dict,
            **{
                f"{names_cilia[col]}": partial(how_many_cilia_per_size, col=col)
                for col in cols_to_use_cilia
            },
            **{
                f"{names_nuc[col]}": partial(how_many_nuc_per_size, col=col)
                for col in cols_to_use_cent
            },
            **{
                f"{names_cent[col]}": partial(how_many_cent_per_size, col=col)
                for col in cols_to_use_cent
            },
        }

        for title, func in histogram_dispatch_dict.items():
            x, bins, y_label = func(
                    cell_df=cell_df,
                    cilia_df=cilia_df,
                    centriole_df=centriole_df,
                    custom_bins=args.get("bins"),
                )
            plt.hist(x=x, bins=bins)
            plt.title(title)
            plt.xlabel(title)
            plt.ylabel(y_label)
            formatted_title = title.replace(' ', '_').replace('/', '_').lower()
            plt.savefig(os.path.join(output_folder, f"{formatted_title}.png"))
            plt.close()

if __name__ == "__main__":
    main()

