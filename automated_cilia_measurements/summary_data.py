from bokeh.models.sources import ColumnDataSource
import pandas as pd
from bokeh.plotting import figure, curdoc
from bokeh.models import Dropdown
from bokeh.layouts import column, row
from functools import partial

################################# TO CHANGE #################################
CSV_FOLDER = "/Users/sneha/Desktop/ciliaJan22/spreadsheets_im_output"
OUTPUT_CSV_DIR_PATH = "/Users/sneha/Desktop/c2coutput/threshold_none"
################################# TO CHANGE #################################
# Initialize figures
def make_figure(title, x_axis_label="", y_axis_label=""):
    p = figure(
        plot_height=600,
        plot_width=600,
        title=title,
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
    )
    return p


# Calculate number of nuclei per image
def num_nuc_per_im(image_df, **kwargs):
    return image_df["Count_Nucleus"].values.tolist()


# Calculate nuclei to cilia ratio
def nuc_per_cilia(grouped_valid_cilia, num_im, image_df, **kwargs):
    num_nuc = num_nuc_per_im(image_df)
    num_cilia = num_cilia_per_im(grouped_valid_cilia, num_im)

    nuc_per_cilia = []
    for im in range(num_im):
        nuc_per_cilia.append(num_cilia[im] / num_nuc[im])

    return nuc_per_cilia


# Calculate number of cilia per image
def num_cilia_per_im(grouped_valid_cilia, num_im, **kwargs):
    return [
        len(make_lists(num, grouped_valid_cilia, "0")) for num in range(1, num_im + 1)
    ]


# Make grouped dataframes into lists
def make_lists(num_im, grouped, colname="ImageNumber", **kwargs):
    """
    Group dataframe into only rows where image is im_num and return the values in a list

    :param num_im: The image number
    :param grouped: The dataframe we want to get relevant rows of
    :returns: list of (x,y) coordinates for all relevant rows of dataframe
    """

    im_df = grouped.get_group(num_im)
    im_df.drop(colname, axis=1, inplace=True)
    return im_df.values.tolist()


# Calculate ratio of nuclei with one centriole to nuclei with two centrioles
def single_cent_to_two_cent(grouped_associates, num_im, **kwargs):
    ratios = []
    for num in range(1, num_im + 1):
        associates_list = make_lists(num, grouped_associates)
        double = 0
        for row in associates_list:
            if "," in row[2]:
                double += 1
        ratios.append((len(associates_list) - double) / double)
    return ratios


# Calculate ratio of the length of the cilia to the size of the nucleus for each cell
def len_cilia_to_size_nucleus(
    grouped_cell, grouped_cilia, grouped_valid_cilia, num_im, **kwargs
):
    result = []
    for num in range(1, num_im + 1):
        cell_li = make_lists(num, grouped_cell)
        avg_nuc = sum(x[1] for x in cell_li) / len(cell_li)

        valid_cilia = make_lists(num, grouped_valid_cilia, "0")
        valid_cilia = set(x[1] for x in valid_cilia)  # Column 1 contains cilia number
        cilia_li = make_lists(num, grouped_cilia)

        cilia_lens = []
        for cilia in cilia_li:
            if int(cilia[0]) not in valid_cilia:
                continue
            cilia_lens.append(cilia[15])
        avg_cilia = sum(cilia_lens) / len(cilia_lens)
        result.append(avg_cilia / avg_nuc)
    return result


# Calculate number of cilia per size for each column
def how_many_cilia_per_size(grouped_cilia, grouped_valid_cilia, num_im, col_idx):
    if col_idx == 0:
        NUM_BINS = 1000
    elif col_idx == 17:
        NUM_BINS = 7
    else:
        NUM_BINS = 500
    result = [[] for bin in range(0, NUM_BINS)]

    col_idx_to_range = {}
    # ex list(range(50,-1,-5))
    for num in range(1, num_im + 1):
        valid_cilia = make_lists(num, grouped_valid_cilia, "0")
        valid_cilia = set(x[1] for x in valid_cilia)  # Column 1 contains cilia number
        cilia_li = make_lists(num, grouped_cilia)

        cilia_size = []
        for cilia in cilia_li:
            if int(cilia[0]) not in valid_cilia:
                continue

            cur_area = cilia[col_idx]
            ranges = list(range(NUM_BINS, 0, -1))
            for c_ind, bucket_distance in enumerate(ranges):
                if cur_area - bucket_distance >= 0:
                    result[len(ranges) - 1 - c_ind].append(bucket_distance)
                    break
        new_result = []
        for bucket in result:
            new_result.append(len(bucket))
    return new_result


# Calculate per-image averages of all columns for cilia
def avg_blank_cilia(grouped_cilia, grouped_valid_cilia, num_im, col_idx, **kwargs):
    return avg_blank_helper(grouped_cilia, grouped_valid_cilia, num_im, col_idx)


# Calculate per-image averages of all columns for centrioles
def avg_blank_centriole(
    grouped_centriole, grouped_valid_cent, num_im, col_idx, **kwargs
):
    return avg_blank_helper(grouped_centriole, grouped_valid_cent, num_im, col_idx)


def avg_blank_helper(grouped, valid, num_im, col_idx):
    result = []
    for num in range(1, num_im + 1):
        valid_li = make_lists(num, valid, "0")
        valid_li = set(x[1] for x in valid_li)  # Column 1 contains cilia number
        measurements_li = make_lists(num, grouped)

        size = []
        for thing in measurements_li:
            if int(thing[0]) not in valid_li:
                continue
            size.append(thing[col_idx + 1])
        result.append(sum(size) / len(size))
    return result


# Calculate per-image averages of all columns for nuclei
def avg_blank_nucleus(grouped_cell, col, **kwargs):
    mean_df = grouped_cell[col].mean()
    return mean_df.values.tolist()


# Calculate nuclei area to cilia area / cilia len / cilia diff diam
def nuclei_to_cilia_scatter(associate_df, cilia_df, cell_df, attr, **kwargs):

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

    cilia_df = cilia_df.rename(columns={"ObjectNumber": "Cilia", attr: f"Cilia {attr}"})
    cell_df = cell_df.rename(
        columns={"ObjectNumber": "Nucleus", "AreaShape_Area": "NucleusArea"}
    )

    df = associate_df.merge(cilia_df, on=["ImageNumber", "Cilia"])
    df = df.merge(cell_df, on=["ImageNumber", "Nucleus"])

    df = df[["NucleusArea", f"Cilia {attr}"]]
    df = df.dropna()
    return [
        df["NucleusArea"].values.tolist(),
        df[f"Cilia {attr}"].values.tolist(),
        "Nuclei Area",
        f"Cilia {attr}",
    ]


# Calculate ratio of cilia area to cilia length
def cilia_area_to_len(valid_cilia_df, cilia_df, **kwargs):
    valid_cilia_df = valid_cilia_df.rename(
        columns={"0": "ImageNumber", "1": "ObjectNumber"}
    )
    df = valid_cilia_df.merge(cilia_df, on=["ImageNumber", "ObjectNumber"])
    df = df[["AreaShape_Area", "AreaShape_MajorAxisLength"]]
    df = df.dropna()
    return [
        df["AreaShape_Area"].values.tolist(),
        df["AreaShape_MajorAxisLength"].values.tolist(),
        "Cilia Area",
        "Cilia length",
    ]


# NOTE This assumes the cell-> centriole, centriole -> cilia
# Calculate ratio of number of cilia to number of centrioles
def nuc_cilia_to_nuc_cent(grouped_associates, num_im, image_df, **kwargs):
    # cilia is not there if cilia ==-2, cent is not there if nucleus is not represented
    result_cilia = []
    result_cent = []
    nuc_count = image_df["Count_Nucleus"].values.tolist()

    for num in range(1, num_im + 1):
        nuc_im = nuc_count[num - 1]
        associates_list = make_lists(num, grouped_associates)
        # no cent means that last nuc in associates list will be offset to the actual number of nuclei
        # this is a list of the form
        cilia_present = 0
        for row in associates_list:
            if row[4] != -2.0:
                cilia_present += 1
        all_rows = len(associates_list)

        result_cilia.append((cilia_present) / nuc_im)
        result_cent.append((all_rows) / nuc_im)
    return (
        result_cilia,
        result_cent,
        "Nuclei with cilia attached",
        "Nuclei with centrioles attached",
    )


# get num cilia/im and average length of cilia/im
def avg_length_cilia(grouped_cilia, grouped_valid_cilia, num_im, **kwargs):
    num_cilia = [
        len(make_lists(num, grouped_valid_cilia, "0")) for num in range(1, num_im + 1)
    ]
    avg_len_cilia = []
    for num in range(1, num_im + 1):
        valid_cilia = make_lists(num, grouped_valid_cilia, "0")
        valid_cilia = set(x[1] for x in valid_cilia)  # Column 1 contains cilia number
        cilia_li = make_lists(num, grouped_cilia)

        cilia_size = []
        for cilia in cilia_li:
            if int(cilia[0]) not in valid_cilia:
                continue
            cilia_size.append(cilia[15])
        avg_len_cilia.append(sum(cilia_size) / len(cilia_size))
    return num_cilia, avg_len_cilia, "Number of cilia", "Average length of cilia"


# Calculate nuclei area to centriole number
def nuc_area_to_cent(
    grouped_associates, grouped_cell, grouped_centriole, num_im, **kwargs
):
    # so, what do we have to do?
    nuc_areas = []
    cent_areas = []
    for num in range(1, num_im + 1):
        associates_li = make_lists(num, grouped_associates)
        cell_li = make_lists(num, grouped_cell)
        cent_li = make_lists(num, grouped_centriole)

        for row in associates_li:
            cur_nuc = (
                int(row[0]) - 1
            )  # this is 1 indexed in the output file so make it 0 indexed for ez access
            nuc_area = cell_li[cur_nuc][1]
            cur_cent = row[2].strip("[]")
            if not "," in cur_cent:  # we are at a singleton, just get the area
                nuc_areas.append(nuc_area)
                cent_areas.append(cent_li[int(cur_cent) - 1][1])
            else:
                cents = cur_cent.split(", ")
                for cent in cents:
                    nuc_areas.append(nuc_area)
                    cent_areas.append(cent_li[int(cent) - 1][1])

    return nuc_areas, cent_areas, "Nucleus area", "Centriole area"


# Calculate ratio of centriole area to cilia number
def cent_area_to_cilia(
    grouped_associates, grouped_cilia, grouped_centriole, num_im, attr, **kwargs
):
    cilia_measures = []
    cent_areas = []
    measure_dict = {
        "AreaShape_Area": 1,
        "AreaShape_MajorAxisLength": 15,
        "AreaShape_EquivalentDiameter": 11,
    }
    for num in range(1, num_im + 1):
        associates_li = make_lists(num, grouped_associates)
        cell_li = make_lists(num, grouped_cilia)
        cent_li = make_lists(num, grouped_centriole)

        for row in associates_li:
            if not int(row[0]) == -2:
                cur_cilia = (
                    int(row[4]) - 1
                )  # this is 1 indexed in the output file so make it 0 indexed for ez access
                cilia_measure = cell_li[cur_cilia][measure_dict[attr]]
                cur_cent = row[2].strip("[]")
                if not "," in cur_cent:  # we are at a singleton, just get the area
                    cilia_measures.append(cilia_measure)
                    cent_areas.append(cent_li[int(cur_cent) - 1][1])
                else:
                    cents = cur_cent.split(", ")
                    for cent in cents:
                        cilia_measures.append(cilia_measure)
                        cent_areas.append(cent_li[int(cent) - 1][1])

    return cent_areas, cilia_measures, "Centriole area", f"Cilia {attr}"


def main():
    # Load data
    cell_df = pd.read_csv(CSV_FOLDER + "/MyExpt_Nucleus.csv", skipinitialspace=True)
    num_im = cell_df.ImageNumber.iat[-1]
    grouped_cell = cell_df.groupby(["ImageNumber"])
    centriole_df = pd.read_csv(
        CSV_FOLDER + "/MyExpt_Centriole.csv", skipinitialspace=True
    )
    grouped_centriole = centriole_df.groupby(["ImageNumber"])
    cilia_df = pd.read_csv(CSV_FOLDER + "/MyExpt_Cilia.csv", skipinitialspace=True)
    grouped_cilia = cilia_df.groupby(["ImageNumber"])
    cols_to_use = list(cilia_df.columns)[2:]
    image_df = pd.read_csv(CSV_FOLDER + "/MyExpt_Image.csv", skipinitialspace=True)

    associate_df = pd.read_csv(
        OUTPUT_CSV_DIR_PATH + "/c2coutput.csv", skipinitialspace=True
    )
    grouped_associates = associate_df.groupby(["ImageNumber"])

    valid_cilia_df = pd.read_csv(
        OUTPUT_CSV_DIR_PATH + "/new_cilia.csv", skipinitialspace=True
    )
    grouped_valid_cilia = valid_cilia_df.groupby(["0"])
    valid_cent_df = pd.read_csv(
        OUTPUT_CSV_DIR_PATH + "/new_cent.csv", skipinitialspace=True
    )
    grouped_valid_cent = valid_cent_df.groupby(["0"])

    # Make dispatch dictionaries for all graphs to easily convert between them
    histogram_dispatch_dict = {
        "Num nuclei per im": num_nuc_per_im,
        "Nuclei with 2 cent/Nuclei with 1 cent": single_cent_to_two_cent,
        "Num cilia per im": num_cilia_per_im,
        "Avg len of cilia/size of nuclei": len_cilia_to_size_nucleus,
        "Num nuclei to num cilia": nuc_per_cilia,
    }
    histogram_dispatch_dict = {
        **histogram_dispatch_dict,
        **{
            f"Avg {col} of cilia": partial(avg_blank_cilia, col_idx=idx)
            for idx, col in enumerate(cols_to_use)
        },
        **{
            f"Avg {col} of centriole": partial(avg_blank_centriole, col_idx=idx)
            for idx, col in enumerate(cols_to_use)
        },
        **{
            f"Avg {col} of nuclei": partial(avg_blank_nucleus, col=col)
            for col in cols_to_use
        },
    }

    valid_cilia_attrs = [
        "AreaShape_Area",
        "AreaShape_MajorAxisLength",
        "AreaShape_EquivalentDiameter",
    ]

    scatter_dispatch_dict = {
        "Cilia area to length": cilia_area_to_len,
        "Proportion of nuclei with cilia to proportion of nuclei with centrioles": nuc_cilia_to_nuc_cent,
        "Number of cilia/avg len of cilia": avg_length_cilia,
        "Nucleus area to centriole area": nuc_area_to_cent,
    }
    scatter_dispatch_dict = {
        **scatter_dispatch_dict,
        **{
            f"Nucleus area to cilia {attr}": partial(nuclei_to_cilia_scatter, attr=attr)
            for attr in valid_cilia_attrs
        },
        **{
            f"Centriole area to cilia {attr}": partial(cent_area_to_cilia, attr=attr)
            for attr in valid_cilia_attrs
        },
    }

    cilia_per_thing_dispatch_dict = {}

    cilia_per_thing_dispatch_dict = {
        **cilia_per_thing_dispatch_dict,
        **{
            f"Number of cilia per 5 in {col}": partial(
                how_many_cilia_per_size, col_idx=idx
            )
            for idx, col in enumerate(cols_to_use)
        },
    }

    # Make figures
    histogram_figure = make_figure(title="Histograms", x_axis_label="Image")
    scatter_figure = make_figure(title="Scattergrams")
    cilia_per_thing_figure = make_figure(
        title="Cilia per measurement", y_axis_label="Amount of Cilia"
    )

    histogram = ColumnDataSource({"top": [], "left": [], "right": []})
    scatter = ColumnDataSource({"x": [], "y": []})
    cilia_per_thing = ColumnDataSource({"top": [], "left": [], "right": []})

    histogram_figure.quad(
        source=histogram, top="top", left="left", right="right", bottom=0
    )
    scatter_figure.scatter(source=scatter, x="x", y="y")
    cilia_per_thing_figure.quad(
        source=cilia_per_thing, top="top", left="left", right="right", bottom=0
    )

    # Make graphs interactive
    def histogram_selection_callback(event):
        new_data = histogram_dispatch_dict[event.item](
            num_im=num_im,
            image_df=image_df,
            grouped_associates=grouped_associates,
            grouped_cell=grouped_cell,
            grouped_cilia=grouped_cilia,
            grouped_centriole=grouped_centriole,
            grouped_valid_cilia=grouped_valid_cilia,
            grouped_valid_cent=grouped_valid_cent,
        )
        histogram.data = {
            "left": [i for i in range(len(new_data))],
            "right": [i + 1 for i in range(len(new_data))],
            "top": new_data,
        }

        histogram_figure.yaxis.axis_label = event.item

    def cilia_per_thing_selection_callback(event):
        new_data = cilia_per_thing_dispatch_dict[event.item](
            num_im=num_im,
            grouped_cilia=grouped_cilia,
            grouped_valid_cilia=grouped_valid_cilia,
        )
        cilia_per_thing.data = {  # TODO fix whatever's going wrong here ....
            "left": [i for i in list(range(0, 500))],
            "right": [i + 1 for i in list(range(0, 500))],
            "top": new_data,
        }

        cilia_per_thing_figure.xaxis.axis_label = event.item

    def scatter_selection_callback(event):
        new_x, new_y, x_label, y_label = scatter_dispatch_dict[event.item](
            associate_df=associate_df,
            cilia_df=cilia_df,
            cell_df=cell_df,
            image_df=image_df,
            centriole_df=centriole_df,
            valid_cilia_df=valid_cilia_df,
            grouped_associates=grouped_associates,
            num_im=num_im,
            grouped_cilia=grouped_cilia,
            grouped_valid_cilia=grouped_valid_cilia,
            grouped_cell=grouped_cell,
            grouped_centriole=grouped_centriole,
        )
        scatter.data = {
            "x": new_x,
            "y": new_y,
        }
        scatter_figure.yaxis.axis_label = y_label
        scatter_figure.xaxis.axis_label = x_label

    # Add dropdown to change graphs
    histogram_dropdown = Dropdown(
        label="Summary Statistic", menu=[(key, key) for key in histogram_dispatch_dict]
    )
    scatter_dropdown = Dropdown(
        label="Summary Scatterplot", menu=[(key, key) for key in scatter_dispatch_dict]
    )

    cilia_per_thing_dropdown = Dropdown(
        label="Measurement to sort cilia by",
        menu=[(key, key) for key in cilia_per_thing_dispatch_dict],
    )

    # Integrate dropdown and put together layout
    histogram_dropdown.on_click(histogram_selection_callback)
    scatter_dropdown.on_click(scatter_selection_callback)
    cilia_per_thing_dropdown.on_click(cilia_per_thing_selection_callback)
    layout = column(
        histogram_dropdown,
        scatter_dropdown,
        row(histogram_figure, scatter_figure),
        cilia_per_thing_dropdown,
        cilia_per_thing_figure,
    )
    curdoc().add_root(layout)


main()
