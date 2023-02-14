import pandas as pd
from PIL import Image, ImageDraw
import argparse
from os.path import join


def make_paths(num, channel, label, path):
    """
    Make paths to get labels for images

    :param num: Image number
    :param channel: Channel of image
    :param label: Whether the image is labeled or not
    :param path: Image directory path
    :returns: Path of image to draw on
    """
    CHANNEL_DICT = {
        "01": "NucleusOverlay",
        "02": "CiliaOverlay",
        "03": "CentrioleOverlay",
    }
    path = join(
        path,
        (CHANNEL_DICT[channel] + f"{num:04}" + ("_LABELED.tiff" if label else ".tiff")),
    )
    return path


def helper_make_lists(im_num, grouped):
    """
    Group dataframe into only rows where image is im_num and return the values in a list

    :param im_num: The image number
    :param grouped: The dataframe we want to get relevant rows of
    :returns: list of (x,y) coordinates for all relevant rows of dataframe
    """
    im_df = (grouped.get_group(im_num)).copy()
    im_df.drop("ImageNumber", axis=1, inplace=True)
    new_list = im_df.values.tolist()
    return new_list



def make_lists(im_num, grouped_cell, grouped_cilia, grouped_centriole):
    """
    Makes lists of coordinates for each type of stain

    :param im_num: The image number
    :param grouped_cell: Cell measurements dataframe
    :param grouped_cilia: Cilia measurements dataframe
    :param grouped_centriole: Centriole measurements dataframe
    :returns: List of measurements for each dataframe
    """
    cell_list = helper_make_lists(im_num, grouped_cell)
    cilia_list = helper_make_lists(im_num, grouped_cilia)
    centriole_list = None
    centriole_list = grouped_centriole and helper_make_lists(im_num, grouped_centriole)

    return cell_list, cilia_list, centriole_list


def label_im(coordinate_list, im, num, channel, output_path):
    """
    Draw numbers onto images

    :param coordinate_list: (x,y) coordinates for each organelle 
    :param im: Image path
    :param num: Image number
    :param channel: Channel of stain
    :param output_path: Image output directory
    :returns: None
    """
    img = Image.open(im)

    # Writes number onto image at center
    for i, val in enumerate(coordinate_list):
        x_coord = val[0]
        y_coord = val[1]
        d = ImageDraw.Draw(img)

        write_num = str(i + 1)
        d.text((x_coord, y_coord), write_num, fill=(255, 255, 255, 255))

    path = make_paths(num, channel, True, output_path)
    img.save(path)


def parse_args():
    """
    Parse passed in arguments

    :returns: Necessary arguments to use the script
    """
    # parsing
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", help="folder with cellprofiler CSVs path", required=True
    )
    parser.add_argument(
        "-m", "--images", help="folder with cellprofiler images path", required=True
    )
    parser.add_argument("-o", "--output", help="output folder path", required=True)
    parser.add_argument(
        "-n",
        "--num",
        help="number of images to label, if specific number of im wanted",
        required=False,
    )
    parser.add_argument(
        "-c",
        "--centriole",
        help="enter something here if you want centriole",
        required=False,
    )

    return vars(parser.parse_args())


def main(**args):

    args = args or parse_args()

    CSV_FOLDER = args["input"]
    IM_CSV_DIR_PATH = args["images"]
    OUTPUT_IM_DIR_PATH = args["output"]

    # Columns we need to keep
    fields = ["ImageNumber", "Location_Center_X", "Location_Center_Y"]

    # Reads csv and groups by the im num
    cell_df = pd.read_csv(
        CSV_FOLDER + "/MyExpt_Nucleus.csv", skipinitialspace=True, usecols=fields
    )
    grouped_cell = cell_df.groupby(["ImageNumber"])

    grouped_cilia = pd.read_csv(
        CSV_FOLDER + "/MyExpt_Cilia.csv", skipinitialspace=True, usecols=fields
    ).groupby(["ImageNumber"])

    grouped_centriole = None
    # If we have centriole images, read them too. If not, keep the grouped as none (so that we can pass it into the next func)
    grouped_centriole = args.get("centriole") and pd.read_csv(
        CSV_FOLDER + "/MyExpt_Centriole.csv",
        skipinitialspace=True,
        usecols=fields,
    ).groupby(["ImageNumber"])

    # Get number of images, either from the number inputted or from the total number of images
    # int(args.get("num"))

    images = args.get("num") or cell_df.ImageNumber.iat[-1]
    images = int(images)

    # Iterate through the images. Make list of nuclei/cilia/centrioles, then make paths for our current image & label+save
    # image.
    for num in range(1, images + 1):
        cell_list, cilia_list, centriole_list = make_lists(
            num, grouped_cell, grouped_cilia, grouped_centriole
        )

        im_path_cell = make_paths(num, "01", False, IM_CSV_DIR_PATH)
        label_im(cell_list, im_path_cell, num, "01", OUTPUT_IM_DIR_PATH)

        im_path_cilia = make_paths(num, "02", False, IM_CSV_DIR_PATH)
        label_im(cilia_list, im_path_cilia, num, "02", OUTPUT_IM_DIR_PATH)

        if args.get("centriole"):
            im_path_centriole = make_paths(num, "03", False, IM_CSV_DIR_PATH)
            label_im(centriole_list, im_path_centriole, num, "03", OUTPUT_IM_DIR_PATH)


if __name__ == "__main__":
    main()
