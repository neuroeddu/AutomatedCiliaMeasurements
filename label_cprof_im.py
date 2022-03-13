import pandas as pd
from PIL import Image, ImageDraw
import argparse

# Makes paths for us to be able to find init imgs / for images to go
def make_paths(num, channel, label, path):
    CHANNEL_DICT = {
        "01": "NucleusOverlay",
        "02": "CiliaOverlay",
        "03": "CentrioleOverlay",
    }
    path = (
        path
        + CHANNEL_DICT[channel]
        + f"{num:04}"
        + ("_LABELED.tiff" if label else ".tiff")
    )
    print(path)
    return path


# Makes list of coordinates for each df
def helper_make_lists(im_num, grouped, changed_num=None):
    im_df = grouped.get_group(im_num)
    im_df.drop("ImageNumber", axis=1, inplace=True)
    new_list = im_df.values.tolist()
    if changed_num:
        im_df_num = im_df[changed_num]
        new_li_num = im_df_num.values.tolist()
        return new_list, new_li_num
    return new_list


# Makes lists of coordinates
def make_lists(im_num, grouped_cell, grouped_cilia, grouped_centriole):
    cell_list = helper_make_lists(im_num, grouped_cell)
    cilia_list, cilia_list_num = helper_make_lists(im_num, grouped_cilia, "Cilia")
    centriole_list = None
    centriole_list = grouped_centriole and helper_make_lists(im_num, grouped_centriole)

    return cell_list, cilia_list, centriole_list, cilia_list_num


# Labels image
def label_im(coordinate_list, im, num, channel, output_path, li_num=None):
    img = Image.open(im)

    # Writes number onto image at center
    for i, val in enumerate(coordinate_list):
        x_coord = val[0]
        y_coord = val[1]
        d = ImageDraw.Draw(img)
        if li_num:
            write_num = str(int(li_num[i]))
        else:
            write_num = str(i + 1)
        d.text((x_coord, y_coord), write_num, fill=(255, 255, 255, 255))

    path = make_paths(num, channel, True, output_path)
    img.save(path)


def parse_args():
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


def batch_script():

    args = parse_args()

    CSV_FOLDER = args["input"]
    IM_CSV_DIR_PATH = args["images"]
    OUTPUT_IM_DIR_PATH = args["output"]

    # Columns we need to keep
    cilia_fields = ["ImageNumber", "Cilia", "Location_Center_X", "Location_Center_Y"]
    nuclei_fields = ["ImageNumber", "Location_Center_X", "Location_Center_Y"]
    centriole_fields = [
        "ImageNumber",
        "Centriole",
        "Location_Center_X",
        "Location_Center_Y",
    ]
    # Reads csv and groups by the im num
    cell_df = pd.read_csv(
        CSV_FOLDER + "/MyExpt_Nucleus.csv", skipinitialspace=True, usecols=nuclei_fields
    )
    grouped_cell = cell_df.groupby(["ImageNumber"])

    grouped_cilia = pd.read_csv(
        CSV_FOLDER + "/MyExpt_Cilia.csv", skipinitialspace=True, usecols=cilia_fields
    ).groupby(["ImageNumber"])

    grouped_centriole = None
    # If we have centriole images, read them too. If not, keep the grouped as none (so that we can pass it into the next func)
    grouped_centriole = args.get("centriole") and pd.read_csv(
        CSV_FOLDER + "/MyExpt_Centriole.csv",
        skipinitialspace=True,
        usecols=centriole_fields,
    ).groupby(["ImageNumber"])

    # Get number of images, either from the number inputted or from the total number of images
    images = int(args.get("num")) or cell_df.ImageNumber.iat[-1]

    # Iterate through the images. Make list of nuclei/cilia/centrioles, then make paths for our current image & label+save
    # image.
    for num in range(1, images + 1):
        cell_list, cilia_list, centriole_list, cilia_list_num = make_lists(
            num, grouped_cell, grouped_cilia, grouped_centriole
        )

        im_path_cell = make_paths(num, "01", False, IM_CSV_DIR_PATH)
        label_im(cell_list, im_path_cell, num, "01", OUTPUT_IM_DIR_PATH)

        im_path_cilia = make_paths(num, "02", False, IM_CSV_DIR_PATH)
        label_im(
            cilia_list, im_path_cilia, num, "02", OUTPUT_IM_DIR_PATH, cilia_list_num
        )

        if args.get("centriole"):
            im_path_centriole = make_paths(num, "03", False, IM_CSV_DIR_PATH)
            label_im(centriole_list, im_path_centriole, num, "03", OUTPUT_IM_DIR_PATH)


def main():
    batch_script()


if __name__ == "__main__":
    main()
