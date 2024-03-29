# Labels images for a particular organelle
import pandas as pd
from PIL import Image, ImageDraw
import argparse
from os.path import join

CHANNEL_DICT = {"01": "Nucleus", "02": "Cilia", "03": "Centriole"}


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
    for val in coordinate_list:
        # Coordinate list values are slightly different
        offset = -1 if channel == "01" else 0
        x_coord = val[2 + offset]
        try:
            y_coord = val[3 + offset]
        except:
            print(val)
            raise
        d = ImageDraw.Draw(img)
        write_num = str(val[1 + offset])
        d.text((x_coord, y_coord), write_num, fill=(255, 0, 0, 255))

    path = make_paths(num, True, channel, output_path)
    img.save(path)


def make_paths(num, label, channel, path):
    """
    Make paths to get labels for images

    :param num: Image number
    :param label: Whether the image is labeled or not
    :param channel: Channel of image
    :param path: Image directory path
    :returns: Path of image to draw on
    """
    path = join(
        path,
        (
            CHANNEL_DICT[channel]
            + "Overlay"
            + f"{num:04}"
            + ("_LABELED.tiff" if label else ".tiff")
        ),
    )
    return path


def parse_args():
    """
    Parse passed in arguments

    :returns: Necessary arguments to use the script
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--measurements", help="path to cellprofiler measurements", required=True
    )
    parser.add_argument(
        "-c",
        "--c2c",
        help="path to c2c cilia/cent CSV if labeling cilia/cent",
        required=False,
    )
    parser.add_argument("-o", "--output", help="output folder path", required=True)
    parser.add_argument(
        "-g", "--images", help="folder with cellprofiler images path", required=True
    )
    parser.add_argument(
        "-n",
        "--num",
        help="number of images to label, if specific number of im wanted",
        required=False,
    )
    parser.add_argument(
        "-a",
        "--channel",
        help="channel number for images (01 is nuclei, 02 is cilia, 03 is centriole)",
        required=True,
    )

    return vars(parser.parse_args())


def main(**args):

    args = args or parse_args()

    measurements_path = (
        join(args["measurements"], "MyExpt_") + CHANNEL_DICT[args["channel"]] + ".csv"
    )

    measurements_df = pd.read_csv(
        measurements_path,
        skipinitialspace=True,
        usecols=[
            "ImageNumber",
            "ObjectNumber",
            "Location_Center_X",
            "Location_Center_Y",
        ],
    )

    # if not all measurements are valid, merge
    c2c_path = args.get("c2c", "")
    if c2c_path:
        c2c_result_type = "new_cilia.csv" if args["channel"] == "02" else "new_cent.csv"
        c2c_path = join(c2c_path, c2c_result_type)
        valid_df = pd.read_csv(c2c_path, skipinitialspace=True)
        valid_df = valid_df.rename(columns={"0": "ImageNumber", "1": "ObjectNumber"})
        measurements_df = valid_df.merge(
            measurements_df, on=["ImageNumber", "ObjectNumber"]
        )

    grouped_cilia = measurements_df.groupby(["ImageNumber"])
    # Get number of images, either from the number inputted or from the total number of images
    images = args.get("num") or measurements_df.ImageNumber.iat[-1]
    images = int(images)

    for num in range(1, images + 1):
        # Get list of coords to plot
        coords_df = (grouped_cilia.get_group(num)).copy()
        coords_df.drop(["ImageNumber"], axis=1, inplace=True)

        coords_list = coords_df.values.tolist()

        # Get path and label
        im_path = make_paths(num, False, args["channel"], args["images"])
        label_im(coords_list, im_path, num, args["channel"], args["output"])


if __name__ == "__main__":
    main()
