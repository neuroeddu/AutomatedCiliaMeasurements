# Labels images for a particular organelle
import pandas as pd
from PIL import Image, ImageDraw
import argparse
from os.path import join

# TODO POETRY INSTALL AND MAKE IT WORK WITHIN POETRY

CHANNEL_DICT = {"01": "Nucleus", "02": "Cilia", "03": "Centriole"}


def label_im(coordinate_list, im, num, channel, output_path):
    img = Image.open(im)

    # Writes number onto image at center
    for _, val in enumerate(coordinate_list):

        x_coord = val[1]
        y_coord = val[2]
        d = ImageDraw.Draw(img)
        write_num = str(val[0] + 1)
        d.text((x_coord, y_coord), write_num, fill=(255, 0, 0, 255))

    path = make_paths(num, True, channel, output_path)
    img.save(path)


def make_paths(num, label, channel, path):
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


def main():

    args = parse_args()
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
    if args.get("c2c"):
        c2c_result_type = "new_cilia.csv" if args["channel"] == "02" else "new_cent.csv"
        c2c_path = join(args.get("c2c"), c2c_result_type)
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
        coords_df = grouped_cilia.get_group(num)
        coords_df.drop(["ImageNumber"], axis=1, inplace=True)
        coords_list = coords_df.values.tolist()

        # Get path and
        im_path = make_paths(num, False, args["channel"], args["images"])
        label_im(coords_list, im_path, num, args["channel"], args["output"])


if __name__ == "__main__":
    main()
