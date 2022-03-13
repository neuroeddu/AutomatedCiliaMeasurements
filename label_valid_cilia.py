# Labels images for a particular organelle
import pandas as pd
from PIL import Image, ImageDraw
import argparse


def label_im(coordinate_list, im, num, channel, output_path):
    img = Image.open(im)

    # Writes number onto image at center
    for i, val in enumerate(coordinate_list):
        x_coord = val[1]
        y_coord = val[2]
        d = ImageDraw.Draw(img)
        write_num = str(i + 1)
        d.text((x_coord, y_coord), write_num, fill=(255, 0, 0, 255))

    path = make_paths(num, True, channel, output_path)
    img.save(path)


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
        "-h",
        "--channel",
        help="channel number for images (01 is nuclei, 02 is cilia, 03 is centriole)",
        required=True,
    )

    return vars(parser.parse_args())


def main():

    args = parse_args()

    measurements_df = pd.read_csv(
        args["measurements"],
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
        valid_df = pd.read_csv(args["c2c"], skipinitialspace=True)
        valid_df = valid_df.rename(columns={"0": "ImageNumber", "1": "ObjectNumber"})
        measurements_df = valid_df.merge(
            measurements_df, on=["ImageNumber", "ObjectNumber"]
        )

    grouped_cilia = measurements_df.groupby(["ImageNumber"])
    # Get number of images, either from the number inputted or from the total number of images
    images = int(args.get("num")) or measurements_df.ImageNumber.iat[-1]

    for num in range(1, images + 1):
        # Get list of coords to plot
        coords_df = grouped_cilia.get_group(num)
        coords_df.drop(["ImageNumber", "ObjectNumber"], axis=1, inplace=True)
        coords_list = coords_df.values.tolist()

        # Get path and
        im_path = make_paths(num, False, args["channel"], args["images"])
        label_im(coords_list, im_path, num, args["channel"], args["output"])


if __name__ == "__main__":
    main()
