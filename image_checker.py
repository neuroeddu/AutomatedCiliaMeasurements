import pandas as pd
from PIL import Image, ImageDraw

df = pd.read_csv(
    "/Users/sneha/Desktop/mni/cilia_true.csv",
    skipinitialspace=True,
    usecols=["ImageNum", "XCoord", "YCoord"],
)

grouped = df.groupby(["ImageNum"])

for num in range(1, 8):
    im_df = grouped.get_group(num)
    im_df.drop("ImageNum", axis=1, inplace=True)
    coords_list = im_df.values.tolist()
    im = (
        "/Users/sneha/Desktop/cilia_TEST/im_output/CiliaOverlay" + f"{num:04}" + ".tiff"
    )
    img = Image.open(im)

    # Writes number onto image at center
    for _, val in enumerate(coords_list):

        x_coord = val[0]
        y_coord = val[1]
        d = ImageDraw.Draw(img)
        d.text((x_coord, y_coord), "HERE", fill=(255, 0, 0, 255))

    path = "/Users/sneha/Desktop/cilia_TEST/trash/CiliaOverlay_" + f"{num:04}" + ".tiff"
    img.save(path)
