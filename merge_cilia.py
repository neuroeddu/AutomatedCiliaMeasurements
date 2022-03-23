import csv
import pandas as pd
from math import sqrt
from sklearn.cluster import DBSCAN
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
import numpy as np
from PIL import Image, ImageDraw, ImageFont

################################# TO CHANGE #################################
cilia_csv_path = "/Users/sneha/Desktop/mni/cilia_09:12:2021/im_output/MyExpt_Cilia.csv"
output_im_dir_path = "/Users/sneha/Desktop/yann\ is\ cute."
im_csv_dir_path = "/Users/sneha/Desktop/mni/cilia_09:12:2021/im_output/"
################################# TO CHANGE #################################


def make_paths(
    num, label
):  # makes paths for us to be able to find init imgs / for images to go
    if label:
        path = (
            output_im_dir_path + "NucleusOverlay" + f"{num:04}" + "_LABELED_FULL.tiff"
        )

    else:
        path = im_csv_dir_path + "NucleusOverlay" + f"{num:04}" + ".tiff"

    return path


fields = ["ImageNumber", "Location_Center_X", "Location_Center_Y"]
cilia_df = pd.read_csv(cilia_csv_path, skipinitialspace=True, usecols=fields)
num_im = cilia_df.ImageNumber.iat[-1]
grouped_cilia = cilia_df.groupby(["ImageNumber"])

fields_polygons = [
    "ImageNumber",
    "AreaShape_BoundingBoxMaximum_X",
    "AreaShape_BoundingBoxMaximum_Y",
    "AreaShape_BoundingBoxMinimum_X",
    "AreaShape_BoundingBoxMinimum_Y",
]
cilia_df_polygons = pd.read_csv(
    cilia_csv_path, skipinitialspace=True, usecols=fields_polygons
)
grouped_cilia_polygons = cilia_df_polygons.groupby(["ImageNumber"])

db = DBSCAN(eps=5, min_samples=1)

for num in range(1, num_im + 1):
    im_df = grouped_cilia.get_group(num_im)
    im_df.drop("ImageNumber", axis=1, inplace=True)
    im_df = im_df.values.tolist()

    im_df_polygons = grouped_cilia_polygons.get_group(num_im)
    im_df_polygons.drop("ImageNumber", axis=1, inplace=True)
    im_df_polygons = im_df_polygons.values.tolist()

    clustering = db.fit(X=im_df)
    labels = clustering.labels_
    # 0 -1
    # 1 2
    num_clusters = len(np.unique(labels))
    print("Estimated no. of clusters: %d" % num_clusters)
    cluster_li = [[] for cluster in range(num_clusters)]
    noise = []
    for x, label in enumerate(labels):
        if label != -1:
            cluster_li[label].append(x)
        else:
            noise.append(x)

    cilia_vertices = []
    # output: [(minx, miny), (maxx, maxy), (minx, maxy), (maxx, miny)] for each cilia

    for cilia in im_df_polygons:

        maxx, maxy, minx, miny = cilia
        cilia_vertices.append([(minx, miny), (maxx, maxy), (minx, maxy), (maxx, miny)])

    polygon_li = []

    for cluster in cluster_li:
        cur_polygon = None
        for cilia in cluster:
            if not cur_polygon:
                cur_polygon = Polygon(cilia_vertices[cilia])

            else:
                cur_polygon = cascaded_union(
                    [cur_polygon, Polygon(cilia_vertices[cilia])]
                )

        polygon_li.append(cur_polygon)

    im_path = make_paths(num, False)

    img = Image.open(im_path)
    d = ImageDraw.Draw(img)
    for multi_polygon in polygon_li:
        coords_list = []
        multi_polygon = (
            [multi_polygon] if isinstance(multi_polygon, Polygon) else multi_polygon
        )
        for polygon in multi_polygon:
            coords_list += list(zip(*polygon.exterior.coords.xy))

        d.polygon(coords_list, fill=(255, 255, 255, 255))

    path = make_paths(num, True)
    img.save(path)
