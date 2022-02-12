"""TODO centrioles_without_cells docstring."""
import pandas as pd
from math import sqrt
from collections import defaultdict
from PIL import Image, ImageDraw

################################# TO CHANGE #################################
cell_csv_path = '/Users/sneha/Desktop/mni/cilia_09:12:2021/im_output/MyExpt_Nucleus.csv'
cilia_csv_path = '/Users/sneha/Desktop/mni/cilia_09:12:2021/im_output/MyExpt_Cilia.csv'
centriole_csv_path = '/Users/sneha/Desktop/mni/cilia_09:12:2021/im_output/MyExpt_Centriole.csv'
output_im_dir_path = '/Users/sneha/Desktop/mni/cilia_09:12:2021/visualizer/'
im_csv_dir_path = '/Users/sneha/Desktop/mni/cilia_09:12:2021/im_output/'
################################# TO CHANGE #################################


def helper_make_lists(im_num, grouped):
    """TODO helper_make_lists docstring."""
    im_df = grouped.get_group(im_num)
    im_df.drop('ImageNumber', axis=1, inplace=True)
    return im_df.values.tolist()


# makes lists
def make_lists(im_num, grouped_cell, grouped_centriole):
    """TODO make_lists docstring."""
    cell_list = helper_make_lists(im_num, grouped_cell)
    centriole_list = helper_make_lists(im_num, grouped_centriole)

    return cell_list, centriole_list


# finds out which cilia is closest to which cell assuming cutoff passed in (or none if not) & 1:1
# cell:cilia relationship
def which_cilia_closest(cell_list, cilia_list, cutoff=float('inf')):
    """TODO which_cilia_closest docstring."""
    cell_to_cilia = [
        {
            "cilia": None,  # The index of the current closest cilia
            "cilia_tried": set()  # The indicies of previous cilia that shouldn't be tried again
        }
        for cell in range(len(cell_list))
    ]
    cilia_to_cell = [
        {
            "path_length": float('inf'),  # The length of the shortest path
            "cell": None  # The index of the cell to which the shortest path corresponds
        }
        for cilia in cilia_list
    ]

    updated_cilia = True

    # while cilia are being updated, calculate lengths and see if the cilia should be added
    while updated_cilia:
        updated_cilia = False
        for i, cell in enumerate(cell_list):
            x_cell = cell[0]
            y_cell = cell[1]
            for j, cilia in enumerate(cilia_list):
                x_cilia = cilia[0]
                y_cilia = cilia[1]
                result = sqrt(pow((x_cilia - x_cell), 2) + pow((y_cilia - y_cell), 2))

                if (
                    result > cutoff or
                    j in cell_to_cilia[i]["cilia_tried"] or
                    result >= cilia_to_cell[j]["path_length"]
                ):
                    continue

                add_cilia(cell_to_cilia, cilia_to_cell, result, i, j)
                updated_cilia = True

    return cilia_to_cell


# associate a cell with a cilia
def add_cilia(cell_to_cilia, cilia_to_cell, result, cell, cilia):
    """TODO add_cilia docstring."""
    if cilia_to_cell[cilia]["cell"] is None:
        cilia_to_cell[cilia]["cell"] = cell+1
        cilia_to_cell[cilia]["path_length"] = result
        cell_to_cilia[cell]["cilia"] = cilia
        return

    else:
        old_cell = cilia_to_cell[cilia]["cell"]
        cilia_to_cell[cilia]["cell"] = cell+1
        cilia_to_cell[cilia]["path_length"] = result
        cell_to_cilia[cell]["cilia"] = cilia
        cell_to_cilia[old_cell]["cilia"] = None
        cell_to_cilia[old_cell]["cilia_tried"].add(cilia)


def get_abandoned_centrioles(cilia_to_cell, cell_list):
    """TODO get_abandoned_centrioles docstring."""
    # let's store each cell and its centrioles here
    cell_to_cilia = defaultdict(list)
    for cilia_index, cilia_info in enumerate(cilia_to_cell):
        cell_to_cilia[cilia_info['cell']].append(
            (cilia_index, cilia_info['path_length'])
        )

    duplicate_cell_to_cilia = {}
    for cell, cilia_list in cell_to_cilia.items():
        if len(cilia_list) >= 3:
            duplicate_cell_to_cilia[cell] = cilia_list

    return duplicate_cell_to_cilia


def make_paths(num, label):  # makes paths for us to be able to find init imgs / for images to go
    """TODO add make_paths docstring."""
    if label:
        path = (output_im_dir_path + 'NucleusOverlay' + f"{num:04}" + '_LABELED_FULL.tiff')

    else:
        path = (im_csv_dir_path + 'NucleusOverlay' + f"{num:04}" + '.tiff')

    return path


# {cell: [(centriole, path length), (centriole, path length), (centriole, path length)]}
def label_centrioles_without_cells(cell_list, centriole_list, dup_dict, im, num):
    """TODO add label_centrioles_without_cells docstring."""
    img = Image.open(im)

    for i, val in enumerate(cell_list):  # labels cell li pt

        centriole_li = dup_dict.get(i+1)
        if centriole_li:
            x_coord, y_coord = val
            d = ImageDraw.Draw(img)
            write_num = str(i+1)
            d.text((x_coord, y_coord), write_num, fill=(255, 255, 255, 255))

            centriole_li.sort(key=lambda x: x[1])
            for num, (centriole, p) in enumerate(centriole_li):
                x_coord_cent, y_coord_cent = centriole_list[centriole]
                d = ImageDraw.Draw(img)
                write_num = str(centriole)
                d.text((x_coord_cent, y_coord_cent), write_num, fill=(255, 255, 255, 255))
                if num < 2:
                    line_xy = [(x_coord_cent, y_coord_cent), (x_coord, y_coord)]
                    d = ImageDraw.Draw(img)
                    d.line(line_xy, fill=(0, 0, 255, 255))
                else:
                    line_xy = [(x_coord_cent, y_coord_cent), (x_coord, y_coord)]
                    d = ImageDraw.Draw(img)
                    d.line(line_xy, fill=(255, 255, 255, 255))

    path = make_paths(num, True)
    img.save(path)


def main():
    """TODO add main docstring."""
    fields = ['ImageNumber', 'Location_Center_X', 'Location_Center_Y']
    cell_df = pd.read_csv(cell_csv_path, skipinitialspace=True, usecols=fields)
    num_im = cell_df.ImageNumber.iat[-1]
    grouped_cell = cell_df.groupby(['ImageNumber'])
    centriole_df = pd.read_csv(centriole_csv_path, skipinitialspace=True, usecols=fields)
    grouped_centriole = centriole_df.groupby(['ImageNumber'])

    for num in range(1, num_im+1):
        cell_list, centriole_list = make_lists(num, grouped_cell, grouped_centriole)
        centriole_to_cell = which_cilia_closest(cell_list, centriole_list)
        dups_dict = get_abandoned_centrioles(centriole_to_cell, cell_list)
        im_path = make_paths(num, False)
        label_centrioles_without_cells(cell_list, centriole_list, dups_dict, im_path, num)


if __name__ == "__main__":
    main()
