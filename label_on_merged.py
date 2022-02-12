"""TODO label_on_merged docstring."""
import pandas as pd
from PIL import Image, ImageDraw

################################# TO CHANGE #################################
CSV_FOLDER = '/Users/sneha/Desktop/ciliaJan22/spreadsheets_im_output'
IM_CSV_DIR_PATH = '/Users/sneha/Desktop/ciliaNov11/im_output/'
OUTPUT_PATH = '/Users/sneha/Desktop/ciliaNov11/labeled_cp_im'
CHANNEL_DICT = {'01': 'NucleusOverlay', '02': 'CiliaOverlay', '03': 'CentrioleOverlay'}
CENTRIOLE = None  # True or None
################################# TO CHANGE #################################


# Makes paths for us to be able to find init imgs / for images to go
def make_paths(num, channel, label):
    """TODO make_paths docstring."""
    path = (
        (OUTPUT_PATH if label else IM_CSV_DIR_PATH) +
        'CombinedIm' +
        f"{num:04}" +
        ('_LABELED.tiff' if label else '.tiff')
    )
    print(path)
    return path


# Makes list of coordinates for each df
def helper_make_lists(im_num, grouped, changed_num=None):
    """TODO helper_make_lists docstring."""
    im_df = grouped.get_group(im_num)
    im_df.drop('ImageNumber', axis=1, inplace=True)
    new_list = im_df.values.tolist()
    if changed_num:
        im_df_num = im_df[changed_num]
        new_li_num = im_df_num.values.tolist()
        return new_list, new_li_num
    return new_list


# Makes lists of coordinates
def make_lists(im_num, grouped_cell, grouped_cilia, grouped_centriole):
    """TODO make_lists docstring."""
    cell_list = helper_make_lists(im_num, grouped_cell)
    cilia_list, cilia_list_num = helper_make_lists(im_num, grouped_cilia, 'Cilia')
    centriole_list = None

    centriole_list = grouped_centriole and helper_make_lists(im_num, grouped_centriole)

    return cell_list, cilia_list, centriole_list, cilia_list_num


# Labels image
def label_im(coordinate_list, coordinates_2, im, num, channel, li_num):
    """TODO label_im docstring."""
    img = Image.open(im)

    # Writes number onto image at center
    for i, val in enumerate(coordinate_list):
        x_coord = val[0]
        y_coord = val[1]
        d = ImageDraw.Draw(img)
        write_num = str(i+1)
        d.text((x_coord, y_coord), write_num, fill=(255))

    for i, val in enumerate(coordinates_2):
        x_coord = val[0]
        y_coord = val[1]
        d = ImageDraw.Draw(img)
        write_num = str(int(li_num[i]))
        d.text((x_coord, y_coord), write_num, fill=(100))

    path = make_paths(num, channel, True)
    img.save(path)


def batch_script():
    """TODO add batch_script docstring."""
    # Columns we need to keep
    cilia_fields = ['ImageNumber', 'Cilia', 'Location_Center_X', 'Location_Center_Y']
    nuclei_fields = ['ImageNumber', 'Location_Center_X', 'Location_Center_Y']
    centriole_fields = ['ImageNumber', 'Centriole', 'Location_Center_X', 'Location_Center_Y']
    # Reads csv and groups by the im num
    cell_df = pd.read_csv(
        CSV_FOLDER + '/MyExpt_Nucleus.csv',
        skipinitialspace=True,
        usecols=nuclei_fields
    )
    num_im = cell_df.ImageNumber.iat[-1]
    grouped_cell = cell_df.groupby(['ImageNumber'])

    grouped_cilia = pd.read_csv(
        CSV_FOLDER + '/MyExpt_Cilia.csv',
        skipinitialspace=True,
        usecols=cilia_fields
    ).groupby(['ImageNumber'])

    grouped_centriole = None
    # If we have centriole images, read them too. If not, keep the grouped as none (so that we can
    # pass it into the next func)
    grouped_centriole = CENTRIOLE and pd.read_csv(
        CSV_FOLDER + '/MyExpt_Centriole.csv',
        skipinitialspace=True,
        usecols=centriole_fields
    ).groupby(['ImageNumber'])

    # Iterate through the images. Make list of nuclei/cilia/centrioles, then make paths for our
    # current image & label+save image.
    for num in range(1, num_im+1):
        cell_list, cilia_list, centriole_list, cilia_list_num = \
            make_lists(num, grouped_cell, grouped_cilia, grouped_centriole)

        im_path = make_paths(num, '01', False)
        label_im(cell_list, cilia_list, im_path, num, '01', cilia_list_num)

        if CENTRIOLE:
            im_path_centriole = make_paths(num, '03', False)
            label_im(centriole_list, im_path_centriole, num, '03')


def main():
    """TODO main docstring."""
    batch_script()


if __name__ == "__main__":
    main()
