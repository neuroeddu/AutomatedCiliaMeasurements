"""TODO vis_cilia_to_cent docstring."""
import pandas as pd
from PIL import Image, ImageDraw

################################# TO CHANGE #################################
cell_csv_path = '/Users/sneha/Desktop/ciliaNov22/spreadsheets_im_output/MyExpt_Nucleus.csv'
cilia_csv_path = '/Users/sneha/Desktop/ciliaNov22/spreadsheets_im_output/MyExpt_Cilia.csv'
centriole_csv_path = '/Users/sneha/Desktop/ciliaNov22/spreadsheets_im_output/MyExpt_Centriole.csv'
im_csv_dir_path = '/Users/sneha/Desktop/ciliaNov22/im_output/'
c2c_output_path = '/Users/sneha/Desktop/ciliaNov22/cent2cilia.csv'
output_im_dir_path = '/Users/sneha/Desktop/ciliaNov/visualizer/'
channel_dict = {'01': 'NucleusOverlay', '02': 'CiliaOverlay', '03': 'CentrioleOverlay'}
################################# TO CHANGE #################################


def draw_things(
    cur_nuc, visited_nuc, cur_cent, visited_cent, img, new_list_cell, new_list_centriole
):
    """TODO draw_things docstring."""
    if int(cur_nuc) not in visited_nuc and cur_cent not in visited_cent:
        nuc_x = new_list_cell[int(cur_nuc)-1][0]
        nuc_y = new_list_cell[int(cur_nuc)-1][1]

        try:
            cent_x = new_list_centriole[int(cur_cent)-1][0]
        except Exception:
            raise
        cent_y = new_list_centriole[int(cur_cent)-1][1]
        d = ImageDraw.Draw(img)
        d.text((int(nuc_x), int(nuc_y)), str(cur_nuc), fill=(255, 255, 255, 255))
        d.text((int(cent_x), int(cent_y)), str(cur_cent), fill=(255, 0, 0, 255))
        line_xy = [(int(nuc_x), int(nuc_y)), (int(cent_x), int(cent_y))]
        d.line(line_xy, fill=(255, 255, 255, 255))
        visited_cent.add(cur_cent)
        visited_nuc.add(int(cur_nuc))
    return visited_nuc, visited_cent


fields = ['ImageNumber', 'Location_Center_X', 'Location_Center_Y']

cell_df = pd.read_csv(cell_csv_path, skipinitialspace=True, usecols=fields)
num_im = cell_df.ImageNumber.iat[-1]
grouped_cell = cell_df.groupby(['ImageNumber'])

cilia_df = pd.read_csv(cilia_csv_path, skipinitialspace=True, usecols=fields)
grouped_cilia = cilia_df.groupby(['ImageNumber'])

fields_c2c = ['ImageNumber', 'Cilia', 'Centriole']
associate_df = pd.read_csv(c2c_output_path, skipinitialspace=True, usecols=fields_c2c)
grouped_associates = associate_df.groupby(['ImageNumber'])


centriole_df = pd.read_csv(centriole_csv_path, skipinitialspace=True, usecols=fields)
grouped_centriole = centriole_df.groupby(['ImageNumber'])

for num in range(1, num_im+1):

    df_centriole = grouped_centriole.get_group(num)
    df_centriole.drop('ImageNumber', axis=1, inplace=True)
    new_list_centriole = df_centriole.values.tolist()

    df_cilia = grouped_cilia.get_group(num)
    df_cilia.drop('ImageNumber', axis=1, inplace=True)
    new_list_cilia = df_cilia.values.tolist()

    df_associates = grouped_associates.get_group(num)
    df_associates.drop('ImageNumber', axis=1, inplace=True)
    new_list_associates = df_associates.values.tolist()

    path = (im_csv_dir_path + 'CentrioleOverlay' + f"{num:04}" + '.tiff')
    img = Image.open(path)

    for x, thing in enumerate(new_list_associates):
        visited_nuc = set()
        visited_cent = set()
        cur_nuc = thing[0]
        cur_cent = thing[1]

        if cur_nuc == 195:
            print()
        visited_nuc, visited_cent = draw_things(
            cur_nuc,
            visited_nuc,
            float(cur_cent),
            visited_cent,
            img,
            new_list_cilia,
            new_list_centriole,
        )

    pathnew = (
        '/Users/sneha/Desktop/ciliaNov22/labeled_im_FULL/cilia_to_cent/cilia_to_cent_' +
        f"{num:04}" +
        '.tiff'
    )
    img.save(pathnew)
