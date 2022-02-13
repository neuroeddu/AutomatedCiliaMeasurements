import pandas as pd
import csv
from PIL import Image, ImageDraw, ImageFont

################################# TO CHANGE #################################
CILIA_INFO='/Users/sneha/Desktop/ciliaJan22/spreadsheets_im_output/MyExpt_Cilia.csv'
VALID_CILIA='/Users/sneha/Desktop/c2coutput/threshold_none/new_cilia.csv'
IM_CSV_DIR_PATH='/Users/sneha/Desktop/ciliaJan22/im_output'
################################# TO CHANGE #################################

def label_im(coordinate_list, im, num):
    img = Image.open(im)

    # Writes number onto image at center 
    for i, val in enumerate(coordinate_list):
        x_coord = val[1]
        y_coord = val[2]
        d = ImageDraw.Draw(img)
        write_num = str(i+1)
        d.text((x_coord, y_coord), write_num, fill=(255,0,0,255))
    
    path = make_paths(num, True)
    img.save(path)

def make_paths(num, label): 
    path = IM_CSV_DIR_PATH + '/CiliaOverlay' + f"{num:04}" + ('_LABELED.tiff' if label else '.tiff')
    return path

cilia_info = pd.read_csv(CILIA_INFO, skipinitialspace=True, usecols=['ImageNumber', 'ObjectNumber', 'Location_Center_X', 'Location_Center_Y'])
valid_cilia = pd.read_csv(VALID_CILIA, skipinitialspace=True)
num_im = cilia_info.ImageNumber.iat[-1]

valid_cilia = valid_cilia.rename(columns={'0': 'ImageNumber', '1': 'ObjectNumber'})
full_cilia_df = valid_cilia.merge(cilia_info, on=['ImageNumber', 'ObjectNumber'])

grouped_cilia = full_cilia_df.groupby(['ImageNumber'])

for num in range(1, num_im+1):
    coords_df = grouped_cilia.get_group(num) 
    coords_df.drop(['ImageNumber', 'ObjectNumber'], axis=1, inplace=True)
    coords_list=coords_df.values.tolist()

    im_path=make_paths(num, False)
    label_im(coords_list, im_path, num)






    

