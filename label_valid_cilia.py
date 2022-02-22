# Labels images for a particular organelle

import pandas as pd
from PIL import Image, ImageDraw

################################# TO CHANGE #################################
MEASUREMENTS_CSV='/Users/sneha/Desktop/ciliaJan22/spreadsheets_im_output/MyExpt_Cilia.csv'
VALID_CSV='/Users/sneha/Desktop/c2coutput/threshold_none/new_cilia.csv' # ONLY if cent or cilia, otherwise None
IM_CSV_DIR_PATH='/Users/sneha/Desktop/ciliaJan22/im_output'
CHANNEL_DICT={'01': 'NucleusOverlay', '02': 'CiliaOverlay', '03': 'CentrioleOverlay'}
CHANNEL='02'
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
    
    path = make_paths(num, True, CHANNEL)
    img.save(path)

def make_paths(num, channel, label): 
    path = IM_CSV_DIR_PATH + CHANNEL_DICT[channel] + f"{num:04}" + ('_LABELED.tiff' if label else '.tiff')
    return path
    
def main():
    measurements_df = pd.read_csv(MEASUREMENTS_CSV, skipinitialspace=True, usecols=['ImageNumber', 'ObjectNumber', 'Location_Center_X', 'Location_Center_Y'])
    num_im = measurements_df.ImageNumber.iat[-1]

    # if not all measurements are valid, merge
    if VALID_CSV:
        valid_df = pd.read_csv(VALID_CSV, skipinitialspace=True)
        valid_df = valid_df.rename(columns={'0': 'ImageNumber', '1': 'ObjectNumber'})
        measurements_df = valid_df.merge(measurements_df, on=['ImageNumber', 'ObjectNumber'])

    grouped_cilia = measurements_df.groupby(['ImageNumber'])

    for num in range(1, num_im+1):
        # Get list of coords to plot
        coords_df = grouped_cilia.get_group(num) 
        coords_df.drop(['ImageNumber', 'ObjectNumber'], axis=1, inplace=True)
        coords_list=coords_df.values.tolist()

        # Get path and 
        im_path=make_paths(num, False, CHANNEL)
        label_im(coords_list, im_path, num)

if __name__ == "__main__":
    main()




    

