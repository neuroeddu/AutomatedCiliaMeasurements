import pandas as pd
import csv
from PIL import Image, ImageDraw, ImageFont

################################# TO CHANGE #################################
cell_csv_path='/Users/sneha/Desktop/mni/cilia_09:12:2021/im_output/MyExpt_Nucleus.csv'
cilia_csv_path='/Users/sneha/Desktop/mni/cilia_09:12:2021/im_output/MyExpt_Cilia.csv'
im_csv_dir_path='/Users/sneha/Desktop/mni/cilia_09:12:2021/im_output/'
center_to_center_fol_path='/Users/sneha/Desktop/mni/cilia_09:12:2021/csv_centers'
output_im_dir_path='/Users/sneha/Desktop/mni/cilia_09:12:2021/visualizer/'

################################# TO CHANGE #################################


def cilia_to_line(): # big func that calls everything else

    fields = ['ImageNumber', 'Location_Center_X', 'Location_Center_Y']
    cell_df = pd.read_csv(cell_csv_path, skipinitialspace=True, usecols=fields)
    num_im = cell_df.ImageNumber.iat[-1]
    grouped_cell = cell_df.groupby(['ImageNumber'])
    cilia_df = pd.read_csv(cilia_csv_path, skipinitialspace=True, usecols=fields)
    grouped_cilia = cilia_df.groupby(['ImageNumber'])

    for num in range(1, num_im+1):
        cell_list, cilia_list, associate_list = make_lists_c2c(num, grouped_cell, grouped_cilia)
        im_path=make_paths(num, False)
        c2c_label(cell_list, cilia_list, associate_list, im_path, num)

        
def make_paths(num, label): #makes paths for us to be able to find init imgs / for images to go 
    #path = im_csv_dir_path + 'NucleusOverlay' + f"{num:04}" + ('_LABELED_FULL.tiff' if label else '.tiff')
    #path = ((output_im_dir_path + 'NucleusOverlay' + f"{num:04}" + '_LABELED_FULL.tiff') if label else (im_csv_dir_path + 'NucleusOverlay' + f"{num:04}" + '.tiff'))
    if label:
        path = (output_im_dir_path + 'NucleusOverlay' + f"{num:04}" + '_LABELED_FULL.tiff')
    
    else: 
        path=(im_csv_dir_path + 'NucleusOverlay' + f"{num:04}" + '.tiff')
        print("here")
    return path

def make_lists_c2c(im_num, grouped_cell, grouped_cilia): 
    cell_list = helper_make_lists(im_num, grouped_cell)
    cilia_list = helper_make_lists(im_num, grouped_cilia)
    associate_list = helper_c2c_make_list(im_num)
    return cell_list, cilia_list, associate_list

def helper_c2c_make_list(im_num): # finds out what our csv path is 
    csv_path = center_to_center_fol_path + '/im_' + str(im_num) + '.csv'
    fields = ['Cilia', 'Nucleus']
    df = pd.read_csv(csv_path, skipinitialspace=True, usecols=fields)
    new_list = df.values.tolist()
    return new_list

# cilia to cell line
# takes in two lists of coords, and cell im
def c2c_label(cell_list, cilia_list, associate_list, im, num):
    img = Image.open(im)
    for i, val in enumerate(cell_list): # labels cell li pt
        x_coord = val[0]
        y_coord = val[1]
        d = ImageDraw.Draw(img)
        write_num = str(i+1)
        d.text((x_coord, y_coord), write_num, fill=(255,255,255,255))
    
    for i, val in enumerate(cilia_list): # labels cilia li pt
        x_coord = val[0]
        y_coord = val[1]
        d = ImageDraw.Draw(img)
        write_num = str(i+1)
        d.text((x_coord, y_coord), write_num, fill=(246, 71, 71, 1))
    
    for i, val in enumerate(associate_list):
        cilia_num = val[0]-1
        cell_num = val[1]-1
        if not cell_num == -2: # if cell num exists, find x and y coords of cilia and cell 
            x_cilia= cilia_list[cilia_num][0]
            y_cilia= cilia_list[cilia_num][1]
            x_cell= cell_list[cell_num][0]
            y_cell= cell_list[cell_num][1] 
            line_xy = [(x_cilia, y_cilia), (x_cell, y_cell)]
            d = ImageDraw.Draw(img)
            d.line(line_xy, fill=(255,255,255,255))

    path = make_paths(num, True)
    img.save(path)

def helper_make_lists(im_num, grouped):
    im_df = grouped.get_group(im_num) 
    im_df.drop('ImageNumber', axis=1, inplace=True)
    new_list = im_df.values.tolist()
    return new_list

# makes lists
def make_lists(im_num, grouped_cell, grouped_cilia): 
    cell_list = helper_make_lists(im_num, grouped_cell)
    cilia_list = helper_make_lists(im_num, grouped_cilia)

    return cell_list, cilia_list

def main(): 
    # batch_script()
    cilia_to_line()

if __name__ == "__main__":
    main()
            