import pandas as pd
import csv
import sys
from PIL import Image, ImageDraw, ImageFont

################################# TO CHANGE #################################
cell_csv_path='/Users/sneha/Desktop/mni/cilia-output/MyExpt_Cell.csv'
cilia_csv_path='/Users/sneha/Desktop/mni/cilia-output/MyExpt_Cilia.csv'
im_csv_dir_path='/Users/sneha/Desktop/mni/cilia-output/'
center_to_center_fol_path='/Users/sneha/Desktop/mni/csv_output_ex'
output_im_dir_path='/Users/sneha/Desktop/mni/'
num_im=3
################################# TO CHANGE #################################

# TODO : map which cilia to which cell -- line 

def batch_script(): # runs methods for all images
    for num in range(1, num_im+1):
        cell_list, cilia_list = make_lists(num)
        cell_channel = '01'
        cilia_channel = '02'
        im_path_cell=make_paths(num, '01', False)
        im_path_cilia=make_paths(num, '02', False)
        label_im(cell_list, im_path_cell, num, '01')
        label_im(cilia_list, im_path_cilia, num, '02')

def make_paths(num, channel, label): #makes paths for us to be able to find init imgs / for images to go 
    if label:
        path = output_im_dir_path + '210115_Cortical_NPC_' + str(num) + '_ch' + channel + '_LABELED.tiff'
    
    else:
        path = im_csv_dir_path + '210115_Cortical_NPC_' + str(num) + '_ch' + channel + '.tiff'

    return path

def helper_make_lists(csv_path, im_num): # makes list grouped by im 
    fields = ['ImageNumber', 'Location_Center_X', 'Location_Center_Y']
    df = pd.read_csv(csv_path, skipinitialspace=True, usecols=fields)
    grouped = df.groupby(df.ImageNumber)
    im_df = grouped.get_group(im_num) 
    im_df.drop('ImageNumber', axis=1, inplace=True)
    new_list = im_df.values.tolist()
    return new_list

# makes lists
def make_lists(im_num): 
    cell_list = helper_make_lists(cell_csv_path, im_num)
    cilia_list = helper_make_lists(cilia_csv_path, im_num)
    associate_list = helper_c2c_make_list(im_num)
    return cell_list, cilia_list, associate_list

def helper_c2c_make_list(im_num):
    csv_path = center_to_center_fol_path + '/im_' + im_num 
    fields = ['Cilia', 'Cell']
    df = pd.read_csv(csv_path, skipinitialspace=True, usecols=fields)
    new_list = df.values.tolist()
    return new_list

# labels image
def label_im(coordinate_list, im, num, channel):
    img = Image.open(im)
    for i, val in enumerate(coordinate_list):
        x_coord = val[0]
        y_coord = val[1]
        d = ImageDraw.Draw(img)
        write_num = str(i+1)
        d.text((x_coord, y_coord), write_num, fill=(255,255,255,255))
    
    path = make_paths(num, channel, True)
    img.save(path)

# cilia to cell line
# takes in two lists of coords, and cell im
def all_label(cell_list, cilia_list, associate_list, im, num, channel):
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
        d.text((x_coord, y_coord), write_num, fill=(255,255,255,255))
    
    for i, val in enumerate(associate_list):
        cilia_num = val[0]
        cell_num = val[1]
        
    path = make_paths(num, channel, True)
    img.save(path)

def main(): 
    batch_script()

if __name__ == "__main__":
    main()
            