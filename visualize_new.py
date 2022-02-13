# visualize pairs
import pandas as pd
import csv
from PIL import Image, ImageDraw, ImageFont

################################# TO CHANGE #################################
CSV_FOLDER='/Users/sneha/Desktop/ciliaJan22/spreadsheets_im_output'
IM_CSV_DIR_PATH='/Users/sneha/Desktop/ciliaJan13/combinedim/'
C2C_OUTPUT_PATH='/Users/sneha/Desktop/ciliaJan22/c2coutputnone/c2coutput.csv'
OUTPUT_IM_DIR_PATH='/Users/sneha/Desktop/ciliaJan22/visualizernone/'
CHANNEL_DICT={'01': 'NucleusOverlay', '02': 'CiliaOverlay', '03': 'CentrioleOverlay'}
################################# TO CHANGE #################################

def draw_things(cur_nuc, visited_nuc, cur_cent, visited_cent, img, new_list_cell, new_list_centriole):
    #if int(cur_nuc) not in visited_nuc and cur_cent not in visited_cent:
    nuc_x = new_list_cell[int(cur_nuc)-1][0]
    nuc_y = new_list_cell[int(cur_nuc)-1][1]

    try:
        cent_x=new_list_centriole[int(cur_cent)-1][0]
    except:
        raise
    cent_y=new_list_centriole[int(cur_cent)-1][1]
    d = ImageDraw.Draw(img)
    d.text((int(nuc_x), int(nuc_y)), str(cur_nuc), fill=(0,100,0))
    d.text((int(cent_x), int(cent_y)), str(cur_cent), fill=(128,0,0))
    line_xy = [(int(nuc_x), int(nuc_y)), (int(cent_x), int(cent_y))]
    d.line(line_xy, fill=(255,255,255,255))
    visited_cent.add(cur_cent)
    visited_nuc.add(int(cur_nuc))
    return visited_nuc, visited_cent

fields = ['ImageNumber', 'Location_Center_X', 'Location_Center_Y']

cell_df = pd.read_csv(CSV_FOLDER+'/MyExpt_Nucleus.csv', skipinitialspace=True, usecols=fields)
num_im = cell_df.ImageNumber.iat[-1]
grouped_cell = cell_df.groupby(['ImageNumber'])

cilia_df = pd.read_csv(CSV_FOLDER+'/MyExpt_Cilia.csv', skipinitialspace=True, usecols=fields)
grouped_cilia = cilia_df.groupby(['ImageNumber'])

fields_c2c = ['ImageNumber','Nucleus', 'Centriole']
associate_df = pd.read_csv(C2C_OUTPUT_PATH, skipinitialspace=True, usecols=fields_c2c)
grouped_associates= associate_df.groupby(['ImageNumber'])


centriole_df = pd.read_csv(CSV_FOLDER+'/MyExpt_Centriole.csv', skipinitialspace=True, usecols=fields)
grouped_centriole = centriole_df.groupby(['ImageNumber'])

for num in range(1, num_im+1):
    df_cell = grouped_cell.get_group(num) 
    df_cell.drop('ImageNumber', axis=1, inplace=True)
    new_list_cell = df_cell.values.tolist()

    df_centriole = grouped_centriole.get_group(num) 
    df_centriole.drop('ImageNumber', axis=1, inplace=True)
    new_list_centriole = df_centriole.values.tolist()

    df_cilia = grouped_cilia.get_group(num) 
    df_cilia.drop('ImageNumber', axis=1, inplace=True)
    new_list_cilia = df_cilia.values.tolist()

    df_associates = grouped_associates.get_group(num) 
    df_associates.drop('ImageNumber', axis=1, inplace=True)
    new_list_associates = df_associates.values.tolist()

    path=(IM_CSV_DIR_PATH + 'CombinedIm' + f"{num:04}" + '.tiff')
    img = Image.open(path)

    for x, thing in enumerate(new_list_associates):
        visited_nuc=set()
        visited_cent=set()
        cur_nuc=thing[0]
        cur_cent=thing[1]
        cur_cent = cur_cent.strip('[')
        cur_cent = cur_cent.strip(']')
        if not 'nan' in cur_cent:
            if ',' in cur_cent:
                split_cent=cur_cent.split(', ')
                visited_nuc, visited_cent = draw_things(cur_nuc, visited_nuc, float(split_cent[0]), visited_cent, img, new_list_cell, new_list_centriole)
                visited_nuc, visited_cent = draw_things(cur_nuc, visited_nuc, float(split_cent[1]), visited_cent, img, new_list_cell, new_list_centriole)
            else:
                visited_nuc, visited_cent = draw_things(cur_nuc, visited_nuc, float(cur_cent), visited_cent, img, new_list_cell, new_list_centriole)

    pathnew='/Users/sneha/Desktop/ciliaJan22/visualizer25/combined_'+ f"{num:04}" + '.tiff'
    img.save(pathnew)