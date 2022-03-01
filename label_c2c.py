import pandas as pd
from PIL import Image, ImageDraw

################################# TO CHANGE #################################
CSV_FOLDER='/Users/sneha/Desktop/ciliaJan22/spreadsheets_im_output'
IM_CSV_DIR_PATH='/Users/sneha/Desktop/ciliaJan13/combinedim/'
INPUT_IM_NUM=None # change to number if we want a specific number of input images 
C2C_OUTPUT_PATH='/Users/sneha/Desktop/ciliaJan22/c2coutputnone/c2coutput.csv'
OUTPUT_IM_DIR_PATH='/Users/sneha/Desktop/ciliaJan22/visualizernone/'
CHANNEL_DICT={'01': 'NucleusOverlay', '02': 'CiliaOverlay', '03': 'CentrioleOverlay'}
################################# TO CHANGE #################################

def draw_things(cur_nuc, cur_cent, img, new_list_cell, new_list_centriole):

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

def main():
    # Load data
    fields = ['ImageNumber', 'Location_Center_X', 'Location_Center_Y']

    cell_df = pd.read_csv(CSV_FOLDER+'/MyExpt_Nucleus.csv', skipinitialspace=True, usecols=fields)
    grouped_cell = cell_df.groupby(['ImageNumber'])

    cilia_df = pd.read_csv(CSV_FOLDER+'/MyExpt_Cilia.csv', skipinitialspace=True, usecols=fields)
    grouped_cilia = cilia_df.groupby(['ImageNumber'])

    fields_c2c = ['ImageNumber','Nucleus', 'Centriole']
    associate_df = pd.read_csv(C2C_OUTPUT_PATH, skipinitialspace=True, usecols=fields_c2c)
    grouped_associates= associate_df.groupby(['ImageNumber'])


    centriole_df = pd.read_csv(CSV_FOLDER+'/MyExpt_Centriole.csv', skipinitialspace=True, usecols=fields)
    grouped_centriole = centriole_df.groupby(['ImageNumber'])

    # Get number of images, either from the number inputted or from the total number of images
    images=None
    images=INPUT_IM_NUM and cell_df.ImageNumber.iat[-1]

    for num in range(1, images+1):
        # Load grouped data into lists
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

        # Load combined images
        path=(IM_CSV_DIR_PATH + 'CombinedIm' + f"{num:04}" + '.tiff')
        img = Image.open(path)

        for _, associate in enumerate(new_list_associates):
            cur_nuc=associate[0]
            cur_cent=associate[1]

            # Strip list [] from centriole
            cur_cent = cur_cent.strip('[')
            cur_cent = cur_cent.strip(']')

            # Paint centriole(s) if they are there
            if not 'nan' in cur_cent:

                # If multiple centriole, paint each
                if ',' in cur_cent:
                    split_cent=cur_cent.split(', ')
                    draw_things(cur_nuc, float(split_cent[0]), img, new_list_cell, new_list_centriole)
                    draw_things(cur_nuc, float(split_cent[1]), img, new_list_cell, new_list_centriole)
                
                # If single centriole, paint
                else:
                    draw_things(cur_nuc, float(cur_cent), img, new_list_cell, new_list_centriole)

            cur_cilia=associate[2]

            # Paint cilia if it is there 
            if cur_cilia>0:
                draw_things(cur_nuc, float(cur_cilia), img, new_list_cell, new_list_cilia)

        # Save image
        new_path=OUTPUT_IM_DIR_PATH + 'COMBINED_LABEL_'+ f"{num:04}" + '.tiff'
        img.save(new_path)