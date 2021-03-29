import csv
import pandas as pd
from math import sqrt

################################# TO CHANGE #################################
cell_csv_path='/Users/sneha/Desktop/mni/cilia-output/MyExpt_Nuclei.csv'
cilia_csv_path='/Users/sneha/Desktop/mni/cilia-output/MyExpt_Cilia.csv'
output_csv_dir_path='/Users/sneha/Desktop/mni/pkg-cilia-v3/csv_centers_output'
################################# TO CHANGE #################################

# does the analysis for multiple images 
def batch_script():
    fields = ['ImageNumber', 'Location_Center_X', 'Location_Center_Y']
    cell_df = pd.read_csv(cell_csv_path, skipinitialspace=True, usecols=fields)
    num_im = cell_df.ImageNumber.iat[-1]
    grouped_cell = cell_df.groupby(['ImageNumber'])
    cilia_df = pd.read_csv(cilia_csv_path, skipinitialspace=True, usecols=fields)
    grouped_cilia = cilia_df.groupby(['ImageNumber'])

    for num in range(1, num_im+1):
        cell_list, cilia_list = make_lists(num, grouped_cell, grouped_cilia)
        cilia_to_cell = which_cilia_closest(cell_list, cilia_list) 
        cilia_to_cell_no_dups = remove_dups_dict(cilia_to_cell)
        print(cilia_to_cell_no_dups)
        output_path=output_csv_dir_path + '/' + 'im_' + str(num) + '.csv'
        convert_dict_to_csv(cilia_to_cell_no_dups, output_path)

# makes df, segregates by im, returns as li
def helper_make_lists(im_num, grouped):
    im_df = grouped.get_group(im_num) 
    print(im_df)
    im_df.drop('ImageNumber', axis=1, inplace=True)
    new_list = im_df.values.tolist()
    return new_list

# makes lists
def make_lists(im_num, grouped_cell, grouped_cilia): 
    cell_list = helper_make_lists(im_num, grouped_cell)
    cilia_list = helper_make_lists(im_num, grouped_cilia)

    return cell_list, cilia_list

# finds out which cilia is closest to which cell assuming cutoff passed in (or none if not) & 1:1 cell:cilia relationship
def which_cilia_closest(cell_list, cilia_list, cutoff = float('inf')):
    cell_to_cilia = [
        {
            "cilia": None, # The index of the current closest cilia
            "cilia_tried": set() # The indicies of previously tried cilia that shouldn't be tried again
        }
        for cell in range(len(cell_list))
    ]
    cilia_to_cell = [
        {
            "path_length": float('inf'), # The length of the shortest path
            "cell": None # The index of the cell to which the shortest path corresponds
        }
        for cilia in cilia_list
    ]

    updated_cilia = True
    
    while (updated_cilia): # while cilia are being updated, calculate lengths and see if the cilia should be added
        updated_cilia = False
        for i, cell in enumerate(cell_list):
            x_cell = cell[0]
            y_cell = cell[1]
            for j, cilia in enumerate(cilia_list):
                x_cilia = cilia[0]
                y_cilia = cilia[1]
                result = sqrt(pow((x_cilia - x_cell), 2) + pow((y_cilia - y_cell), 2))
                
                if result > cutoff or j in cell_to_cilia[i]["cilia_tried"] or result >= cilia_to_cell[j]["path_length"]:
                    continue

                add_cilia(cell_to_cilia, cilia_to_cell, result, i, j)
                updated_cilia = True

    return cilia_to_cell

# associate a cell with a cilia 
def add_cilia(cell_to_cilia, cilia_to_cell, result, cell, cilia):
    if cilia_to_cell[cilia]["cell"] == None:
        cilia_to_cell[cilia]["cell"] = cell+1
        cilia_to_cell[cilia]["path_length"] = result
        cell_to_cilia[cell]["cilia"] = cilia
        return;
    
    else:
        old_cell = cilia_to_cell[cilia]["cell"]
        cilia_to_cell[cilia]["cell"] = cell+1
        cilia_to_cell[cilia]["path_length"] = result
        cell_to_cilia[cell]["cilia"] = cilia
        cell_to_cilia[old_cell]["cilia"] = None
        cell_to_cilia[old_cell]["cilia_tried"].add(cilia)

# convert 
# TODO put it all into one csv
def convert_dict_to_csv(cilia_to_cell, output_path):
    df = pd.DataFrame.from_dict(cilia_to_cell)
    df.index = df.index + 1
    result = df.to_csv(path_or_buf=output_path, header=["PathLength", "Nuclei"], index_label="Cilia")

# remove duplicates from the dictionary to ensure 1:1 relationship
def remove_dups_dict(cilia_to_cell):

    cell_to_celia_visitation_dict = {}
    for cilia_index, cilia in enumerate(cilia_to_cell):
        if cilia["cell"] in cell_to_celia_visitation_dict: # if cell alr in visited list of cells
            old_cilia_index = cell_to_celia_visitation_dict[cilia["cell"]]
            old_cilia = cilia_to_cell[old_cilia_index]
            if cilia["path_length"] < old_cilia["path_length"]: # if cur path length < path length of prev 
                old_cilia["path_length"] = 0.00 # set the prev one's path length to 0 and cell to none
                old_cilia["cell"] = -1
                cell_to_celia_visitation_dict[cilia["cell"]] = cilia_index
            else: # if path length of prev is better / same, keep prev 
                cilia["path_length"] = 0.00
                cilia["cell"] = -1
        else:
            cell_to_celia_visitation_dict[cilia["cell"]] = cilia_index
    
    return cilia_to_cell

def main(): 
    batch_script()

if __name__ == "__main__":
    main()