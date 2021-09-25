import csv
import pandas as pd
from math import sqrt

################################# TO CHANGE #################################
cell_csv_path='/Users/sneha/Desktop/mni/cilia_09:12:2021/im_output/MyExpt_Nucleus.csv'
cilia_csv_path='/Users/sneha/Desktop/mni/cilia_09:12:2021/im_output/MyExpt_Cilia.csv'
centriole_csv_path='/Users/sneha/Desktop/mni/cilia_09:12:2021/im_output/MyExpt_Centriole.csv'
output_csv_dir_path='/Users/sneha/Desktop/mni/cilia_09:12:2021/csv_centers'
################################# TO CHANGE #################################

def helper_make_lists(im_num, grouped):
    im_df = grouped.get_group(im_num) 
    im_df.drop('ImageNumber', axis=1, inplace=True)
    return im_df.values.tolist()

# makes lists
def make_lists(im_num, grouped_cell, grouped_centriole): 
    cell_list = helper_make_lists(im_num, grouped_cell)
    centriole_list = helper_make_lists(im_num, grouped_centriole)

    return cell_list, centriole_list

# finds out which cilia is closest to which cell assuming cutoff passed in (or none if not) & 1:1 cell:cilia relationship
def which_cilia_closest(cell_list, cilia_list, cutoff = float('inf')):
    cell_to_cilia = [
        {
            "cilia": None, # The index of the current closest cilia
            "cilia_tried": set() # The indicies of previously tried cilia that shouldn't be tried again
        }
        for _ in cell_list
    ]
    cilia_to_cell = [
        {
            "path_length": float('inf'), # The length of the shortest path
            "cell": None # The index of the cell to which the shortest path corresponds
        }
        for _ in cilia_list
    ]

    updated_cilia = True
    
    while updated_cilia: # while cilia are being updated, calculate lengths and see if the cilia should be added
        updated_cilia = False
        for i, cell in enumerate(cell_list):
            x_cell, y_cell = cell
            for j, cilia in enumerate(cilia_list):
                x_cilia, y_cilia = cilia
                result = sqrt(pow((x_cilia - x_cell), 2) + pow((y_cilia - y_cell), 2))
                
                if result > cutoff or j in cell_to_cilia[i]["cilia_tried"] or result >= cilia_to_cell[j]["path_length"]:
                    continue

                add_cilia(cell_to_cilia, cilia_to_cell, result, i, j)
                updated_cilia = True

    return cilia_to_cell

# associate a cell with a cilia 
def add_cilia(cell_to_cilia, cilia_to_cell, result, cell, cilia):

    old_cell = cilia_to_cell[cilia]["cell"]
    cilia_to_cell[cilia]["cell"] = cell+1
    cilia_to_cell[cilia]["path_length"] = result
    cell_to_cilia[cell]["cilia"] = cilia

    if old_cell is None:
        return
    
    cell_to_cilia[old_cell]["cilia"] = None
    cell_to_cilia[old_cell]["cilia_tried"].add(cilia)

# remove duplicates from the dictionary to ensure 1:1 relationship
def remove_dups_dict(cilia_to_cell):

    cell_to_cilia_visitation_dict = {}
    for cilia_index, cilia in enumerate(cilia_to_cell):

        if cilia["cell"] in cell_to_cilia_visitation_dict: # if cell alr in visited list of cells
            old_cilia_index = cell_to_cilia_visitation_dict[cilia["cell"]]
            old_cilia = cilia_to_cell[old_cilia_index]

            if cilia["path_length"] < old_cilia["path_length"]: # if cur path length < path length of prev 
                old_cilia["path_length"] = 0.00 # set the prev one's path length to 0 and cell to none
                old_cilia["cell"] = -1
                cell_to_cilia_visitation_dict[cilia["cell"]] = cilia_index

            else: # if path length of prev is better / same, keep prev 
                cilia["path_length"] = 0.00
                cilia["cell"] = -1
                
        else:
            cell_to_cilia_visitation_dict[cilia["cell"]] = cilia_index
    
    return cilia_to_cell

# remove some duplicates to ensure 1:1 or 2 relation
def remove_some_dups_dict(cilia_to_cell, cell_list):
    cell_to_cilia = [
    {
        "cilia1": None, # The index of the current closest cilia
        "cilia2": None # The index of the second closest cilia
    }
    for cell in range(len(cell_list))
    ]


    for cilia_index, cilia in enumerate(cilia_to_cell):
        cur_cell=cilia["cell"]-1
        # case 1: cilia1==none, put it in cilia1
        if not cell_to_cilia[cur_cell]["cilia1"]:
            cell_to_cilia[cur_cell]["cilia1"] = cilia_index
        # case 2: cilia2==none, put closest one in 1 and put second in 2
        elif not cell_to_cilia[cur_cell]["cilia2"]:
            old_cilia_index = cell_to_cilia[cur_cell]["cilia1"]
            old_cilia = cilia_to_cell[old_cilia_index]
            if cilia["path_length"] < old_cilia["path_length"]:
                cell_to_cilia[cur_cell]["cilia1"] = cilia_index
                cell_to_cilia[cur_cell]["cilia2"] = old_cilia_index

            else:
                cell_to_cilia[cur_cell]["cilia2"] = cilia_index

        # case 3: cilia1 and cilia2 full, check whether cilia2 path length> new cilia
        else:  
            old_cilia_index = cell_to_cilia[cur_cell]["cilia2"]
            old_cilia = cilia_to_cell[old_cilia_index]
            # case 3a: cilia2 path length< new cilia, new cilia's path length n cell are 0 
            if old_cilia["path_length"] < cilia["path_length"]:
                cilia["path_length"] = 0.00
                cilia["cell"] = -1
            
            # case 3b: cilia2 path length>new cilia, cilia2's path length/cell r 0, check whether new cilia pl>cilia1
            else:
                old_cilia["path_length"] = 0.00 # set the prev one's path length to 0 and cell to none
                old_cilia["cell"] = -1
                old_cilia_index = cell_to_cilia[cur_cell]["cilia1"]
                old_cilia = cilia_to_cell[old_cilia_index]

                # case 3b1: new cilia pl>cilia1, new cilia in cilia2
                if old_cilia["path_length"] < cilia["path_length"]:
                    cell_to_cilia[cur_cell]["cilia1"] = old_cilia_index
                    cell_to_cilia[cur_cell]["cilia2"] = cilia_index

                # case 3b2: new cilia pl<cilia1, new cilia in cilia1 and old in cilia2
                else:
                    cell_to_cilia[cur_cell]["cilia1"] = cilia_index
                    cell_to_cilia[cur_cell]["cilia2"] = old_cilia_index
        
    return cilia_to_cell

def combine_dicts(centriole_to_cell_no_dups, centriole_to_cilia_no_dups):

    c2c_output = [
        {
            'path_length_cell': float('inf'),
            'cell': None,
            'path_length_cilia': float('inf'),
            'cilia': None
        }
        for num in range(len(centriole_to_cell_no_dups))
    ]

    for num in range(len(centriole_to_cell_no_dups)):
        c2c_output[num]['path_length_cell'] = centriole_to_cell_no_dups[num]['path_length']
        c2c_output[num]['cell'] = centriole_to_cell_no_dups[num]['cell']
        c2c_output[num]['path_length_cilia'] = centriole_to_cilia_no_dups[num]['path_length']
        c2c_output[num]['cilia'] = centriole_to_cilia_no_dups[num]['cell']

    return c2c_output
        
# convert 
# TODO put it all into one csv
def convert_dict_to_csv(cilia_to_cell, output_path, num, cilia_id=False):
    df = pd.DataFrame.from_dict(cilia_to_cell)
    df.index = df.index + 1
    result = df.to_csv(path_or_buf=output_path, header=["PathLengthCell", "Nucleus", "PathLengthCilia", "Cilia"], index_label="Centriole")
  

def main(): 
    fields = ['ImageNumber', 'Location_Center_X', 'Location_Center_Y']
    cell_df = pd.read_csv(cell_csv_path, skipinitialspace=True, usecols=fields)
    num_im = cell_df.ImageNumber.iat[-1]
    grouped_cell = cell_df.groupby(['ImageNumber'])
    centriole_df = pd.read_csv(centriole_csv_path, skipinitialspace=True, usecols=fields)
    grouped_centriole = centriole_df.groupby(['ImageNumber'])
    cilia_df = pd.read_csv(cilia_csv_path, skipinitialspace=True, usecols=fields)
    grouped_cilia = cilia_df.groupby(['ImageNumber'])

    for num in range(1, num_im+1):
        cell_list, centriole_list = make_lists(num, grouped_cell, grouped_centriole)
        centriole_to_cell = which_cilia_closest(cell_list, centriole_list) 
        centriole_to_cell_no_dups = remove_some_dups_dict(centriole_to_cell, cell_list)
        
        cilia_list, centriole_list = make_lists(num, grouped_cilia, grouped_centriole)
        centriole_to_cilia = which_cilia_closest(cilia_list, centriole_list) 
        centriole_to_cilia_no_dups = remove_dups_dict(centriole_to_cilia)
        
        output_path=output_csv_dir_path + '/im_' + str(num) + '.csv'
        c2c_output=combine_dicts(centriole_to_cell_no_dups, centriole_to_cilia_no_dups)
        convert_dict_to_csv(c2c_output, output_path, num)


if __name__ == "__main__":
    main()