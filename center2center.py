import csv
import pandas as pd
from math import sqrt

def make_lists(fields):
    cell_df = pd.read_csv('/Users/sneha/Desktop/mni/cilia-output/MyExpt_Cell.csv', skipinitialspace=True, usecols=fields)
    cell_list = cell_df.values.tolist()

    cilia_df = pd.read_csv('/Users/sneha/Desktop/mni/cilia-output/MyExpt_Cilia.csv', skipinitialspace=True, usecols=fields)
    cilia_list = cilia_df.values.tolist()

    return cell_list, cilia_list
    
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
    
    while (updated_cilia):
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

                 
def add_cilia(cell_to_cilia, cilia_to_cell, result, cell, cilia):
    if cilia_to_cell[cilia]["cell"] == None:
        cilia_to_cell[cilia]["cell"] = cell
        cilia_to_cell[cilia]["path_length"] = result
        cell_to_cilia[cell]["cilia"] = cilia
        return;
    
    else:
        old_cell = cilia_to_cell[cilia]["cell"]
        cilia_to_cell[cilia]["cell"] = cell
        cilia_to_cell[cilia]["path_length"] = result
        cell_to_cilia[cell]["cilia"] = cilia
        cell_to_cilia[old_cell]["cilia"] = None
        cell_to_cilia[old_cell]["cilia_tried"].add(cilia)

def convert_dict_to_csv(cilia_to_cell):
    df = pd.DataFrame.from_dict(cilia_to_cell)
    result = df.to_csv(path_or_buf='/Users/sneha/Desktop/mni/newneighbor.csv', header=["PathLength", "Cell"], index_label="Cilia")

def remove_dups_dict(cilia_to_cell):
    # what do in case of eq? ask, put method in to easily fix
    # ok so what do i want to do? remove duplicates from dict
    # so store list of visited stuff, need cilia & cell so can store as tuple (cilia, cell)
    # cell : celi

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


# TODO SEGREGATE BY IMAGES 
# TODO REMOVE DUPS FROM CSV
 
# TODO CLEAN UP CODE -- stop hard coding things!
def main(): 
    cell_list, cilia_list = make_lists(['Location_Center_X', 'Location_Center_Y']) # TODO SEGREGATE IM HERE
    cilia_to_cell = which_cilia_closest(cell_list, cilia_list) 
    cilia_to_cell_no_dups = remove_dups_dict(cilia_to_cell)
    print(cilia_to_cell_no_dups)
    convert_dict_to_csv(cilia_to_cell_no_dups)
    # convert_to_csv(cilia_to_cell) #TODO ADD THIRD CILIA COL TO CSV
    # add_cilia_col()
    # remove_dups() #TODO REMOVE DUPS FROM CSV

if __name__ == "__main__":
    main()