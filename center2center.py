import csv
import pandas as pd
from math import sqrt

def make_lists(fields):
    cell_df = pd.read_csv('/Users/sneha/Desktop/mni/cilia-output/MyExpt_Cell.csv', skipinitialspace=True, usecols=fields)
    cell_list = cell_df.values.tolist()

    cilia_df = pd.read_csv('/Users/sneha/Desktop/mni/cilia-output/MyExpt_Cilia.csv', skipinitialspace=True, usecols=fields)
    cilia_list = cilia_df.values.tolist()

    return cell_list, cilia_list
    
# TODO FIX CILIA REPEATS
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

# todo: make it into csv (todo make cilia num col?)
def convert_to_csv(cilia_to_cell):
    csv_columns = ['cell','path_length']
    with open("Neighbors.csv", 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in cilia_to_cell:
            writer.writerow(data)

def add_cilia_col():
    with open('Neighbors.csv', 'w') as csvoutput:
        writer = csv.writer(csvoutput)
        index = 0
        for row in writer:
            index += 1
            if row[0] == "cell":
                writer.writerow(row+["cilia"])
            else:
                writer.writerow(row+[index])

def remove_dups():
    modified_csv_dict = {}
    with open('Neighbors.csv', newline='') as input:
        reader = csv.DictReader(input)
        for line_dict in reader:
            if line_dict["cell"] in modified_csv_dict:
                if line_dict["path_length"] < modified_csv_dict[line_dict["cell"]]["path_length"]:
                    modified_csv_dict[line_dict["cell"]] = line_dict
            else:
                modified_csv_dict[line_dict["cell"]] = line_dict

    for row in modified_csv_dict.values():
        print(row)

# TODO SEGREGATE BY IMAGES -- ROW 0
# TODO REMOVE DUPS FROM CSV
# TODO ADD CILIA COL TO CSV 
# TODO CLEAN UP CODE -- stop hard coding things!
def main(): 
    cell_list, cilia_list = make_lists(['Location_Center_X', 'Location_Center_Y']) # TODO SEGREGATE IM HERE
    cilia_to_cell = which_cilia_closest(cell_list, cilia_list) 
    convert_to_csv(cilia_to_cell) #TODO ADD THIRD CILIA COL TO CSV
    add_cilia_col()
    remove_dups() #TODO REMOVE DUPS FROM CSV

if __name__ == "__main__":
    main()