import csv
import pandas as pd
from math import sqrt, isnan
import numpy as np
from pandas.core.frame import DataFrame
from shapely.geometry import Polygon, LineString, MultiPoint
from shapely.ops import cascaded_union, unary_union

################################# TO CHANGE #################################
csv_folder='/Users/sneha/Desktop/ciliaJan22/spreadsheets_im_output'
output_csv_dir_path='/Users/sneha/Desktop/ciliaJan22/c2coutputnone'
################################# TO CHANGE #################################

def make_lists(im_num, grouped):
    """
    Group dataframe into only rows where image is im_num and return the values in a list

    :param im_num: The image number
    :param grouped: The dataframe we want to get relevant rows of 
    :returns: list of (x,y) coordinates for all relevant rows of dataframe
    """

    im_df = grouped.get_group(im_num) 
    im_df.drop('ImageNumber', axis=1, inplace=True)
    return im_df.values.tolist()

def which_x_closest(y_list, x_list, cutoff = float('inf')):
    """
    Finds out which x is closest to which y assuming the cutoff distance 

    :param y_list: List of coordinates for y
    :param x_list: List of coordinates for x
    :param cutoff: Cutoff for how distant an x and y can be if it's the closest y to that x 
    :returns: list that maps each x to its closest y, with x being 0 indexed and y being 1 indexed
    """

    y_to_x = [
        {
            "x": None, # The index of the current closest x
            "x_tried": set() # The indicies of previously tried x that shouldn't be tried again
        }
        for _ in y_list
    ]
    x_to_y = [
        {
            "path_length": float('inf'), # The length of the shortest path
            "y": None # The index of the cell to which the shortest path corresponds
        }
        for _ in x_list
    ]

    updated_y = True
    
    while updated_y: # while y are being updated, calculate lengths and see if the y should be added
        updated_y = False
        for i, y in enumerate(y_list):
            x_y, y_y = y
            for j, x in enumerate(x_list):
                x_x, y_x = x
                result = sqrt(pow((x_x - x_y), 2) + pow((y_x - y_y), 2))
                
                if result > cutoff or j in y_to_x[i]["x_tried"] or result >= x_to_y[j]["path_length"]:
                    continue

                add_x(y_to_x, x_to_y, result, i, j)
                updated_y = True

    return x_to_y

def add_x(y_to_x, x_to_y, result, y, x):
    """
    Helper function that associates y with an x

    :param y_to_x: Shows what x we're currently at and which ones we've tried already for each y
    :param x_to_y: Maps x to its closest y 
    :param result: Distance between x and y
    :param y: Index of current y
    :param x: Index of current x 
    :returns: N/A, edits y_to_x and x_to_y defined in main func (which_x_closest)
    """

    old_y = x_to_y[x]["y"]
    x_to_y[x]["y"] = y+1
    x_to_y[x]["path_length"] = result
    y_to_x[y]["x"] = x

    if old_y is None:
        return
    
    y_to_x[old_y]["x"] = None
    y_to_x[old_y]["x_tried"].add(x)

def remove_dups_dict(x_to_y):
    """
    Remove all duplicate ys in the x_to_y list to maintain a 1:1 x:y ratio

    :param x_to_y: List that we want to remove duplicates from
    :param x_list: List of coordinates for each x, used to calculate noise list 
    :returns: x_to_y without duplicates, list of x that don't have a y (ie noise)
    """
    y_to_x_visitation_dict = {}
    x_to_remove=set()
    for x_index, x in enumerate(x_to_y):
        

        if x["y"] in y_to_x_visitation_dict: # if cell alr in visited list of cells
            old_x_index = y_to_x_visitation_dict[x["y"]]
            old_x = x_to_y[old_x_index]

            if x["path_length"] < old_x["path_length"]: # if cur path length < path length of prev 
                old_x["path_length"] = 0.00 # set the prev one's path length to 0 and cell to none
                old_x["y"] = -1
                x_to_remove.add(old_x_index)
                y_to_x_visitation_dict[x["y"]] = x_index

            else: # if path length of prev is better / same, keep prev 
                x["path_length"] = 0.00
                x["y"] = -1
                x_to_remove.add(x_index)
                
        else:
            y_to_x_visitation_dict[x["y"]] = x_index
    


    return x_to_y, x_to_remove

def remove_some_dups_dict(x_to_y, len_y_list, x_list):
    """
    Remove some duplicate ys in the x_to_y list to maintain a 1 or 2:1 x:y ratio

    :param x_to_y: List that we want to remove duplicates from
    :param len_y_list: Number of y there are 
    :param x_list: List of coordinates for each x, used to calculate noise list
    :returns: x_to_y without duplicates, list of x that don't have a y (ie noise)
    """
    y_to_x = [
        {
            "x1": None, # The index of the current closest cilia
            "x2": None # The index of the second closest cilia
        }
        for y in range(len_y_list)
    ]

    x_to_remove=set()

    for x_index, x in enumerate(x_to_y):
        if x["y"] is None:
            continue
        cur_y=x["y"]-1
        # case 1: x1==none, put it in x1
        if not y_to_x[cur_y]["x1"]:
            y_to_x[cur_y]["x1"] = x_index
        # case 2: x2==none, put closest one in x1 and put second in x2
        elif not y_to_x[cur_y]["x2"]:
            old_x_index = y_to_x[cur_y]["x1"]
            old_x = x_to_y[old_x_index]
            if x["path_length"] < old_x["path_length"]:
                y_to_x[cur_y]["x1"] = x_index
                y_to_x[cur_y]["x2"] = old_x_index

            else:
                y_to_x[cur_y]["x2"] = x_index

        # case 3: x1 and x2 full, check whether x2 path length > new x
        else:  
            old_x_index = y_to_x[cur_y]["x2"]
            old_x = x_to_y[old_x_index]
            # case 3a: x2 path length < new x, new x's path length and cell are 0 
            if old_x["path_length"] < x["path_length"]:
                x_to_remove.add(x_index)
                x["path_length"] = 0.00
                x["y"] = -1
            
            # case 3b: x2 path length > new x, x2's path length/cell are 0, check whether new x path > x1
            else:
                x_to_remove.add(old_x_index)
                old_x["path_length"] = 0.00 # set the prev one's path length to 0 and cell to none
                old_x["y"] = -1
                old_x_index = y_to_x[cur_y]["x1"]
                old_x = x_to_y[old_x_index]

                # case 3b1: new x path > x1, new x in x2
                if old_x["path_length"] < x["path_length"]:
                    y_to_x[cur_y]["x1"] = old_x_index
                    y_to_x[cur_y]["x2"] = x_index

                # case 3b2: new x path < x1, new x in x1 and old in x2
                else:
                    y_to_x[cur_y]["x1"] = x_index
                    y_to_x[cur_y]["x2"] = old_x_index

        
    return x_to_y, x_to_remove

def remove_noise(x_list, noise_list, num):
    """
    Make a list of indices of x list that are attached to some y

    :param x_list: List of coordinates for each x
    :param noise_list: Noise indices to get rid of
    :param num: Image number 
    :returns: List of x that only has the x that have been paired, List of indices with invalid centrioles skipped
    """

    valid_list=[]
    true_idx_mapping=[]
    for idx, cur_x in enumerate(x_list):
        if idx not in noise_list:
            valid_list.append(cur_x)
            true_idx_mapping.append([num, idx])

    return valid_list, true_idx_mapping


def combine_dicts(centriole_to_cell_no_dups, cilia_to_centriole_no_dups, num_im, real_idx_cent, cent_max):
    """
    Combine information for the nuclei/centriole pairings and the centriole/cilia pairings

    :param centriole_to_cell_no_dups: List of pairings for nuclei/centriole
    :param cilia_to_centriole_no_dups: List of pairings for centriole/cilia
    :param num_im: Image number 
    :param real_idx_cent: The correct indices of each of the centrioles in the new centriole list 
    :param cent_max: The number that the centrioles go up to 
    :returns: List of dictionaries that contain a pairing from nuclei to centriole to cilia
    """

    c2c_output = [
        {
            'num': num_im,
            'path_length_cell': float('inf'),
            'cell': None,
            'path_length_cilia': float('inf'),
            'cilia': None,
            'centriole': num+1
        }
        for num in range(len(centriole_to_cell_no_dups))
    ]

    cent_to_cilia = [
        {
            "path_length": float('inf'), # The length of the shortest path
            "y": None # The index of the cell to which the shortest path corresponds
        }
        for _ in range(cent_max+1)
        #for _ in range(max(cent_info['y'] for cent_info in cilia_to_centriole_no_dups)+1)
    ]
    
    
    # Converting cilia to centriole into centriole to cilia for easier combination with the other dictionary
    for idx, cilia_info in enumerate(cilia_to_centriole_no_dups):
        cent_to_cilia[cilia_info['y']-1]['path_length']=cilia_info['path_length']
        cent_to_cilia[cilia_info['y']-1]['y']=idx

    # Get centriole indices and combine dicts relative to centriole information
    for index, num in enumerate(real_idx_cent):

        # Make sure the only centriole numbers that are represented are the ones that are valid 
        c2c_output[index]['centriole']=num[1]+1

        c2c_output[index]['path_length_cell'] = centriole_to_cell_no_dups[num[1]]['path_length']
        c2c_output[index]['cell'] = centriole_to_cell_no_dups[num[1]]['y']

        if cent_to_cilia[index]['y'] is None:
            continue

        c2c_output[index]['path_length_cilia'] = cent_to_cilia[index]['path_length']
        c2c_output[index]['cilia'] = cent_to_cilia[index]['y']+1

    return c2c_output

def convert_format_output(c2c_output, num_im, cell_list, cilia_list):
    """
    Convert output from in terms of centrioles to in terms of cells & maintain 1:1 cell:cilia ratio

    :param c2c_output: Combined dictionaries with all the information we want to store 
    :param num_im: Image number 
    :param cell_list: List of nuclei coordinates that we will use to calculate how far cilia are from nuclei
    :param cilia_list: List of cilia coordinates that we will use to calculate how far cilia are from nuclei
    :returns: Formatted list of dictionaries 
    """

    # Set up dictionary
    c2c_output_formatted = [
        {
            'num': num_im,
            'cell': None,
            'centrioles':[],
            'path_length_centrioles':[],
            'path_length_cilia': float('inf'),
            'cilia': None,
        }
        for _ in range(len(cell_list)+1)
    ]

    # Set of cilia that get removed
    cilia_to_remove=set()

    # Convert previous output to dataframe and combine all rows with same nucleus into one row
    df = DataFrame(c2c_output)
    grouped=df.groupby("cell").agg(
        cell=pd.NamedAgg(column="cell", aggfunc=list),
        centriole_li=pd.NamedAgg(column="centriole", aggfunc=list),
        path_length_centriole=pd.NamedAgg(column="path_length_cell", aggfunc=list), 
        cilia_li=pd.NamedAgg(column="cilia", aggfunc=list), 
        path_length_cilia=pd.NamedAgg(column="path_length_cilia", aggfunc=list)
    )


    for item in grouped.values:

        # Add centrioles and cell info into output 
        cilia=item[3]
        cell=int(item[0][0])
        c2c_output_formatted[cell]['cell']=cell
        c2c_output_formatted[cell]['centrioles']=item[1]
        if item[1] is None:
            c2c_output_formatted[cell]['centrioles']=-1
        c2c_output_formatted[cell]['path_length_centrioles']=item[2]
        cell_x, cell_y=cell_list[int(cell-1)]

        # If cilia in a list, check how many and add
        if len(cilia)>1:
            # If both nan, don't add anything
            if isnan(cilia[1]) and isnan(cilia[0]):
                c2c_output_formatted[cell]['cilia']=-1
                c2c_output_formatted[cell]['path_length_cilia']=-1
                continue

            # If only one is nan, add the other and get dist. to cell
            if isnan(cilia[1]) and not isnan(cilia[0]):
                c2c_output_formatted[cell]['cilia']=item[3][0]
                cilia1_x, cilia1_y=cilia_list[int(cilia[0])-1]
                dist = sqrt(pow((cilia1_x - cell_x), 2) + pow((cilia1_y - cell_y), 2))
                c2c_output_formatted[cell]['path_length_cilia']=dist

            elif isnan(cilia[0]) and not isnan(cilia[1]):
                c2c_output_formatted[cell]['cilia']=item[3][1]
                cilia2_x, cilia2_y=cilia_list[int(cilia[1])-1]
                dist= sqrt(pow((cilia2_x - cell_x), 2) + pow((cilia2_y - cell_y), 2))
                c2c_output_formatted[cell]['path_length_cilia']=dist
                
            # Otherwise, calculate both distances to cell and only put in the one 
            else:
                cilia1_x, cilia1_y=cilia_list[int(cilia[0])-1]
                cilia2_x, cilia2_y=cilia_list[int(cilia[1])-1]

                dist1= sqrt(pow((cilia1_x - cell_x), 2) + pow((cilia1_y - cell_y), 2))
                dist2= sqrt(pow((cilia2_x - cell_x), 2) + pow((cilia2_y - cell_y), 2))

                if dist1>=dist2:
                    cilia_to_remove.add(cilia[0]-1)
                    c2c_output_formatted[cell]['cilia']=cilia[1]
                    c2c_output_formatted[cell]['path_length_cilia']=dist2

                else:
                    # NOTE This is -1 because the valid cilia list is initially made via 0 indexing 
                    cilia_to_remove.add(cilia[1]-1)
                    c2c_output_formatted[cell]['cilia']=cilia[0]
                    c2c_output_formatted[cell]['path_length_cilia']=dist1

        
        # If singleton cilia from aggregate(i.e. cell matched with one centriole), add indiscriminately 
        else:
            if isnan(item[3][0]):
                c2c_output_formatted[cell]['cilia']=-1
                c2c_output_formatted[cell]['path_length_cilia']=-1
            else:
                c2c_output_formatted[cell]['cilia']=item[3][0]
                cilia_x, cilia_y=cilia_list[int(item[3][0])-1]
                dist= sqrt(pow((cilia_x - cell_x), 2) + pow((cilia_y - cell_y), 2))
                c2c_output_formatted[cell]['path_length_cilia']=dist

    return c2c_output_formatted, cilia_to_remove
    
def convert_dict_to_csv(c2c_output, output_path):
    """
    Convert our output into a csv

    :param c2c_output: Output to store
    :param output_path: Path to store output to
    :returns: None
    """

    df = pd.DataFrame.from_dict(c2c_output)
    df=df.dropna()
    cols = df.columns.tolist()
    df = df[['num', 'cell', 'path_length_centrioles', 'centrioles', 'path_length_cilia', 'cilia']]
    df.to_csv(path_or_buf=output_path, header=["ImageNumber", "Nucleus", "PathLengthCentriole", "Centriole", "PathLengthCilia", "Cilia"], index=False, float_format="%.10g")
        
def main(): 
    
    # Read input from input folder
    fields = ['ImageNumber', 'Location_Center_X', 'Location_Center_Y']

    cell_df = pd.read_csv(csv_folder+'/MyExpt_Nucleus.csv', skipinitialspace=True, usecols=fields)
    num_im = cell_df.ImageNumber.iat[-1]
    grouped_cell = cell_df.groupby(['ImageNumber'])
    centriole_df = pd.read_csv(csv_folder+'/MyExpt_Centriole.csv', skipinitialspace=True, usecols=fields)
    grouped_centriole = centriole_df.groupby(['ImageNumber'])
    cilia_df = pd.read_csv(csv_folder+'/MyExpt_Cilia.csv', skipinitialspace=True, usecols=fields)
    grouped_cilia = cilia_df.groupby(['ImageNumber'])

    # Make lists for the output
    c2c_output = []
    valid_cent=[]
    valid_cilia=[]
    full_cent_to_cilia=[]
    for num in range(1, num_im+1):
        # Convert groupby objects to list for easy access
        cell_list = make_lists(num, grouped_cell)
        centriole_list = make_lists(num, grouped_centriole)
        cilia_list = make_lists(num, grouped_cilia)

        # Match centrioles to cell (nuclei)
        centriole_to_cell = which_x_closest(cell_list, centriole_list) 
        centriole_to_cell_no_dups, cent_to_remove = remove_some_dups_dict(centriole_to_cell, len(cell_list), centriole_list)

        # Only use valid (paired) centrioles for the next step
        valid_cent_coords, valid_cent_indices=remove_noise(centriole_list, cent_to_remove, num)

        # 1 index the valid centrioles and store them 
        valid_cent_indices_1_idx=[[idx[0],(idx[1]+1)] for idx in valid_cent_indices]
        valid_cent+=valid_cent_indices_1_idx

        # Make dictionary from indices to true indices
        idx_dict={}
        for idx, true_idx in enumerate(valid_cent_indices_1_idx):
            idx_dict[idx+1] = true_idx[1]

        # Match cilia to centrioles 
        cilia_to_centriole = which_x_closest(valid_cent_coords, cilia_list) 
        cilia_to_centriole_no_dups, cilia_to_remove = remove_dups_dict(cilia_to_centriole)
        cilia_to_cent = []

        # Make cilia to cent csv 
        cilia_to_cent = [{'ImageNumber':num, 'Cilia':x+1, 'Centriole':idx_dict[dict_cent['y']], 'path':dict_cent['path_length']} for x, dict_cent in enumerate(cilia_to_centriole_no_dups) if dict_cent['y']!=-1]
        full_cent_to_cilia += cilia_to_cent

        # Make output and format it 
        c2c_output_part = combine_dicts(centriole_to_cell_no_dups, cilia_to_centriole_no_dups, num, valid_cent_indices, len(centriole_list))
        c2c_output_format, more_cilia_to_remove = convert_format_output(c2c_output_part, num, cell_list, cilia_list)

        # Make list of valid cilia and store 
        _, valid_cilia_indices=remove_noise(cilia_list, cilia_to_remove.union(more_cilia_to_remove), num)
        valid_cilia_indices=[[idx[0],(idx[1]+1)] for idx in valid_cilia_indices]
        valid_cilia+=valid_cilia_indices

        c2c_output += c2c_output_format

    valid_cent_df = pd.DataFrame(valid_cent)
    valid_cilia_df = pd.DataFrame(valid_cilia)
    cent_to_cilia_df = pd.DataFrame(full_cent_to_cilia)

    convert_dict_to_csv(c2c_output,output_csv_dir_path + '/c2coutput.csv')
    valid_cent_df.to_csv(output_csv_dir_path+'/new_cent.csv')
    valid_cilia_df.to_csv(output_csv_dir_path+'/cent2cilia.csv')
    cent_to_cilia_df.to_csv(output_csv_dir_path+'/new_cilia.csv')



if __name__ == "__main__":
    main()