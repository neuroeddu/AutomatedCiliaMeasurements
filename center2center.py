import csv
import pandas as pd
from math import sqrt, isnan
import numpy as np
from pandas.core.frame import DataFrame
from shapely.geometry import Polygon, LineString, MultiPoint
from shapely.ops import cascaded_union, unary_union

################################# TO CHANGE #################################
cell_csv_path='/Users/sneha/Desktop/ciliaNov22/spreadsheets_im_output/MyExpt_Nucleus.csv'
cilia_csv_path='/Users/sneha/Desktop/ciliaNov22/spreadsheets_im_output/MyExpt_Cilia.csv'
centriole_csv_path='/Users/sneha/Desktop/ciliaNov22/spreadsheets_im_output/MyExpt_Centriole.csv'
output_csv_dir_path='/Users/sneha/Desktop/ciliaNov22/'
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
                if j==194 and i==88:
                    print()
                x_x, y_x = x
                result = sqrt(pow((x_x - x_y), 2) + pow((y_x - y_y), 2))
                
                if result > cutoff or j in y_to_x[i]["x_tried"] or result >= x_to_y[j]["path_length"]:
                    continue

                add_x(y_to_x, x_to_y, result, i, j)
                updated_x = True

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

def remove_dups_dict(x_to_y, x_list):
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
                y_to_x_visitation_dict[x["y"]] = x_index

            else: # if path length of prev is better / same, keep prev 
                x["path_length"] = 0.00
                x["y"] = -1
                
        else:
            y_to_x_visitation_dict[x["y"]] = x_index
    
    for x in range(len(x_list)):
        if x_to_y[x]["y"]==-1:
            x_to_remove.add(x)


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
                x["path_length"] = 0.00
                x["y"] = -1
            
            # case 3b: x2 path length > new x, x2's path length/cell are 0, check whether new x path > x1
            else:
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

    for x in range(len(x_list)):
        try:
            if x_to_y[x]["y"]==-1:
                x_to_remove.add(x)
        except:
            raise
        
    return x_to_y, x_to_remove

def remove_noise(x_list, noise_list, num):
    """
    Make a list of indices of x list that are attached to some y

    :param x_list: List of coordinates for each x
    :param noise_list: Noise indices to get rid of
    :param num: Image number 
    :returns: List of x that only has the x that have been paired, list of indices those x are at in the original csv
    """

    new_list=[]
    idx_li=[]
    for idx, cur_x in enumerate(x_list):
        if idx not in noise_list:
            new_list.append(cur_x)
            idx_li.append([num, idx])

    return new_list, idx_li


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
    
    for idx, cilia_info in enumerate(cilia_to_centriole_no_dups):
        try:
            cent_to_cilia[cilia_info['y']-1]['path_length']=cilia_info['path_length']
        except:
            raise
        cent_to_cilia[cilia_info['y']-1]['y']=idx

    for index, num in enumerate(real_idx_cent):

        c2c_output[index]['centriole']=num[1]+1
        c2c_output[index]['path_length_cell'] = centriole_to_cell_no_dups[num[1]]['path_length']
        c2c_output[index]['cell'] = centriole_to_cell_no_dups[num[1]]['y']

        if cent_to_cilia[index]['y'] is None:
            continue
        c2c_output[index]['path_length_cilia'] = cent_to_cilia[index]['path_length']
        c2c_output[index]['cilia'] = cent_to_cilia[index]['y']+1

    return c2c_output

def convert_format_output(c2c_output, num_im, nuc_list, cilia_list):
    """
    Convert output format to the format we want 

    :param c2c_output: Combined dictionaries with all the information we want to store 
    :param num_im: Image number 
    :param nuc_list: List of nuclei coordinates that we will use to calculate how far cilia are from nuclei
    :param cilia_list: List of cilia coordinates that we will use to calculate how far cilia are from nuclei
    :returns: Formatted list of dictionaries 
    """

    c2c_output_formatted = [
        {
            'num': num_im,
            'cell': None,
            'centrioles':[],
            'path_length_centrioles':[],
            'path_length_cilia': float('inf'),
            'cilia': None,
        }
        for _ in range(len(nuc_list)+1)
    ]

    df = DataFrame(c2c_output)
    grouped=df.groupby("cell").agg(
        cell=pd.NamedAgg(column="cell", aggfunc=list),
        centriole_li=pd.NamedAgg(column="centriole", aggfunc=list),
        path_length_centriole=pd.NamedAgg(column="path_length_cell", aggfunc=list), 
        cilia_li=pd.NamedAgg(column="cilia", aggfunc=list), 
        path_length_cilia=pd.NamedAgg(column="path_length_cilia", aggfunc=list)
    )
    for item in grouped.values:
        cilia=item[3]
        cell=int(item[0][0])
        c2c_output_formatted[cell]['cell']=cell
        c2c_output_formatted[cell]['centrioles']=item[1]
        c2c_output_formatted[cell]['path_length_centrioles']=item[2]
        nuc_x, nuc_y=nuc_list[int(cell-1)]
        if len(cilia)>1:
            if isnan(cilia[1]) and isnan(cilia[0]):
                continue
            if isnan(cilia[1]) and not isnan(cilia[0]):
                c2c_output_formatted[cell]['cilia']=item[3][0]
                cilia1_x, cilia1_y=cilia_list[int(cilia[0])-1]
                dist = sqrt(pow((cilia1_x - nuc_x), 2) + pow((cilia1_y - nuc_y), 2))
                c2c_output_formatted[cell]['path_length_cilia']=dist

            elif isnan(cilia[0]) and not isnan(cilia[1]):
                c2c_output_formatted[cell]['cilia']=item[3][1]
                try:
                    cilia2_x, cilia2_y=cilia_list[int(cilia[1])-1]
                except:
                    raise
                dist= sqrt(pow((cilia2_x - nuc_x), 2) + pow((cilia2_y - nuc_y), 2))
                c2c_output_formatted[cell]['path_length_cilia']=dist
                
            else:
                cilia1_x, cilia1_y=cilia_list[int(cilia[0])-1]
                cilia2_x, cilia2_y=cilia_list[int(cilia[1])-1]

                dist1= sqrt(pow((cilia1_x - nuc_x), 2) + pow((cilia1_y - nuc_y), 2))
                dist2= sqrt(pow((cilia2_x - nuc_x), 2) + pow((cilia2_y - nuc_y), 2))

                min_cil= min((dist1, cilia[0]), (dist2, cilia[1]))

                c2c_output_formatted[cell]['cilia']=min_cil[1]
                c2c_output_formatted[cell]['path_length_cilia']=min_cil[0]

        else:
            c2c_output_formatted[cell]['cilia']=item[3]
            c2c_output_formatted[cell]['path_length_cilia']=item[4]

    return c2c_output_formatted
    
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
    df.to_csv(path_or_buf=output_path, header=["ImageNumber", "Nucleus", "PathLengthCentriole", "Centriole", "PathLengthCilia", "Cilia"], index=False)
        
def main(): 
    fields = ['ImageNumber', 'Location_Center_X', 'Location_Center_Y']

    cell_df = pd.read_csv(cell_csv_path, skipinitialspace=True, usecols=fields)
    num_im = cell_df.ImageNumber.iat[-1]
    num_cells=cell_df.shape[0]
    grouped_cell = cell_df.groupby(['ImageNumber'])
    centriole_df = pd.read_csv(centriole_csv_path, skipinitialspace=True, usecols=fields)
    grouped_centriole = centriole_df.groupby(['ImageNumber'])
    cilia_df = pd.read_csv(cilia_csv_path, skipinitialspace=True, usecols=fields)
    grouped_cilia = cilia_df.groupby(['ImageNumber'])

    c2c_output = []
    new_cent=[]
    new_cilia=[]
    full_cent_to_cilia=[]
    for num in range(1, num_im+1):
        cell_list = make_lists(num, grouped_cell)
        centriole_list = make_lists(num, grouped_centriole)
        cilia_list = make_lists(num, grouped_cilia)
        centriole_to_cell = which_x_closest(cell_list, centriole_list) 
        centriole_to_cell_no_dups, cent_to_remove = remove_some_dups_dict(centriole_to_cell, len(cell_list), centriole_list)
        new_cent_list, idx_li1=remove_noise(centriole_list, cent_to_remove, num)
        idx_li_1_indexed=[[idx[0],(idx[1]+1)] for idx in idx_li1]
        new_cent+=idx_li_1_indexed
        idx_dict={}
        for idx, l2 in enumerate(idx_li_1_indexed):
            idx_dict[idx+1] = l2[1]
        cilia_to_centriole = which_x_closest(new_cent_list, cilia_list) 
        cilia_to_centriole_no_dups, cilia_to_remove = remove_dups_dict(cilia_to_centriole, cilia_list)
        cilia_to_cent=[{'ImageNumber':num, 'Cilia':x+1, 'Centriole':idx_dict[dict_cent['y']], 'path':dict_cent['path_length']} for x, dict_cent in enumerate(cilia_to_centriole_no_dups) if dict_cent['y']!=-1]
        full_cent_to_cilia+= cilia_to_cent
        #[{}, {}, {}]
        #[[1, {}], [1, {}]]
        _, idx_li2=remove_noise(cilia_list, cilia_to_remove, num)
        idx_li_2_1_indexed=[[idx[0],(idx[1]+1)] for idx in idx_li2]
        new_cilia+=idx_li_2_1_indexed


        c2c_output_part=combine_dicts(centriole_to_cell_no_dups, cilia_to_centriole_no_dups, num, idx_li1, len(centriole_list))
        c2c_output_format = convert_format_output(c2c_output_part, num, cell_list, cilia_list)
        c2c_output+=c2c_output_format
     
    output_path=output_csv_dir_path + '/c2coutput.csv' 
    df2 = pd.DataFrame(new_cent)
    df3 = pd.DataFrame(new_cilia)
    df4 = pd.DataFrame(full_cent_to_cilia)
    output_cent=output_csv_dir_path+'/new_cent.csv'
    output_new_cent=output_csv_dir_path+'/cent2cilia.csv'
    output_cilia=output_csv_dir_path+'/new_cilia.csv'
    df2.to_csv(output_cent)
    df3.to_csv(output_cilia)
    df4.to_csv(output_new_cent)


    convert_dict_to_csv(c2c_output, output_path)


if __name__ == "__main__":
    main()