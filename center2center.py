import csv
import pandas as pd
from math import sqrt, isnan
import operator
import numpy as np
from pandas.core.frame import DataFrame
from shapely.geometry import Polygon, LineString, MultiPoint
from shapely.ops import cascaded_union, unary_union

################################# TO CHANGE #################################
cell_csv_path='/Users/sneha/Desktop/ciliaNov22/spreadsheets_im_output/MyExpt_Nucleus.csv'
cilia_csv_path='/Users/sneha/Desktop/ciliaNov22/spreadsheets_im_output/MyExpt_Cilia.csv'
centriole_csv_path='/Users/sneha/Desktop/ciliaNov22/spreadsheets_im_output/MyExpt_Centriole.csv'
output_csv_dir_path='/Users/sneha/Desktop/ciliaNov22/c2c_output'
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

# finds out which x is closest to which y assuming cutoff passed in (or none if not) 
# returns list of x to y wherein each x is matched with its closest y (ignoring if y is already matched)
def which_x_closest(y_list, x_list, cutoff = float('inf')):
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
                updated_x = True

    return x_to_y

# associate a y with an x 
# NOTE this is 1-indexed
def add_x(y_to_x, x_to_y, result, y, x):

    old_y = x_to_y[x]["y"]
    x_to_y[x]["y"] = y+1
    x_to_y[x]["path_length"] = result
    y_to_x[y]["x"] = x

    if old_y is None:
        return
    
    y_to_x[old_y]["x"] = None
    y_to_x[old_y]["x_tried"].add(x)

# remove duplicates from the dictionary to ensure 1:1 relationship
# xs will be merged if they ae 

# TODO EFFICIENCY CHANGES FOR THIS 
def merge_two_paintings(old_x_index, x_index, x_list, merged_list, im_num):
    # getting all coordinates for old x and new x 
    old_maxx, old_maxy, old_minx, old_miny=x_list[old_x_index]
    maxx, maxy, minx, miny=x_list[x_index]
    # combining them into a polygon
    cur_polygon=MultiPoint([(minx, miny), (maxx, maxy), (minx, maxy), (maxx, miny), (old_minx, old_miny), (old_maxx, old_maxy), (old_minx, old_maxy), (old_maxx, old_miny)]).convex_hull

    # this means that we have already combined this, we need to get the coordinates from 
    merged_list_old_x=[row[1] for row in merged_list]
    if old_x_index in merged_list_old_x:
        index=merged_list_old_x.index(old_x_index)
        full_list=[(minx, miny), (maxx, maxy), (minx, maxy), (maxx, miny)]+merged_list[index][8]
        cur_polygon=MultiPoint(full_list).convex_hull
        
    #cur_polygon=cascaded_union([cur_polygon, Polygon([(old_minx, old_miny), (old_maxx, old_maxy), (old_minx, old_maxy), (old_maxx, old_miny)])])
    #cur_polygon=unary_union([cur_polygon, Polygon([(old_minx, old_miny), (old_maxx, old_maxy), (old_minx, old_maxy), (old_maxx, old_miny)])
    
    # getting polygon length and width 
    perimeter=cur_polygon.length
    area=cur_polygon.area

    # get the minimum bounding rectangle and zip coordinates into a list of point-tuples
    mbr_points = list(zip(*cur_polygon.minimum_rotated_rectangle.exterior.coords.xy))

    # calculate the length of each side of the minimum bounding rectangle
    mbr_lengths = [LineString((mbr_points[i], mbr_points[i+1])).length for i in range(len(mbr_points) - 1)]

    # get major/minor axis measurements
    minor_axis = min(mbr_lengths)
    major_axis = max(mbr_lengths)

    center = list(cur_polygon.centroid.coords)
    #vertices = cur_polygon.exterior.coords
    vertices = list(cur_polygon.exterior.coords)

    merged_list.append([im_num, old_x_index, x_index, area, major_axis, minor_axis, perimeter, center[0], vertices])
    return center
    # what do i need
    # all coordinates of old x and new x, so i should pass those in 
    # what do i want to do
    # merge the two polygons, then return (a) list of merged polygons DONE and (b) rewrite path length to new path length of merged polygon


def remove_dups_dict(x_to_y, x_list, y_list, x_spatial_coordinates, im_num, merge_threshold=float('inf')):
    merged_list=[]
    y_to_x_visitation_dict = {}
    cent_to_remove=set()
    for x_index, x in enumerate(x_to_y):

        if x["y"] in y_to_x_visitation_dict: # if cell alr in visited list of cells
            old_x_index = y_to_x_visitation_dict[x["y"]]
            old_x = x_to_y[old_x_index]
            old_x_x, old_x_y=x_list[old_x_index]
            try:
                new_x_x, new_x_y=x_list[x_index]
            except:
                raise
            result = False
            old_x_coords=x_spatial_coordinates[old_x_index]
            new_x_coords=x_spatial_coordinates[x_index]

            # for coord1 in old_x_coords:
            #     for coord2 in new_x_coords:
            #         if abs(coord1-coord2)<0:
            #             result = True
            #             break
            #         else:
            #             continue
            #     break

            # if result:
            #     new_center = merge_two_paintings(old_x_index, x_index, x_spatial_coordinates, merged_list, im_num)
            #     y_to_update=x["y"]
            #     y_x, y_y = y_list[y_to_update-1]
            #     path = sqrt(pow((old_x_x - new_x_x), 2) + pow((old_x_y - new_x_y), 2)) 
            #     old_x["path_length"] = path
            #     x["path_length"] = -1.00
            #     x["y"] = -2
            #    continue
            if x["path_length"] < old_x["path_length"]: # if cur path length < path length of prev 
                old_x["path_length"] = 0.00 # set the prev one's path length to 0 and cell to none
                old_x["y"] = -1
                cent_to_remove.add(old_x_index)
                y_to_x_visitation_dict[x["y"]] = x_index

            else: # if path length of prev is better / same, keep prev 
                x["path_length"] = 0.00
                x["y"] = -1
                cent_to_remove.add(x_index)
                
        else:
            y_to_x_visitation_dict[x["y"]] = x_index
    
    return x_to_y, merged_list, cent_to_remove

# remove some duplicates to ensure 1:1 or 2 relation between y : x 
def remove_some_dups_dict(x_to_y, y_list):
    y_to_x = [
        {
            "x1": None, # The index of the current closest cilia
            "x2": None # The index of the second closest cilia
        }
        for y in range(len(y_list))
    ]

    cent_to_remove=set()

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

        # case 3: cilia1 and cilia2 full, check whether cilia2 path length> new cilia
        else:  
            old_x_index = y_to_x[cur_y]["x2"]
            old_x = x_to_y[old_x_index]
            # case 3a: cilia2 path length< new cilia, new cilia's path length n cell are 0 
            if old_x["path_length"] < x["path_length"]:
                x["path_length"] = 0.00
                x["y"] = -1
                cent_to_remove.add(x_index)
            
            # case 3b: cilia2 path length>new cilia, cilia2's path length/cell r 0, check whether new cilia pl>cilia1
            else:
                old_x["path_length"] = 0.00 # set the prev one's path length to 0 and cell to none
                old_x["y"] = -1
                cent_to_remove.add(old_x_index)
                old_x_index = y_to_x[cur_y]["x1"]
                old_x = x_to_y[old_x_index]

                # case 3b1: new cilia pl>cilia1, new cilia in cilia2
                if old_x["path_length"] < x["path_length"]:
                    y_to_x[cur_y]["x1"] = old_x_index
                    y_to_x[cur_y]["x2"] = x_index

                # case 3b2: new cilia pl<cilia1, new cilia in cilia1 and old in cilia2
                else:
                    y_to_x[cur_y]["x1"] = x_index
                    y_to_x[cur_y]["x2"] = old_x_index
        
    return x_to_y, cent_to_remove

def remove_centrioles(centriole_list, cent_to_remove, num):
    new_cent_list=[]
    idx_li=[]
    for idx, cent in enumerate(centriole_list):
        if idx not in cent_to_remove:
            new_cent_list.append(cent)
            idx_li.append([num, idx])

    return new_cent_list, idx_li


def combine_dicts(centriole_to_cell_no_dups, cilia_to_centriole_no_dups, num_im, real_idx_cent, cent_max):

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
    
    already_visited_nuc=set()
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
    c2c_output_formatted = [
        {
            'num': num_im,
            'cell': None,
            'centrioles':[],
            'path_length_centrioles':[],
            'path_length_cilia': float('inf'),
            'cilia': None,
        }
        for num in range(len(nuc_list)+1)
    ]

    df = DataFrame(c2c_output)
    print(df.columns)
    grouped=df.groupby("cell").agg(
        cell=pd.NamedAgg(column="cell", aggfunc=list),
        centriole_li=pd.NamedAgg(column="centriole", aggfunc=list),
        path_length_centriole=pd.NamedAgg(column="path_length_cell", aggfunc=list), 
        cilia_li=pd.NamedAgg(column="cilia", aggfunc=list), 
        path_length_cilia=pd.NamedAgg(column="path_length_cilia", aggfunc=list)
    )

    #grouped=df.groupby('cell').agg(centriole_list=pd.NamedAgg(column="centriole", aggfunc="list"{'centriole':list, 'path_length_cell':list, 'cilia':list, 'path_length_cilia':list })
    for x, item in enumerate(grouped.values):
        cilia=item[3]
        print(item[0])
        cell=int(item[0][0])
        print(cell)
        if cell==11:
            print()
        print(len(nuc_list))
        print(num_im)
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
    df = pd.DataFrame.from_dict(c2c_output)
    df=df.dropna()
    cols = df.columns.tolist()
    df = df[['num', 'cell', 'path_length_centrioles', 'centrioles', 'path_length_cilia', 'cilia']]
    df.to_csv(path_or_buf=output_path, header=["ImageNumber", "Nucleus", "PathLengthCentriole", "Centriole", "PathLengthCilia", "Cilia"], index=False)

def make_new_cilia_csv(cilia_output, output_path):
    df=pd.DataFrame(cilia_output)
    df = df.set_axis(['Location_Center_X', 'Location_Center_Y', 'Cilia', 'ImageNumber'], axis=1)
    df = df[['ImageNumber', 'Cilia', 'Location_Center_X', 'Location_Center_Y']]
    df.to_csv(path_or_buf=output_path)

# TODO Add this func that will merge the merged cilia list and the normal cilia list
def merge_dfs(cilia_only_x_y, merged_cilia_list, num):
    set_of_cilias={cilia[1] for cilia in merged_cilia_list}
    set_of_cilias2={cilia[2] for cilia in merged_cilia_list}
    set_of_cilias_full=set_of_cilias.union(set_of_cilias2)
    
    cilia_only_x_y['index_cilia']=range(1, len(cilia_only_x_y)+1)
    df_no_merged=cilia_only_x_y[~cilia_only_x_y['index_cilia'].isin(set_of_cilias_full)]
    
    already_visited=set()
    to_add=[]
    for thing in reversed(merged_cilia_list):
        if thing[1] not in already_visited:
            #df_no_merged.loc[og_length+len(already_visited)]=[thing[7][0], thing[7][1], thing[1]]
            to_add.append([thing[7][0], thing[7][1], thing[1]])
            already_visited.add(thing[1])
    new_df=pd.DataFrame(to_add, columns=['Location_Center_X', 'Location_Center_Y', 'index_cilia'])
    df_no_merged=df_no_merged.append(new_df)
    df_no_merged=df_no_merged.sort_values(by=['index_cilia'])
    df_no_merged['ImageNumber']=num
    
    return df_no_merged.values.tolist()
        
def main(): 
    fields = ['ImageNumber', 'Location_Center_X', 'Location_Center_Y']
    fields_cilia = ['ImageNumber', 'Location_Center_X', 'Location_Center_Y', 'AreaShape_BoundingBoxMaximum_X', 'AreaShape_BoundingBoxMaximum_Y', 'AreaShape_BoundingBoxMinimum_X', 'AreaShape_BoundingBoxMinimum_Y']
    cell_df = pd.read_csv(cell_csv_path, skipinitialspace=True, usecols=fields)
    num_im = cell_df.ImageNumber.iat[-1]
    num_cells=cell_df.shape[0]
    grouped_cell = cell_df.groupby(['ImageNumber'])
    centriole_df = pd.read_csv(centriole_csv_path, skipinitialspace=True, usecols=fields)
    grouped_centriole = centriole_df.groupby(['ImageNumber'])
    cilia_df = pd.read_csv(cilia_csv_path, skipinitialspace=True, usecols=fields_cilia)
    cilia_only_x_y=cilia_df.drop(columns=['AreaShape_BoundingBoxMaximum_X', 'AreaShape_BoundingBoxMaximum_Y', 'AreaShape_BoundingBoxMinimum_X', 'AreaShape_BoundingBoxMinimum_Y'])
    cilia_spatial_coordinates=cilia_df.drop(columns=['Location_Center_X', 'Location_Center_Y'])
    grouped_cilia = cilia_only_x_y.groupby(['ImageNumber'])
    grouped_cilia_spatial=cilia_spatial_coordinates.groupby(['ImageNumber'])

    c2c_output = []
    merged_list_full=[]
    merged_df_full=[]
    new_cent=[]
    new_cilia=[]
    for num in range(1, num_im+1):
        cell_list, centriole_list = make_lists(num, grouped_cell, grouped_centriole)
        centriole_to_cell = which_x_closest(cell_list, centriole_list) 
        centriole_to_cell_no_dups, cent_to_remove = remove_some_dups_dict(centriole_to_cell, cell_list)
        # here : pass in cent to remove, list of 
        new_cent_list, idx_li1=remove_centrioles(centriole_list, cent_to_remove, num)
        idx_li_1_indexed=[[idx[0],(idx[1]+1)] for idx in idx_li1]
        new_cent+=idx_li_1_indexed
        # TODO this is redundant make new make_lists func
        cilia_list, centriole_list = make_lists(num, grouped_cilia, grouped_centriole)
        cilia_to_centriole = which_x_closest(new_cent_list, cilia_list) 
        im_df = grouped_cilia_spatial.get_group(num) 
        im_df.drop('ImageNumber', axis=1, inplace=True)
        cilia_spatial = im_df.values.tolist()
        cilia_to_centriole_no_dups, merged_list, cilia_to_remove = remove_dups_dict(cilia_to_centriole, cilia_list, new_cent_list, cilia_spatial, num)
        new_cilia_list, idx_li2=remove_centrioles(cilia_list, cilia_to_remove, num)
        idx_li_2_1_indexed=[[idx[0],(idx[1]+1)] for idx in idx_li2]
        new_cilia+=idx_li_2_1_indexed


        c2c_output_part=combine_dicts(centriole_to_cell_no_dups, cilia_to_centriole_no_dups, num, idx_li1, len(centriole_list))
        c2c_output_format = convert_format_output(c2c_output_part, num, cell_list, cilia_list)
        im_df = grouped_cilia.get_group(num) 
        im_df.drop('ImageNumber', axis=1, inplace=True)
        df_merged=merge_dfs(im_df, merged_list, num) 
        merged_df_full+=df_merged
        c2c_output+=c2c_output_format
        merged_list_full+=merged_list
     
    output_path=output_csv_dir_path + '/c2coutput.csv' 
    df = pd.DataFrame(merged_list_full)
    just_merged_path=output_csv_dir_path+'/merged_cilia_info.csv'
    #output_path_merged=
    df2 = pd.DataFrame(new_cent)
    df3 = pd.DataFrame(new_cilia)
    output_cent=output_csv_dir_path+'/new_cent.csv'
    output_cilia=output_csv_dir_path+'/new_cilia.csv'
    # make_new_cilia_csv(merged_df_full, output_cilia)
    df2.to_csv(output_cent)
    df3.to_csv(output_cilia)

    convert_dict_to_csv(c2c_output, output_path)


if __name__ == "__main__":
    main()