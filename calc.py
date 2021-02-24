import pandas as pd

################################# TO CHANGE #################################
cell_csv_path='/Users/sneha/Desktop/mni/cilia-output/MyExpt_Cell.csv'
cilia_csv_path='/Users/sneha/Desktop/mni/cilia-output/MyExpt_Cilia.csv'
fields = ['ImageNumber', 'AreaShape_Area', 'AreaShape_MajorAxisLength', 'AreaShape_MinorAxisLength']
################################# TO CHANGE #################################

def main(): 
    cell_df = pd.read_csv(cell_csv_path, skipinitialspace=True, usecols=fields)
    num_im = cell_df.ImageNumber.iat[-1]
    grouped_cell = cell_df.groupby(['ImageNumber'])
    cilia_df = pd.read_csv(cilia_csv_path, skipinitialspace=True, usecols=fields)
    grouped_cilia = cilia_df.groupby(['ImageNumber'])
    for num in range(1, num_im+1):
        calc_dict_cell, calc_dict_cilia = make_lists(num, grouped_cell, grouped_cilia)
        print(calc_dict_cell)
        print(calc_dict_cilia)

# calculates mean and puts it into dict 
def calc_mean(df, im_num):
    output_dict =dict()
    for field in fields:
        if not field == 'ImageNumber':
            mean = df[field].mean()
            stdev =df[field].std()
            mean_name='im_' + str(im_num) + '_mean_' + field
            stdev_name='im_' + str(im_num) + '_stdev_' + field
            add_to_dict = {mean_name: mean, stdev_name: stdev}
            output_dict.update(add_to_dict)
    return output_dict

# makes lists
def make_lists(im_num, grouped):
    im_df = grouped.get_group(im_num) 
    im_df.drop('ImageNumber', axis=1, inplace=True)
    calc_dict = calc_mean(im_df, im_num)
    return calc_dict

def make_and_calccalc(im_num, grouped_cell, grouped_cilia): 
    calc_dict_cell = make_lists(im_num, grouped_cell)
    calc_dict_cilia = make_lists(im_num, grouped_cilia)

    return calc_dict_cell, calc_dict_cilia

if __name__ == "__main__":
    main()

