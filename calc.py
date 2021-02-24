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
    output = dict()
    for num in range(1, num_im+1):
        calc_dict_cell, calc_dict_cilia = make_and_calc(num, grouped_cell, grouped_cilia)
        output.update(calc_dict_cell)
        output.update(calc_dict_cilia)

    print(output)

# makes list and calls calc mean func
def make_and_calc(im_num, grouped_cell, grouped_cilia): 
    im_df_cell = grouped_cell.get_group(im_num) 
    im_df_cell.drop('ImageNumber', axis=1, inplace=True)
    im_df_cilia = grouped_cilia.get_group(im_num) 
    im_df_cilia.drop('ImageNumber', axis=1, inplace=True)

    output_dict_cell = calc_mean(im_df_cell, im_num, '_cell')
    output_dict_cilia = calc_mean(im_df_cell, im_num, '_cilia')
    return output_dict_cell, output_dict_cilia

# calculates mean and puts it into dict 
def calc_mean(df, im_num, thing):
    output_dict =dict()
    for field in fields:
        if not field == 'ImageNumber':
            mean = df[field].mean()
            stdev =df[field].std()
            mean_name='im_' + str(im_num) + thing + '_mean_' + field
            stdev_name='im_' + str(im_num) + thing + '_stdev_' + field
            add_to_dict = {mean_name: mean, stdev_name: stdev}
            output_dict.update(add_to_dict)
    return output_dict


if __name__ == "__main__":
    main()

