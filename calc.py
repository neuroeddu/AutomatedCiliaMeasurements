import pandas as pd

################################# TO CHANGE #################################
cell_csv_path='/Users/sneha/Desktop/mni/cilia-output/MyExpt_Cell.csv'
cilia_csv_path='/Users/sneha/Desktop/mni/cilia-output/MyExpt_Cilia.csv'
fields = ['ImageNumber', 'AreaShape_Area', 'AreaShape_MajorAxisLength', 'AreaShape_MinorAxisLength']
csv_path ='/Users/sneha/Desktop/mni/tester.csv'
################################# TO CHANGE #################################

def main(): 
    cell_df = pd.read_csv(cell_csv_path, skipinitialspace=True, usecols=fields)
    num_im = cell_df.ImageNumber.iat[-1]
    grouped_cell = cell_df.groupby(['ImageNumber'])
    cilia_df = pd.read_csv(cilia_csv_path, skipinitialspace=True, usecols=fields)
    grouped_cilia = cilia_df.groupby(['ImageNumber'])
    output = []
    for num in range(1, num_im+1):
        make_and_calc(num, grouped_cell, grouped_cilia, output)

    print(output)
    convert_dict_to_csv(output)

# makes list and calls calc mean func
def make_and_calc(im_num, grouped_cell, grouped_cilia, output): 
    im_df_cell = grouped_cell.get_group(im_num) 
    im_df_cell.drop('ImageNumber', axis=1, inplace=True)
    im_df_cilia = grouped_cilia.get_group(im_num) 
    im_df_cilia.drop('ImageNumber', axis=1, inplace=True)

    output.append(calc_mean(im_df_cell, im_num, 'Cell'))
    output.append(calc_mean(im_df_cell, im_num, 'Cilia'))

# calculates mean and puts it into dict 
def calc_mean(df, im_num, thing):
    # so first i need to set the im and the thing
    #                     0           1         2         3           4          5         6         7
    #  # what do i want ['Im', 'cellorcilia', 'MeanF1', 'Stdev F1', 'MeanF2', 'stdevf2', 'meanf3', 'stdevf3']
    cur_row = []
    cur_row.append(str(im_num))
    cur_row.append(thing)
    for num, field in enumerate(fields): 
        if field != 'ImageNumber':
            mean = df[field].mean() # mean of our current field
            stdev =df[field].std() # stdev of our current field
            cur_row.append(mean)
            cur_row.append(stdev)
    
    return cur_row


def convert_dict_to_csv(output):
    df = pd.DataFrame.from_dict(output)
    df.index = df.index + 1
    result = df.to_csv(path_or_buf=csv_path, header=["Image Num", "Type of image", "Mean AreaShape_Area", "Stdev AreaShape_Area", "Mean AreaShape_MajorAxisLength", "Stdev AreaShape_MajorAxisLength","Mean AreaShape_MinorAxisLength", "Stdev AreaShape_MinorAxisLength",], index=False)

if __name__ == "__main__":
    main()

