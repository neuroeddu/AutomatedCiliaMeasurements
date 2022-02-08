import pandas as pd
################################# TO CHANGE #################################
cell_csv_path='/Users/sneha/Desktop/ciliaNov22/spreadsheets_im_output/MyExpt_Nucleus.csv'
cilia_csv_path='/Users/sneha/Desktop/ciliaNov22/spreadsheets_im_output/MyExpt_Cilia.csv'
centriole_csv_path='/Users/sneha/Desktop/ciliaNov22/spreadsheets_im_output/MyExpt_Centriole.csv'
image_csv_path='/Users/sneha/Desktop/ciliaNov22/spreadsheets_im_output/MyExpt_Image.csv'
im_csv_dir_path='/Users/sneha/Desktop/ciliaNov22/im_output/'
c2c_output_path='/Users/sneha/Desktop/ciliaNov22/c2coutput.csv'
valid_cilia='/Users/sneha/Desktop/ciliaNov22/new_cilia.csv'
################################# TO CHANGE #################################


cell_df = pd.read_csv(cell_csv_path, skipinitialspace=True)
num_im = cell_df.ImageNumber.iat[-1]
num_cells=cell_df.shape[0]
grouped_cell = cell_df.groupby(['ImageNumber'])
centriole_df = pd.read_csv(centriole_csv_path, skipinitialspace=True)
grouped_centriole = centriole_df.groupby(['ImageNumber'])
cilia_df = pd.read_csv(cilia_csv_path, skipinitialspace=True)
grouped_cilia = cilia_df.groupby(['ImageNumber'])
cols_cilia = list(cilia_df.columns)[2:]
associate_df = pd.read_csv(c2c_output_path, skipinitialspace=True)
grouped_associates = associate_df.groupby(['ImageNumber'])

valid_cilia_df = pd.read_csv(valid_cilia, skipinitialspace=True)
grouped_valid_cilia = valid_cilia_df.groupby(['0'])
image_df = pd.read_csv(image_csv_path, skipinitialspace=True)

result_dict={'cilia num': -1, 'nuclei num': -1, 'present cilia/nuclei': -1, 'avg nuclei area': -1, 'avg cilia length': -1, 'avg cilia area': -1}

result_dict['avg nuclei area']=cell_df['AreaShape_Area'].mean()

cell_counts = grouped_cell.size()
cilia_counts = grouped_valid_cilia.size()

result_dict['nuclei num']=cell_counts.mean()
result_dict['cilia num']=cilia_counts.mean()

cilia_df=cilia_df[['ObjectNumber', 'ImageNumber', 'AreaShape_Area', 'AreaShape_MajorAxisLength']]
valid_cilia_df = valid_cilia_df.rename(columns={'0': 'ImageNumber', '1': 'ObjectNumber'})

df_merged = valid_cilia_df.merge(cilia_df, on=['ImageNumber', 'ObjectNumber'])

result_dict['avg cilia area']=df_merged['AreaShape_Area'].mean()
result_dict['avg cilia length']=df_merged['AreaShape_MajorAxisLength'].mean()

nuc_without_cilia=associate_df[associate_df['Cilia'].astype(int)>=0]
result_dict['present cilia/nuclei']=len(nuc_without_cilia)/len(associate_df)


print(result_dict)

print(grouped_cell.mean())