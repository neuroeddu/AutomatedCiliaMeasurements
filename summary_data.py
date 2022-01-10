from bokeh.models.sources import ColumnDataSource
import pandas as pd
from bokeh.plotting import figure, curdoc
from bokeh.io import show, output_notebook
from bokeh.models import Dropdown
from bokeh.layouts import column, layout
from functools import partial

################################# TO CHANGE #################################
cell_csv_path='/Users/sneha/Desktop/ciliaNov22/spreadsheets_im_output/MyExpt_Nucleus.csv'
cilia_csv_path='/Users/sneha/Desktop/ciliaNov22/spreadsheets_im_output/MyExpt_Cilia.csv'
centriole_csv_path='/Users/sneha/Desktop/ciliaNov22/spreadsheets_im_output/MyExpt_Centriole.csv'
image_csv_path='/Users/sneha/Desktop/ciliaNov22/spreadsheets_im_output/MyExpt_Image.csv'
im_csv_dir_path='/Users/sneha/Desktop/ciliaNov22/im_output/'
c2c_output_path='/Users/sneha/Desktop/ciliaNov22/c2coutput.csv'
valid_cilia='/Users/sneha/Desktop/ciliaNov22/new_cilia.csv'
################################# TO CHANGE #################################

def make_figure():
    p = figure(plot_height = 600, plot_width = 600, 
            title = 'Histograms',
            x_axis_label = 'Image number', 
            y_axis_label = '')
    return p

def num_nuc_per_im(image_df, **kwargs):
    return image_df['Count_Nucleus'].values.tolist()

def num_cilia_per_im(grouped_valid_cilia, num_im, **kwargs):
    return [len(make_lists(num, grouped_valid_cilia, '0')) for num in range(1, num_im+1)]

def make_lists(num_im, grouped, colname='ImageNumber', **kwargs):
    """
    Group dataframe into only rows where image is im_num and return the values in a list

    :param num_im: The image number
    :param grouped: The dataframe we want to get relevant rows of 
    :returns: list of (x,y) coordinates for all relevant rows of dataframe
    """

    im_df = grouped.get_group(num_im) 
    im_df.drop(colname, axis=1, inplace=True)
    return im_df.values.tolist()

def single_cent_to_two_cent(grouped_associates, num_im, **kwargs):
    ratios=[]
    for num in range(1, num_im+1):
        associates_list=make_lists(num, grouped_associates)
        double=0
        for row in associates_list:
            if ',' in row[2]:
                double+=1
        ratios.append((len(associates_list)-double)/double)
    return ratios

def len_cilia_to_size_nucleus(grouped_cell, grouped_cilia, grouped_valid_cilia, num_im, **kwargs):
    result=[]
    for num in range(1, num_im+1):
        cell_li=make_lists(num, grouped_cell)
        avg_nuc = sum(x[1] for x in cell_li)/len(cell_li)

        valid_cilia=make_lists(num, grouped_valid_cilia, '0')
        valid_cilia=set(x[1] for x in valid_cilia) # Column 1 contains cilia number 
        cilia_li=make_lists(num, grouped_cilia)
        
        cilia_lens=[]
        for cilia in cilia_li:
            if int(cilia[0]) not in valid_cilia:
                continue
            cilia_lens.append(cilia[15])
        avg_cilia = sum(cilia_lens)/len(cilia_lens)
        result.append(avg_cilia/avg_nuc)
    return result      

def avg_blank_cilia(grouped_cilia, grouped_valid_cilia, num_im, col_idx, **kwargs):
    result=[]
    for num in range(1, num_im+1):
        valid_cilia=make_lists(num, grouped_valid_cilia, '0')
        valid_cilia=set(x[1] for x in valid_cilia) # Column 1 contains cilia number 
        cilia_li=make_lists(num, grouped_cilia)
        
        cilia_size=[]
        for cilia in cilia_li:
            if int(cilia[0]) not in valid_cilia:
                continue
            cilia_size.append(cilia[col_idx+1])
        result.append(sum(cilia_size)/len(cilia_size))
    return result   

def main():

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

    dispatch_dict = {
        'Num nuclei per im': num_nuc_per_im,
        'Nuclei with 2 cent/Nuclei with 1 cent': single_cent_to_two_cent,
        'Num cilia per im': num_cilia_per_im,
        'Avg len of cilia/size of nuclei': len_cilia_to_size_nucleus,
    }

    dispatch_dict = {
        **dispatch_dict,
        **{f'Avg {col} of cilia':  partial(avg_blank_cilia, col_idx=idx) for idx, col in enumerate(cols_cilia)}
    }

    
    p = make_figure()
    histogram = ColumnDataSource({'top': [], 'left': [], 'right': []})
    p.quad(source=histogram, top='top', left='left', right='right', bottom=0)

    def selection_callback(event):
        # print(event.item)
        # print(histogram.data_source)
        # print(histogram.data_source.data)
        new_data = dispatch_dict[event.item](
            num_im=num_im,
            image_df=image_df,
            grouped_associates=grouped_associates,
            grouped_cell=grouped_cell,
            grouped_cilia=grouped_cilia,
            grouped_valid_cilia=grouped_valid_cilia
        )
        histogram.data = {
            'left': [i for i in range(len(new_data))],
            'right': [i+1 for i in range(len(new_data))],
            'top': new_data
        }

        p.yaxis.axis_label = event.item

    dropdown = Dropdown(
        label='Summary Statistic',
        menu=[(key, key) for key in dispatch_dict]
    )
    dropdown.on_click(selection_callback)
    layout = column(dropdown, p)
    curdoc().add_root(layout)

main()
