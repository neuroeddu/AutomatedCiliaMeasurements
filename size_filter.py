import pandas as pd 

csv_path='/Users/sneha/Desktop/mni/cilia_09:12:2021/im_output/MyExpt_Centriole.csv'

df = pd.read_csv(csv_path)

df=df[df['AreaShape_Area']<2000]

df.to_csv('/Users/sneha/Desktop/plswork2.csv')