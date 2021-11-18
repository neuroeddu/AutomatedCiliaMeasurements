import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv('/Users/sneha/Desktop/ciliaNov11/spreadsheets_im_output/MyExpt_Nucleus.csv', usecols=['ImageNumber', 'AreaShape_Area'])


df=df[df['ImageNumber']==1]

ax = df.plot.bar(y='AreaShape_Area', rot=0)
plt.show()
