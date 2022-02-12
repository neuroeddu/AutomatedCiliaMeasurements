"""TODO clustring docstring."""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

cilia_csv_path = '/Users/sneha/Desktop/ciliaNov22/spreadsheets_im_output/MyExpt_Cilia.csv'
valid_cilia = '/Users/sneha/Desktop/ciliaNov22/new_cilia.csv'


def plot(df):
    """TODO add plot docstring."""
    plt.scatter(
        df['Location_Center_X'].values.tolist(),
        df['Location_Center_Y'].values.tolist(),
        c=df['cluster'].values.tolist(),
    )
    plt.show()


cilia_df = pd.read_csv(cilia_csv_path, skipinitialspace=True)
num_im = cilia_df.ImageNumber.iat[-1]
grouped_cilia = cilia_df.groupby(['ImageNumber'])
valid_cilia_df = pd.read_csv(valid_cilia, skipinitialspace=True)
grouped_valid_cilia = valid_cilia_df.groupby(['0'])

valid_cilia_df = valid_cilia_df.rename(columns={'0': 'ImageNumber', '1': 'ObjectNumber'})
df = valid_cilia_df.merge(cilia_df, on=['ImageNumber', 'ObjectNumber'])
df.drop(
    columns=['Location_Center_X', 'Location_Center_Y', 'Location_Center_Z', 'ObjectNumber'],
    inplace=True,
)  # dropping object number bc it is arbitrary

tuned_parameters = [
    {"n_clusters": [3, 5, 1, 50, 7, 9, 2, 4, 25, 6, 8]}  # from 2 to 10
]
scores = ["precision", "recall"]
scaler = StandardScaler()
# kmeans = KMeans(
#     init="random",
#     n_clusters=CLUSTERS,
#     n_init=10,
#     max_iter=300,
#     random_state=42
#     )
clf = GridSearchCV(KMeans(), tuned_parameters)

for num in range(1, num_im+1):
    cilia_df = grouped_cilia.get_group(num_im)

    valid_df = grouped_valid_cilia.get_group(num_im)
    valid_df = valid_df.rename(columns={'0': 'ImageNumber', '1': 'ObjectNumber'})

    cilia_df.drop('ImageNumber', axis=1, inplace=True)
    valid_df.drop('ImageNumber', axis=1, inplace=True)

    df = valid_df.merge(cilia_df, on=['ObjectNumber'])
    df.drop(
        columns=['Location_Center_X', 'Location_Center_Y', 'Location_Center_Z', 'ObjectNumber'],
        inplace=True,
    )

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    scaled_features = scaler.fit_transform(df)

    clf.fit(scaled_features)
    print(clf.best_params_)
    best_clf = clf.best_estimator_
    print(best_clf.cluster_centers_)
    print(best_clf.n_iter_)

    cluster_map = pd.DataFrame()
    cluster_map['ObjectNumber'] = df.index.values
    cluster_map['cluster'] = best_clf.labels_

    cilia_df = cilia_df[['Location_Center_X', 'Location_Center_Y', 'ObjectNumber']]
    df = cluster_map.merge(cilia_df, on=['ObjectNumber'])
    print(df.head)
    plot(df)
