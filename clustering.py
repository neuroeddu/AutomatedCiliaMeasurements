import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

################################# TO CHANGE #################################
cilia_csv_path = (
    "/Users/sneha/Desktop/ciliaNov22/spreadsheets_im_output/MyExpt_Cilia.csv"
)
valid_cilia = "/Users/sneha/Desktop/ciliaNov22/new_cilia.csv"
tuned_parameters = [{"n_clusters": [3, 5, 1, 50, 7, 9, 2, 4, 25, 6, 8]}]
################################# TO CHANGE #################################

# Convert the CSVs into dataframes and group by image
measurements_df = pd.read_csv(cilia_csv_path, skipinitialspace=True)
num_im = measurements_df.ImageNumber.iat[-1]
grouped_measurements = measurements_df.groupby(["ImageNumber"])

valid_cilia_df = pd.read_csv(valid_cilia, skipinitialspace=True)
grouped_valid_cilia = valid_cilia_df.groupby(["0"])

# Set up the K-Means/scaling
scores = ["precision", "recall"]
scaler = StandardScaler()
clf = GridSearchCV(KMeans(), tuned_parameters)

for num in range(1, num_im + 1):
    # Get correct groups
    measurements_df = grouped_measurements.get_group(num_im)
    valid_df = grouped_valid_cilia.get_group(num_im)

    # Prepare to merge
    valid_df = valid_df.rename(columns={"0": "ImageNumber", "1": "ObjectNumber"})
    measurements_df.drop("ImageNumber", axis=1, inplace=True)
    valid_df.drop("ImageNumber", axis=1, inplace=True)

    # Merge so we get the list of valid cilia with all their measurements
    full_df = valid_df.merge(measurements_df, on=["ObjectNumber"])
    full_df.drop(
        columns=[
            "Location_Center_X",
            "Location_Center_Y",
            "Location_Center_Z",
            "ObjectNumber",
        ],
        inplace=True,
    )

    # Prepare for K-Means
    full_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    full_df.dropna(inplace=True)

    # K-Means
    scaled_features = scaler.fit_transform(full_df)
    clf.fit(scaled_features)

    # Print out best result of K-Means
    print(f"for image {num}:")
    print(clf.best_params_)
    best_clf = clf.best_estimator_
    print(best_clf.cluster_centers_)
    print(best_clf.n_iter_)

    # Make mapping of cilia num to cluster
    cluster_map = pd.DataFrame()
    cluster_map["ObjectNumber"] = full_df.index.values
    cluster_map["cluster"] = best_clf.labels_
    measurements_df = measurements_df[
        ["Location_Center_X", "Location_Center_Y", "ObjectNumber"]
    ]
    full_df = cluster_map.merge(measurements_df, on=["ObjectNumber"])

    # Plot
    plt.scatter(
        full_df["Location_Center_X"].values.tolist(),
        full_df["Location_Center_Y"].values.tolist(),
        c=full_df["cluster"].values.tolist(),
    )
    plt.show()
