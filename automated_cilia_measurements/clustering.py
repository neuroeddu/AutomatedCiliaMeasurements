import pandas as pd
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import argparse
from sklearn.preprocessing import normalize
from os.path import join
import plotly.graph_objs as go
from plotly.offline import plot
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
import umap.umap_ as umap


def parse_args():
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--measurements", help="path to CellProfiler CSVs", required=True
    )

    parser.add_argument("-c", "--c2c", help="path to c2c CSV", required=True)
    parser.add_argument(
        "-x", "--xmeans", help="whether xmeans should be included", required=False
    )
    parser.add_argument(
        "-p",
        "--pca_features",
        help="whether pca features should be included",
        required=False,
    )
    parser.add_argument(
        "-hr",
        "--heirarchical",
        help="whether dendrograms should be included",
        required=False,
    )

    parser.add_argument(
        "-u",
        "--umap",
        help="whether umap should be included",
        required=False,
    )

    parser.add_argument(
        "-o",
        "--output",
        help="output folder to save clustering results to",
        required=True,
    )

    return vars(parser.parse_args())


def main(**args):
    args = args or parse_args()
    # params we want to check
    tuned_parameters = [{"n_clusters": [2, 3, 4, 5, 6, 7, 8]}]

    fields = [
        "ImageNumber",
        "ObjectNumber",
        "AreaShape_Area",
        "AreaShape_Compactness",
        "AreaShape_Eccentricity",
        "AreaShape_EquivalentDiameter",
        "AreaShape_EulerNumber",
        "AreaShape_Extent",
        "AreaShape_FormFactor",
        "AreaShape_MajorAxisLength",
        "AreaShape_MaxFeretDiameter",
        "AreaShape_MaximumRadius",
        "AreaShape_MeanRadius",
        "AreaShape_MedianRadius",
        "AreaShape_MinFeretDiameter",
        "AreaShape_MinorAxisLength",
        "AreaShape_Orientation",
        "AreaShape_Perimeter",
        "AreaShape_Solidity",
    ]

    # Convert the CSVs into dataframes and group by image
    measurements_cilia = pd.read_csv(
        join(args["measurements"], "MyExpt_Cilia.csv"),
        skipinitialspace=True,
        usecols=fields,
    )
    num_im = measurements_cilia.ImageNumber.iat[-1]
    grouped_measurements_cilia = measurements_cilia.groupby(["ImageNumber"])

    measurements_nuc = pd.read_csv(
        join(args["measurements"], "MyExpt_Nucleus.csv"),
        skipinitialspace=True,
        usecols=fields,
    )
    grouped_measurements_nuc = measurements_nuc.groupby(["ImageNumber"])

    measurements_cent = pd.read_csv(
        join(args["measurements"], "MyExpt_Centriole.csv"),
        skipinitialspace=True,
        usecols=fields,
    )
    grouped_measurements_cent = measurements_cent.groupby(["ImageNumber"])

    c2c_pairings = pd.read_csv(args["c2c"], skipinitialspace=True)

    scores, clf, pca_2d, pca_7d, grouped_c2c = setup_for_clustering(
        c2c_pairings, tuned_parameters
    )

    for num in range(1, num_im + 1):
        # Get correct groups
        measurements_nuc = grouped_measurements_nuc.get_group(num)
        measurements_cilia = grouped_measurements_cilia.get_group(num)
        measurements_cent = grouped_measurements_cent.get_group(num)
        c2c_df = grouped_c2c.get_group(num)

        full_df, og_df = normalize_and_clean(
            measurements_nuc, measurements_cilia, measurements_cent, c2c_df
        )
        # Assume we do not have xmeans until proven otherwise
        if args.get("umap"):
            clusters = None
        if args.get("xmeans"):
            clusters = xmeans(full_df, clf, num, pca_2d, args.get("output"), og_df)
        # want to use clusters if exists else none
        if args.get("umap"):
            umap_(full_df, num, args.get("output"), clusters)
        if args.get("pca_features"):
            pca_features(full_df, pca_7d, num, args.get("output"))
        if args.get("heirarchical"):
            heirarchical_clustering(full_df, num, args.get("output"))


def setup_for_clustering(c2c_pairings, tuned_parameters):
    c2c_pairings["Centriole"] = (
        c2c_pairings["Centriole"].fillna("[]").apply(lambda x: eval(x))
    )
    c2c_pairings["PathLengthCentriole"] = (
        c2c_pairings["PathLengthCentriole"].fillna("[]").apply(lambda x: eval(x))
    )

    # Edit c2c data to separate centrioles into two columns
    split_df = pd.DataFrame(
        c2c_pairings["Centriole"].to_list(), columns=["Cent1", "Cent2"]
    )
    split_df_2 = pd.DataFrame(
        c2c_pairings["PathLengthCentriole"].to_list(),
        columns=["PathCent1", "PathCent2"],
    )
    c2c_pairings = pd.concat([c2c_pairings, split_df], axis=1)
    c2c_pairings = pd.concat([c2c_pairings, split_df_2], axis=1)
    c2c_pairings = c2c_pairings.drop(["Centriole", "PathLengthCentriole"], axis=1)

    grouped_c2c = c2c_pairings.groupby(["ImageNumber"])

    # Set up the K-Means/scaling/PCA for visualization
    scores = ["precision", "recall"]
    clf = GridSearchCV(KMeans(), tuned_parameters)
    pca_2d = PCA(n_components=2)
    pca_7d = PCA(n_components=7)

    return scores, clf, pca_2d, pca_7d, grouped_c2c


def normalize_and_clean(
    measurements_nuc, measurements_cilia, measurements_cent, c2c_df
):

    # Prepare to merge
    measurements_nuc = measurements_nuc.rename(
        columns={
            "ObjectNumber": "Nucleus",
            "AreaShape_Area": "NucArea",
            "AreaShape_Compactness": "NucCompactness",
            "AreaShape_Eccentricity": "NucEccentricity",
            "AreaShape_EquivalentDiameter": "NucEquivDiameter",
            "AreaShape_EulerNumber": "NucEulerNum",
            "AreaShape_Extent": "NucExtent",
            "AreaShape_FormFactor": "NucFormFactor",
            "AreaShape_MajorAxisLength": "NucMajorAxisLength",
            "AreaShape_MaxFeretDiameter": "NucMaxFeretDiameter",
            "AreaShape_MaximumRadius": "NucMaxRadius",
            "AreaShape_MeanRadius": "NucMeanRadius",
            "AreaShape_MedianRadius": "NucMedianRadius",
            "AreaShape_MinFeretDiameter": "NucMinFeretDiameter",
            "AreaShape_MinorAxisLength": "NucMinorAxisLength",
            "AreaShape_Orientation": "NucOrientation",
            "AreaShape_Perimeter": "NucPerimeter",
            "AreaShape_Solidity": "NucSolidity",
        }
    )

    measurements_cilia = measurements_cilia.rename(
        columns={
            "ObjectNumber": "Cilia",
            "AreaShape_Area": "CiliaArea",
            "AreaShape_Compactness": "CiliaCompactness",
            "AreaShape_Eccentricity": "CiliaEccentricity",
            "AreaShape_EquivalentDiameter": "CiliaEquivDiameter",
            "AreaShape_EulerNumber": "CiliaEulerNum",
            "AreaShape_Extent": "CiliaExtent",
            "AreaShape_FormFactor": "CiliaFormFactor",
            "AreaShape_MajorAxisLength": "CiliaMajorAxisLength",
            "AreaShape_MaxFeretDiameter": "CiliaMaxFeretDiameter",
            "AreaShape_MaximumRadius": "CiliaMaxRadius",
            "AreaShape_MeanRadius": "CiliaMeanRadius",
            "AreaShape_MedianRadius": "CiliaMedianRadius",
            "AreaShape_MinFeretDiameter": "CiliaMinFeretDiameter",
            "AreaShape_MinorAxisLength": "CiliaMinorAxisLength",
            "AreaShape_Orientation": "CiliaOrientation",
            "AreaShape_Perimeter": "CiliaPerimeter",
            "AreaShape_Solidity": "CiliaSolidity",
        }
    )

    measurements_cent_1 = measurements_cent.rename(
        columns={
            "ObjectNumber": "Cent1",
            "AreaShape_Area": "CentArea1",
            "AreaShape_Compactness": "CentCompactness1",
            "AreaShape_Eccentricity": "CentEccentricity1",
            "AreaShape_EquivalentDiameter": "CentEquivDiameter1",
            "AreaShape_EulerNumber": "CentEulerNum1",
            "AreaShape_Extent": "CentExtent1",
            "AreaShape_FormFactor": "CentFormFactor1",
            "AreaShape_MajorAxisLength": "CentMajorAxisLength1",
            "AreaShape_MaxFeretDiameter": "CentMaxFeretDiameter1",
            "AreaShape_MaximumRadius": "CentMaxRadius1",
            "AreaShape_MeanRadius": "CentMeanRadius1",
            "AreaShape_MedianRadius": "CentMedianRadius1",
            "AreaShape_MinFeretDiameter": "CentMinFeretDiameter1",
            "AreaShape_MinorAxisLength": "CentMinorAxisLength1",
            "AreaShape_Orientation": "CentOrientation1",
            "AreaShape_Perimeter": "CentPerimeter1",
            "AreaShape_Solidity": "CentSolidity1",
        }
    )
    measurements_cent_2 = measurements_cent.rename(
        columns={
            "ObjectNumber": "Cent2",
            "AreaShape_Area": "CentArea2",
            "AreaShape_Compactness": "CentCompactness2",
            "AreaShape_Eccentricity": "CentEccentricity2",
            "AreaShape_EquivalentDiameter": "CentEquivDiameter2",
            "AreaShape_EulerNumber": "CentEulerNum2",
            "AreaShape_Extent": "CentExtent2",
            "AreaShape_FormFactor": "CentFormFactor2",
            "AreaShape_MajorAxisLength": "CentMajorAxisLength2",
            "AreaShape_MaxFeretDiameter": "CentMaxFeretDiameter2",
            "AreaShape_MaximumRadius": "CentMaxRadius2",
            "AreaShape_MeanRadius": "CentMeanRadius2",
            "AreaShape_MedianRadius": "CentMedianRadius2",
            "AreaShape_MinFeretDiameter": "CentMinFeretDiameter2",
            "AreaShape_MinorAxisLength": "CentMinorAxisLength2",
            "AreaShape_Orientation": "CentOrientation2",
            "AreaShape_Perimeter": "CentPerimeter2",
            "AreaShape_Solidity": "CentSolidity2",
        }
    )

    measurements_nuc.drop("ImageNumber", axis=1, inplace=True)
    measurements_cilia.drop("ImageNumber", axis=1, inplace=True)
    measurements_cent.drop("ImageNumber", axis=1, inplace=True)
    c2c_df.drop("ImageNumber", axis=1, inplace=True)

    # Merge so we get the list of all measurements we desire
    full_df = c2c_df.merge(measurements_cilia, on=["Cilia"])
    full_df = full_df.merge(measurements_nuc, on=["Nucleus"])

    cent1_na = full_df[full_df["Cent1"].isna()]
    full_df = full_df.merge(measurements_cent_1, on=["Cent1"])
    full_df = pd.concat([full_df, cent1_na], ignore_index=True)

    cent2_na = full_df[full_df["Cent2"].isnull()]
    full_df = full_df.merge(measurements_cent_2, on=["Cent2"])
    full_df = pd.concat([full_df, cent2_na], ignore_index=True)

    # Prepare for clustering via scaling and dropping none values
    full_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    full_df.fillna(0, inplace=True)

    # Normalize data and merge column names back in
    cols = list(full_df.columns)
    cols = ["to_del"] + cols[
        1:
    ]  # NOTE this is done because pandas includes the index column
    normalized_df = normalize(full_df)
    normalized_df = pd.DataFrame(normalized_df, columns=cols)
    normalized_df.drop(columns=["to_del"], axis=0, inplace=True)
    return normalized_df, full_df


def umap_(full_df, num, output, clusters):
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(full_df)
    # fig1, ax1 = plt.subplots()
    if clusters:
        plt.scatter(embedding[:, 0], embedding[:, 1], c=clusters, cmap="Spectral", s=5)
    else:
        plt.scatter(embedding[:, 0], embedding[:, 1], cmap="Spectral", s=5)
    plt.gca().set_aspect("equal", "datalim")
    plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
    title_cluster = " with XMeans clusters" if clusters else ""
    plt.title(f"UMAP projection for Image {num}{title_cluster}", fontsize=24)

    save_name_cluster = "_with_XMeans_clusters" if clusters else ""
    plt.savefig(join(output, f"UMAP_im_{num}{save_name_cluster}.png"))
    plt.close()


def top_list(pc, n):
    top_list = []
    # top list should be a tuple of the form (index, number)
    # index so we can find it later and number so we can cont with top list

    for cur_index, cur_elem in enumerate(pc):
        added = False
        # If the current word in the dictionary is bigger than the one in the list add it here
        for li_index, (old_index, old_elem) in enumerate(top_list):
            if cur_elem >= old_elem:
                top_list = (
                    top_list[:li_index] + [(cur_index, cur_elem)] + top_list[li_index:]
                )
                added = True
                break

        # If we get through the whole top 10 list and we haven't added anything, we must be smaller than everything or the list
        # may be less than the number of elem, so add indiscriminately
        if not added:
            top_list.append((cur_index, cur_elem))

        # Remove the smallest element in the list if the list is already full
        if len(top_list) > n:
            top_list.pop()

    return top_list


def pca_features(full_df, pca_7d, num, output):

    # Perform 7d PCA
    _ = pca_7d.fit_transform(full_df)
    components_list = abs(pca_7d.components_)
    columns_mapping = list(full_df.columns)

    pc_components = []
    # Find top five elem in each PC
    for component in components_list:
        component = component.tolist()
        sorted_components = top_list(component, 5)
        pc_components.append(sorted_components)

    # Print to file
    with open(join(output, f"pca_features_im_{num}.txt"), "w") as f:
        f.write(f"the 5 important features for each principal component are: ")
        for pc_num, sorted_components in enumerate(pc_components):
            f.write(f"PC number {pc_num}:\n")
            for index, _ in sorted_components:
                f.write(f" {columns_mapping[index]}\n")


def heirarchical_clustering(full_df, num, output):
    plt.figure(figsize=(10, 7))
    plt.title(f"Dendrogram for Image {num}")
    dend = shc.dendrogram(shc.linkage(full_df, method="ward"))
    plt.xlabel("Samples")
    plt.ylabel("Distance between samples")
    plt.savefig(join(output, f"dendrogram_im_{num}.png"))
    plt.close()


def xmeans(full_df, clf, num, pca_2d, output, og_df):
    # Perform X-Means
    clf.fit(full_df)
    params = clf.best_params_  # n_clusters=3
    best_clf = clf.best_estimator_  # KMeans(n_clusters=3)

    num_clusters = params["n_clusters"]
    # Get the cluster numbers for each row in the data
    y_kmeans = best_clf.predict(full_df)
    y_kmeans = y_kmeans.tolist()

    # Assign cluster numbers
    og_df["Cluster"] = y_kmeans
    full_df["Cluster"] = y_kmeans

    # Write mean values for features in each cluster (using non-normalized values)
    for cluster in range(num_clusters):

        # Get only the points in one cluster
        cluster_df = og_df[og_df["Cluster"] == cluster]
        cluster_df.drop(columns=["Cluster"], inplace=True)

        # Find the mean of each feature and write to string
        mean_df = cluster_df.mean()
        
        # Combine all mean dataframes
        if cluster == 0:
            result = mean_df
        else:
            result = pd.concat([result, mean_df], axis=1 )

    result.columns =list(range(len(result.columns)))
    result = result.iloc[1: , :]

    result.to_csv(path_or_buf=join(output, "mean_val_features.csv"))
    

    # Perform PCA to get the data in a reduced form
    PCs_2d = pd.DataFrame(pca_2d.fit_transform(full_df.drop(["Cluster"], axis=1)))
    PCs_2d.columns = ["PC1_2d", "PC2_2d"]
    full_df = pd.concat([full_df, PCs_2d], axis=1, join="inner")

    # Print out data for xmeans and clusters into csv
    og_df = pd.concat([og_df, PCs_2d], axis=1, join="inner")
    og_df.to_csv(join(output, f"xmeans_data.csv"))

    # Make data points for each cluster
    clusters_li = []
    for cluster in range(num_clusters):
        color = "%06x" % random.randint(0, 0xFFFFFF)
        cluster_df = full_df[full_df["Cluster"] == cluster]
        trace = go.Scatter(
            x=cluster_df["PC1_2d"],
            y=cluster_df["PC2_2d"],
            mode="markers",
            name=f"Cluster {cluster}",
            marker=dict(color=f"#{color}"),
            text=None,
        )
        clusters_li.append(trace)

    # Finally, set up graph

    title = f"Visualizing Clusters in Two Dimensions Using PCA for Image {num}"

    layout = dict(
        title=title,
        xaxis=dict(title="PC1", ticklen=5, zeroline=False),
        yaxis=dict(title="PC2", ticklen=5, zeroline=False),
    )

    fig = go.Figure(dict(data=clusters_li, layout=layout))

    fig.write_html(join(output, f"xmeans_im_{num}.html"))

    # Return cluster so they can be used for UMAP
    return y_kmeans


if __name__ == "__main__":
    main()
