import pandas as pd
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import argparse
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

    measurements_nuc = pd.read_csv(
        join(args["measurements"], "MyExpt_Nucleus.csv"),
        skipinitialspace=True,
        usecols=fields,
    )

    measurements_cent = pd.read_csv(
        join(args["measurements"], "MyExpt_Centriole.csv"),
        skipinitialspace=True,
        usecols=fields,
    )

    c2c_pairings = pd.read_csv(args["c2c"], skipinitialspace=True)

    scores, clf, pca_2d, pca_7d, c2c_pairings = setup_for_clustering(
        c2c_pairings, tuned_parameters
    )

    full_df = normalize_and_clean(
        measurements_nuc, measurements_cilia, measurements_cent, c2c_pairings
    )

    if args.get("umap"):
        umap_(full_df, args.get("output"))
    if args.get("pca_features"):
        pca_features(full_df, pca_7d, args.get("output"))
    if args.get("heirarchical"):
        heirarchical_clustering(full_df, args.get("output"))
    if args.get("xmeans"):
        xmeans(full_df, clf, pca_2d, args.get("output"))


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

    # Set up the K-Means/scaling/PCA for visualization
    scores = ["precision", "recall"]
    clf = GridSearchCV(KMeans(), tuned_parameters)
    pca_2d = PCA(n_components=2)
    pca_7d = PCA(n_components=7)

    return scores, clf, pca_2d, pca_7d, c2c_pairings


def normalize_and_clean(
    measurements_nuc, measurements_cilia, measurements_cent, c2c_pairings
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

    # Merge so we get the list of all measurements we desire
    full_df = c2c_pairings.merge(measurements_cilia, on=["ImageNumber", "Cilia"])
    full_df = full_df.merge(measurements_nuc, on=["ImageNumber", "Nucleus"])

    cent1_na = full_df[full_df["Cent1"].isna()]
    full_df = full_df.merge(measurements_cent_1, on=["ImageNumber", "Cent1"])
    full_df = pd.concat([full_df, cent1_na], ignore_index=True)

    cent2_na = full_df[full_df["Cent2"].isnull()]
    full_df = full_df.merge(measurements_cent_2, on=["ImageNumber", "Cent2"])
    full_df = pd.concat([full_df, cent2_na], ignore_index=True)

    full_df.drop("ImageNumber", axis=1, inplace=True)
    # Prepare for clustering via scaling and dropping none values
    full_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # full_df.dropna(inplace=True)
    full_df.fillna(0, inplace=True)

    return full_df


def umap_(full_df, output):
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(full_df)
    fig1, ax1 = plt.subplots()
    plt.scatter(embedding[:, 0], embedding[:, 1], cmap="Spectral", s=5)
    plt.gca().set_aspect("equal", "datalim")
    plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
    plt.title(f"UMAP projection", fontsize=24)
    plt.savefig(join(output, f"UMAP.png"))
    plt.close()


def pca_features(full_df, pca_7d, output):
    x_new = pca_7d.fit_transform(full_df)
    components_list = abs(pca_7d.components_)
    columns_mapping = list(full_df.columns)

    for component in components_list:
        component = component.tolist()
        max_value = max(component)

    with open(join(output, f"pca_features.txt"), "w") as f:
        f.write(f"the important features for each principal component are: ")
        f.write(columns_mapping[component.index(max_value)])


def heirarchical_clustering(full_df, output):
    plt.figure(figsize=(10, 7))
    plt.title(f"Dendrogram")
    dend = shc.dendrogram(shc.linkage(full_df, method="ward"))
    plt.xlabel("Samples")
    plt.ylabel("Distance between samples")
    plt.savefig(join(output, f"dendrogram.png"))
    plt.close()


def xmeans(full_df, clf, pca_2d, output):
    # Perform X-Means
    clf.fit(full_df)
    with open(join(output, f"mean_val_features.txt"), "a+") as f:
        # Print out best result of K-Means
        f.write(f"for all images:\n")  # 3,4,5
        params = clf.best_params_  # n_clusters=3
        best_clf = clf.best_estimator_  # KMeans(n_clusters=3)

        num_clusters = params["n_clusters"]
        f.write(f"Best number of clusters is {num_clusters}\n")

        y_kmeans = best_clf.predict(full_df)
        full_df["Cluster"] = y_kmeans
        for cluster in range(num_clusters):
            cluster_df = full_df[full_df["Cluster"] == cluster]
            mean_df = cluster_df.mean()
            mean_df = mean_df.to_string()
            f.write(f"The mean values for features in cluster {cluster} are\n")
            f.write(mean_df)
            f.write("\n*****************************************\n")

    # Perform PCA to get the data in a reduced form
    PCs_2d = pd.DataFrame(pca_2d.fit_transform(full_df.drop(["Cluster"], axis=1)))
    PCs_2d.columns = ["PC1_2d", "PC2_2d"]
    full_df = pd.concat([full_df, PCs_2d], axis=1, join="inner")

    full_df.to_csv(join(output, f"xmeans_data.csv"))
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

    title = f"Visualizing Clusters in Two Dimensions Using PCA"

    layout = dict(
        title=title,
        xaxis=dict(title="PC1", ticklen=5, zeroline=False),
        yaxis=dict(title="PC2", ticklen=5, zeroline=False),
    )

    fig = go.Figure(dict(data=clusters_li, layout=layout))

    fig.write_html(join(output, f"xmeans.html"))


if __name__ == "__main__":
    main()