import pandas as pd
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import normalize
import argparse
from os.path import join
import plotly.graph_objs as go
from plotly.offline import plot
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
import umap.umap_ as umap


def umap_(cluster_df, output, cluster):
    for min_dist in (0.0, 0.1, 0.25, 0.5, 0.6, 0.8, 0.99):
        for metric in ('mahalanobis', 'euclidean', 'cosine', 'canberra'):
            reducer = umap.UMAP( min_dist=min_dist, metric=metric)
            embedding = reducer.fit_transform(cluster_df)
            plt.scatter(embedding[:, 0], embedding[:, 1], c=cluster, cmap="Dark2", s=5)
            plt.gca().set_aspect("equal", "datalim")
            plt.colorbar(boundaries=np.arange(9) - 0.5).set_ticks(np.arange(8))
            plt.title(f"UMAP projection", fontsize=24)
            plt.savefig(join(output, f"UMAP_{min_dist}_{metric}.pdf"), format="pdf")
            plt.close()
            return embedding

def umap_all(full_df, output, embedding):
    cols = ['CiliaArea','CiliaMajorAxisLength', 'CiliaMinorAxisLength', 'NucArea', 'PathLengthCilia', 'CiliaCent1', 'CiliaCent2'] # TODO Add paths to this
    #cols = ['CiliaArea','CiliaMajorAxisLength', 'CiliaMinorAxisLength']

    for col in cols:
        # what we really want is a function that tells us what 80% of our data falls in
        _, bins = pd.qcut(
            full_df[col], 9, labels=False, retbins=True, duplicates="drop"
        )
        vmax= bins[int(0.8*len(bins))] # Only go up to 8th decile, so that outliers are not disproportionately represented on the umap colors
        plt.scatter(embedding[:, 0], embedding[:, 1], c=full_df[col], s=1, vmax=vmax)
        plt.colorbar()
        plt.gca().set_aspect("equal", "datalim")
        plt.title(f"UMAP projection colored by {col}", fontsize=15)
        plt.savefig(join(output, f"UMAP_{col}.pdf"), format="pdf")
        plt.close()



def violin(full_df, output, cluster):
    cols = ['CiliaArea','CiliaMajorAxisLength', 'CiliaMinorAxisLength', 'NucArea', 'PathLengthCilia', 'CiliaCent1', 'CiliaCent2'] # TODO Add paths to this
    full_df['Cluster'] = cluster
    
    for col in cols:
        cluster_subset = []

        for x in np.unique(full_df[["Cluster"]].values):
            cluster_subset.append(full_df[full_df["Cluster"] == x][col].values)

        plt.violinplot(dataset=cluster_subset,positions=np.unique(full_df[["Cluster"]].values))
        plt.savefig(join(output, f"{col}_violin.pdf"), format="pdf")
        plt.close()
    
# cases:
# get thru and add something: change thing and make sure we know the index (for future reference)
# get thru and there's nothing to add: either too small or list is too small add indiscriminately
# remove smallest elem in list iff list alr full 
def top_list(pc, n):
    top_list = []
    # top list should be a tuple of the form (index, number)
    # index so we can find it later and number so we can cont with top list

    for cur_index, cur_elem in enumerate(pc):
        added = False
        # If the current word in the dictionary is bigger than the one in the list add it here
        for li_index, (old_index, old_elem) in enumerate(top_list):
            if cur_elem >= old_elem:
                top_list = top_list[:li_index]+[(cur_index, cur_elem)]+top_list[li_index:]
                added=True
                break

        # If we get through the whole top 10 list and we haven't added anything, we must be smaller than everything or the list 
        # may be less than the number of elem, so add indiscriminately
        if not added:
            top_list.append((cur_index, cur_elem))

        # Remove the smallest element in the list if the list is already full
        if len(top_list) > n:
            top_list.pop()

    return top_list


def pca_features(full_df, pca_7d, output):
    x_new = pca_7d.fit_transform(full_df)
    components_list = abs(pca_7d.components_)
    columns_mapping = list(full_df.columns)
    pc_components=[]
    for component in components_list:
        component = component.tolist()
        # Get the top five elements in component
        sorted_components = top_list(component, 5)
        pc_components.append(sorted_components)

    with open(join(output, f"pca_features.txt"), "w") as f:
        f.write(f"the 5 important features for each principal component are: ")
        for pc_num, sorted_components in enumerate(pc_components):
            f.write(f"PC number {pc_num}:\n")
            for index, _ in sorted_components:
                f.write(f" {columns_mapping[index]}\n")


def heirarchical_clustering(full_df, output):
    plt.figure(figsize=(10, 7))
    plt.title(f"Dendrogram")
    dend = shc.dendrogram(shc.linkage(full_df, method="ward"))
    plt.xlabel("Samples")
    plt.ylabel("Distance between samples")
    plt.savefig(join(output, f"dendrogram.pdf"), format='pdf')
    plt.close()


def xmeans(full_df, clf, pca_2d, output, dfrog):
    # Perform X-Means
    clf.fit(full_df)

    params = clf.best_params_  # n_clusters=3
    best_clf = clf.best_estimator_  # KMeans(n_clusters=3)

    num_clusters = params["n_clusters"]

    y_kmeans = best_clf.predict(full_df)
    y_kmeans = y_kmeans.tolist()

    dfrog['Cluster'] = y_kmeans
    full_df['Cluster'] = y_kmeans
    
    for cluster in range(num_clusters):
        cluster_df = dfrog[dfrog["Cluster"] == cluster]

        cluster_df.drop(columns=['Cluster'], inplace=True)
        mean_df = cluster_df.mean()

        if cluster == 0:
            result = mean_df
        else:
            result = pd.concat([result, mean_df], axis=1 )

    #print(result.columns)
    result.columns =list(range(len(result.columns)))
    result = result.iloc[1: , :]

    result.to_csv(path_or_buf=join(output, f"mean_val_features.csv"))
    

    # Perform PCA to get the data in a reduced form
    PCs_2d = pd.DataFrame(pca_2d.fit_transform(full_df.drop(["Cluster"], axis=1)))
    PCs_2d.columns = ["PC1_2d", "PC2_2d"]
    full_df = pd.concat([full_df, PCs_2d], axis=1, join="inner")
    full_dfrog = pd.concat([dfrog, PCs_2d], axis=1, join="inner")
    full_dfrog.to_csv(join(output, f"xmeans_data.csv"))
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

    return y_kmeans


def setup_for_clustering(tuned_parameters):
    # Set up the K-Means/scaling/PCA for visualization
    scores = ["precision", "recall"]
    clf = GridSearchCV(KMeans(), tuned_parameters)
    pca_2d = PCA(n_components=2)
    pca_7d = PCA(n_components=7)

    return scores, clf, pca_2d, pca_7d

tuned_parameters = [{"n_clusters": [2, 3, 4, 5, 6, 7, 8]}]
# fields for spec columns
#fields = ['CiliaArea','CiliaMajorAxisLength', 'CiliaMinorAxisLength', 'NucArea', 'PathLengthCilia', 'CiliaCent1', 'CiliaCent2', 'Cilia']

# fields for cilia
#fields = ['PathLengthCilia','Cilia','CiliaArea','CiliaCompactness','CiliaEccentricity','CiliaEquivDiameter','CiliaEulerNum','CiliaExtent','CiliaFormFactor','CiliaMajorAxisLength','CiliaMaxFeretDiameter','CiliaMaxRadius','CiliaMeanRadius','CiliaMedianRadius','CiliaMinFeretDiameter','CiliaMinorAxisLength','CiliaOrientation','CiliaPerimeter','CiliaSolidity']

# fields for all
fields = ['PathLengthCilia','Cilia','PathCent1','PathCent2','CiliaArea','CiliaCompactness','CiliaEccentricity','CiliaEquivDiameter','CiliaEulerNum','CiliaExtent','CiliaFormFactor','CiliaMajorAxisLength','CiliaMaxFeretDiameter','CiliaMaxRadius','CiliaMeanRadius','CiliaMedianRadius','CiliaMinFeretDiameter','CiliaMinorAxisLength','CiliaOrientation','CiliaPerimeter','CiliaSolidity','NucArea','NucCompactness','NucEccentricity','NucEquivDiameter','NucEulerNum','NucExtent','NucFormFactor','NucMajorAxisLength','NucMaxFeretDiameter','NucMaxRadius','NucMeanRadius','NucMedianRadius','NucMinFeretDiameter','NucMinorAxisLength','NucOrientation','NucPerimeter','NucSolidity','CentArea1','CentCompactness1','CentEccentricity1','CentEquivDiameter1','CentEulerNum1','CentExtent1','CentFormFactor1','CentMajorAxisLength1','CentMaxFeretDiameter1','CentMaxRadius1','CentMeanRadius1','CentMedianRadius1','CentMinFeretDiameter1','CentMinorAxisLength1','CentOrientation1','CentPerimeter1','CentSolidity1','CentArea2','CentCompactness2','CentEccentricity2','CentEquivDiameter2','CentEulerNum2','CentExtent2','CentFormFactor2','CentMajorAxisLength2','CentMaxFeretDiameter2','CentMaxRadius2','CentMeanRadius2','CentMedianRadius2','CentMinFeretDiameter2','CentMinorAxisLength2','CentOrientation2','CentPerimeter2','CentSolidity2','CiliaCent1', 'CiliaCent2', 'Cent1Bin', 'Cent2Bin']

# fields for cilia/nuc
#fields = ['PathLengthCilia','Cilia','CiliaArea','CiliaCompactness','CiliaEccentricity','CiliaEquivDiameter','CiliaEulerNum','CiliaExtent','CiliaFormFactor','CiliaMajorAxisLength','CiliaMaxFeretDiameter','CiliaMaxRadius','CiliaMeanRadius','CiliaMedianRadius','CiliaMinFeretDiameter','CiliaMinorAxisLength','CiliaOrientation','CiliaPerimeter','CiliaSolidity','NucArea','NucCompactness','NucEccentricity','NucEquivDiameter','NucEulerNum','NucExtent','NucFormFactor','NucMajorAxisLength','NucMaxFeretDiameter','NucMaxRadius','NucMeanRadius','NucMedianRadius','NucMinFeretDiameter','NucMinorAxisLength','NucOrientation','NucPerimeter','NucSolidity']

full_df=pd.read_csv(
        '/Users/sneha/Desktop/mni/full_df.csv', skipinitialspace=True, usecols=fields)

output='/Users/sneha/Desktop/mni/trash2/ALL'
scores, clf, pca_2d, pca_7d = setup_for_clustering(tuned_parameters)

cols=list(full_df.columns)
cols=['useless']+cols[1:]
normalized_df=normalize(full_df, axis=1)
normalized_df= pd.DataFrame(normalized_df, columns=cols)
normalized_df.drop(columns=['useless', 'Cilia'], 
                axis=0, 
                inplace=True)

normalized_df = normalized_df.dropna()


# NOTE: If you want to test something specific, comment the other lines out 
ykmeans = xmeans(normalized_df, clf, pca_2d, output, full_df)
violin(full_df, output, ykmeans)
heirarchical_clustering(full_df, output)
embedding = umap_(normalized_df, output, ykmeans)
umap_all(full_df, output, embedding)
pca_features(normalized_df, pca_7d, output)

