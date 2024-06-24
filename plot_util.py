import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def find_centroids(df_membership, x_pca):
    """
    Args:
    df_membership: (100, 3) df
    x_pca: (100, 2) np array
    """
    # Ground the index of embeddings based on the clusters
    cluster_dict = {}
    for i in range(1,3):
        cluster_dict[i] = []
        for j in range(100):
            if df_membership.iloc[j, 2] == i:
                cluster_dict[i].append(j)
    # Find the centroid of each cluster
    centroids = []
    for i in range(1,3):
        cluster = cluster_dict[i]
        centroid = np.mean(x_pca[cluster], axis=0)
        centroids.append(centroid)
    centroids = np.array(centroids)
    return centroids

def plot_2d_chart_ground_truth(origin_embeddings, ground_truth, save_path):
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(origin_embeddings)

    plt.figure(figsize=(8, 6))
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")

    plt.title("Ground Truth")
    ground_truth = [1 if x == 1 else 2 for x in ground_truth]
    scatter = plt.scatter(x_pca[:, 0], x_pca[:, 1], c=ground_truth, cmap='viridis', edgecolor='k')
    legend = plt.legend(*scatter.legend_elements(), title="Classes")
    plt.add_artist(legend)

    plt.savefig(save_path)

def plot_2d_chart_with_different_fz(origin_embeddings, df_membership_dict, save_path):
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    pca =  PCA(n_components=2)
    x_pca = pca.fit_transform(origin_embeddings)

    for i, (fz, membership) in enumerate(df_membership_dict.items()):
        ax = axs[i//2, i%2]
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")

        centroids = find_centroids(membership, x_pca)

        ax.set_title(f"Fuzzy Membership with Fuzzy Factor = {fz}")
        scatter = ax.scatter(x_pca[:, 0], x_pca[:, 1], c=membership.iloc[:, 1], cmap='viridis', edgecolor='k')
        ax.scatter([x[0] for x in centroids], [x[1] for x in centroids], c='red', s=100, alpha=1, marker='x') # Plot the centroids
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Membership')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(save_path)

def plot_2d_chart_membership(origin_embeddings, df_membership, save_path):
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(origin_embeddings)

    for i in range(2):
        axs[i].set_xlabel("PC 1")
        axs[i].set_ylabel("PC 2")

    # Plot the original embeddings colored with ground truth
    axs[0].set_title("Ground Truth")
    ground_truth = [1 if x == 1 else 2 for x in ground_truth]
    scatter = axs[0].scatter(x_pca[:, 0], x_pca[:, 1], c=ground_truth, cmap='viridis', edgecolor='k')
    legend = axs[0].legend(*scatter.legend_elements(), title="Classes")
    axs[0].add_artist(legend)

    # Ground the index of embeddings based on the clusters
    cluster_dict = {}
    for i in range(1,3):
        cluster_dict[i] = []
        for j in range(100):
            if df_membership.iloc[j, 2] == i:
                cluster_dict[i].append(j)
    # Find the centroid of each cluster
    centroids = []
    for i in range(1,3):
        cluster = cluster_dict[i]
        centroid = np.mean(x_pca[cluster], axis=0)
        centroids.append(centroid)
    centroids = np.array(centroids)

    # Plot the embeddings with intermediately colored clusters
    df_copy = df_membership.copy()
    axs[1].set_title("Fuzzy Membership")
    scatter = axs[1].scatter(x_pca[:, 0], x_pca[:, 1], c=df_copy.iloc[:, 1], cmap='viridis', edgecolor='k')
    axs[1].scatter([x[0] for x in centroids], [x[1] for x in centroids], c='red', s=100, alpha=1, marker='x') # Plot the centroids
    cbar = plt.colorbar(scatter, ax=axs[1])
    cbar.set_label('Membership')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(save_path)

def plot_2d_charts(origin_embeddings, df_membership, ground_truth, fig_name):
    """
    Args: 
    origin_embeddings: (100, 768)
    df_membership: (100, 3) df
    ground_truth: (100,)
    """
    # 3 subplots
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    # Apply pca to the original embeddings
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(origin_embeddings)

    for i in range(3):
        axs[i].set_xlabel("Principal Component 1")
        axs[i].set_ylabel("Principal Component 2")

    # Plot the original embeddings colored with ground truth
    axs[0].set_title("Ground Truth")
    ground_truth = [1 if x == 1 else 2 for x in ground_truth]
    scatter = axs[0].scatter(x_pca[:, 0], x_pca[:, 1], c=ground_truth, cmap='viridis', edgecolor='k')
    legend = axs[0].legend(*scatter.legend_elements(), title="Classes")
    axs[0].add_artist(legend)

    # Ground the index of embeddings based on the clusters
    cluster_dict = {}
    for i in range(1,3):
        cluster_dict[i] = []
        for j in range(100):
            if df_membership.iloc[j, 2] == i:
                cluster_dict[i].append(j)
    # Find the centroid of each cluster
    centroids = []
    for i in range(1,3):
        cluster = cluster_dict[i]
        centroid = np.mean(x_pca[cluster], axis=0)
        centroids.append(centroid)
    centroids = np.array(centroids)
    
    # Plot the original embeddings with color representing the cluster
    axs[1].set_title("Fuzzy Clusters")
    scatter = axs[1].scatter(x_pca[:, 0], x_pca[:, 1], c=df_membership.iloc[:, 2], cmap='viridis', edgecolor='k')
    axs[1].scatter([x[0] for x in centroids], [x[1] for x in centroids], c='red', s=100, alpha=1, marker='x') # Plot the centroids
    legend = axs[1].legend(*scatter.legend_elements(), title="Clusters")
    axs[1].add_artist(legend)

    # Plot the embeddings with intermediately colored clusters
    df_copy = df_membership.copy()
    axs[2].set_title("Fuzzy Membership")
    scatter = axs[2].scatter(x_pca[:, 0], x_pca[:, 1], c=df_copy.iloc[:, 1], cmap='viridis', edgecolor='k')
    axs[2].scatter([x[0] for x in centroids], [x[1] for x in centroids], c='red', s=100, alpha=1, marker='x') # Plot the centroids
    legend = axs[2].legend(*scatter.legend_elements(), title="Membership")
    axs[2].add_artist(legend)

    # Plot the confusion matrix
    axs[3].set_title("Confusion Matrix")
    confusion_matrix = np.zeros((2,2))
    for i in range(100):
        if df_membership.iloc[i, 2] == 1:
            if ground_truth[i] == 1:
                confusion_matrix[0,0] += 1
            else:
                confusion_matrix[0,1] += 1
        else:
            if ground_truth[i] == 1:
                confusion_matrix[1,0] += 1
            else:
                confusion_matrix[1,1] += 1
    df_cm = pd.DataFrame(confusion_matrix, index = ["Normal", "Abnormal"],
                        columns = ["1", "2"])
    sn.heatmap(df_cm, annot=True, ax=axs[3])
    sn.set(font_scale=1)
    axs[3].set_xlabel("Predicted")
    axs[3].set_ylabel("Ground Truth")

    fig.suptitle(fig_name, fontsize=16)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(result_dir + fig_name.replace(" ", "_").replace("=", "_") + ".png")

def plot_2d_charts_with_mixed(origin_embeddings, df_membership, ground_truth, fig_name):
    """
    Args: 
    origin_embeddings: (100, 768)
    df_membership: (100, 3) df
    ground_truth: (100,)
    """
    # 3 subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 16))

    # Apply pca to the original embeddings
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(origin_embeddings)

    for i in range(2):
        for j in range(2):
            axs[i, j].set_xlabel("PC 1")
            axs[i, j].set_ylabel("PC 2")
    # Plot the original embeddings colored with ground truth
    ax = axs[0, 0]
    ax.set_title("Ground Truth")
    ground_truth = [1 if x == 1 else 2 for x in ground_truth]
    scatter = ax.scatter(x_pca[:, 0], x_pca[:, 1], c=ground_truth, cmap='viridis', edgecolor='k')
    legend = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend)

    centroids = find_centroids(df_membership, x_pca)

    # Group the the mixed class based on the first 2 columns of the membership matrix
    # Iterate the membership df
    for i in range(100):
        if df_membership.iloc[i, 1] > 0.4 and df_membership.iloc[i, 1] < 0.6:
            # Set the label to 0
            df_membership.iloc[i, 2] = 0

    # Plot the original embeddings with color representing the cluster
    ax = axs[0, 1]
    ax.set_title("Fuzzy Clusters")
    scatter = ax.scatter(x_pca[:, 0], x_pca[:, 1], c=df_membership.iloc[:, 2], cmap='viridis', edgecolor='k')
    ax.scatter([x[0] for x in centroids], [x[1] for x in centroids], c='red', s=100, alpha=1, marker='x') # Plot the centroids
    legend_elements = scatter.legend_elements()
    legend_elements[1][0] = "mixed"
    legend = ax.legend(*legend_elements, title="Clusters")
    ax.add_artist(legend)

    # Plot the embeddings with intermediately colored clusters
    ax = axs[1, 0]
    df_copy = df_membership.copy()
    ax.set_title("Fuzzy Membership")
    scatter = ax.scatter(x_pca[:, 0], x_pca[:, 1], c=df_copy.iloc[:, 1], cmap='viridis', edgecolor='k')
    ax.scatter([x[0] for x in centroids], [x[1] for x in centroids], c='red', s=100, alpha=1, marker='x') # Plot the centroids
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Membership')

    # Plot the confusion matrix
    ax = axs[1, 1]
    ax.set_title("Confusion Matrix")
    confusion_matrix = np.zeros((2,3))
    for i in range(100):
        if df_membership.iloc[i, 2] == 1:
            if ground_truth[i] == 1:
                confusion_matrix[0,0] += 1
            else:
                confusion_matrix[0,2] += 1
        elif df_membership.iloc[i, 2] == 2:
            if ground_truth[i] == 1:
                confusion_matrix[1,0] += 1
            else:
                confusion_matrix[1,2] += 1
        else:
            if ground_truth[i] == 1:
                confusion_matrix[0,1] += 1
            else:
                confusion_matrix[1,1] += 1
    df_cm = pd.DataFrame(confusion_matrix, index = ["Normal", "Abnormal"],
                        columns = ["1","mixed", "2"])
    sn.heatmap(df_cm, annot=True, ax=ax)
    sn.set(font_scale=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")

    #fig.suptitle(fig_name, fontsize=16)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(fig_name.replace(" ", "_").replace("=", "_") + ".png")