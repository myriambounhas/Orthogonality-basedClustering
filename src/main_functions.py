#Main Clustering functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.neighbors import NearestNeighbors

from .ortho_definition import * 


def normalize(x):
    x_norm = (x-np.min(x))/(np.max(x)-np.min(x))
    return x_norm
    
def build_pairs_idx(X):
    """ Function that generates unique pairs (a, b) where 'a' is the nearest neighbor of 'b',
    avoiding redundant pairs like (i, j) and (j, i).
    Args:
    - X: numpy array of shape (n, m), where n is the number of instances and m is the number of features.
    Returns:
    - A numpy array of shape (k, 2) containing unique index pairs (a, b).
    """
    # Initialize the NearestNeighbors model
    neigh = NearestNeighbors(n_neighbors=2, metric="euclidean")
    neigh.fit(X)

    # Find the nearest neighbors for each instance
    _, indices = neigh.kneighbors(X)

    # Use a set to track unique pairs
    pairs = set()
    for i in range(len(X)):
        a, b = i, indices[i, 1]
        if (b, a) not in pairs:  # Avoid redundant pairs
            pairs.add((a, b))

    pairs_array = np.array(list(pairs))

    return pairs_array


# estimate the numbers of centroids then initialize them by checking the orthogonality maxima in the Ortho_matrix
def initialize_centroids (X, pairs, ortho_matrix, beta):
    """find and returns how many clusteres we need to initilize """ 
    #initial number of clusters
    K = 2   
    # Get column indices of max values
    col_indices = np.argmax(ortho_matrix, axis=1)
    
    # Create row indices
    row_indices = np.arange(ortho_matrix.shape[0])
    
    # Stack them as a 2D numpy array of (row, col) pairs
    max_indices = np.stack((row_indices, col_indices), axis=1)

    n,_ = np.shape(pairs)
    m,_ = np.shape(X)
    cluster_idx = np.full(m,-1)
    l_centroids = []
   
    # pick the two most orthogonal pairs to initialize the first TWO centroids 
    ind_max1, ind_max2 = np.unravel_index(np.argmax(ortho_matrix), ortho_matrix.shape) 
    ind_p = pairs[ind_max1]
    ind_orth_p = pairs[ind_max2]

    l_centroids.append ([X[ind_p[0]] , X[ind_p[1]]])
    l_centroids.append ([X[ind_orth_p[0]] , X[ind_orth_p[1]]])
        
    # Create a first cluster with the pair p
    cluster_idx[ind_p[0]]= 0
    cluster_idx[ind_p[1]]= 0
    
    # Create a second cluster with the second ORTOGONAL pair
    cluster_idx[ind_orth_p[0]]= 1
    cluster_idx[ind_orth_p[1]]= 1
               
    # find other centroids 
    for i, j in max_indices:
        a,b = pairs[i]
        candidate_c = True 
        for e,f in l_centroids:
            ortho = ortho_index_num (X[a], X[b], e,f)
            if (ortho <= beta):  # this pair belongs to an existing cluster
                candidate_c = False
                break

        if (candidate_c):
            cluster_idx[a]= K
            cluster_idx[b]= K
            l_centroids.append ([X[a], X[b] ])
            K+= 1           
            
        c,d = pairs[j]
        candidate_c = True
        for e,f in l_centroids:
            ortho = ortho_index_num (X[c], X[d], e,f)
            if (ortho <= beta): # this pair belongs to an existing cluster
                candidate_c = False
                break
        
        if (candidate_c):
            l_centroids.append ([X[c], X[d] ])
            cluster_idx[c]= K
            cluster_idx[d]= K
            K+=1
            #print("K: ", K, ", new centroids ", l_centroids )     
      
    centroids = np.array(l_centroids)
    #print("K: ", K, ", centroid : ", centroids )      
    return K, centroids, cluster_idx 

    
def best_centroids(a,b, X, centroids, cluster_idx, pairs, K):
    """Finds and returns the index of the most suitable cluster for a given pair (a,b) """
   
    ortho_ab = np.empty(K)
    for i in range(K):
        c,d = centroids[i]
        ortho_ab[i] = ortho_index_num (X[a], X[b], c, d)    
    return np.argmin(ortho_ab)  

def update_centroids(K, X, pairs, cluster_idx):
    """update and returns the new centroids of the clusters"""
    
    _,n = np.shape(X)
    l_centroids= []  
    arr_centroids=[] 
    
    for i in range(K):
        cluster_item_a = []
        cluster_item_b = []
        for a,b in pairs: # a,b are indices 
            if((cluster_idx[a] == i) or (cluster_idx[b] == i)):
                cluster_item_a.append(X[a]) # gather pairs item1 for the cluster i
                cluster_item_b.append(X[b]) # gather pairs item2 for the cluster i

        cluster_item_a_arr = np.array(cluster_item_a)
        cluster_item_b_arr = np.array(cluster_item_b)
                
        p1 = np.mean(cluster_item_a_arr, axis = 0)  
        p2 = np.mean(cluster_item_b_arr, axis = 0)  
        l_centroids.append ([p1, p2]) 
      
    return np.array(l_centroids)

def create_mini_clusters (X, centroids, cluster_idx, pairs, K):
    """Returns an array of cluster indices for all the data pairs"""
    
    n,_ = np.shape(pairs)
    assigned = 0
    
    for a, b in pairs: # a,b are indices 
        p = np.round (assigned/n, decimals=2)
        if((p == 0.2 ) or (p == 0.4) or (p == 0.6) or (p == 0.8) or (p == 1) ):
            centroids = update_centroids(K, X, pairs, cluster_idx)
                       
        if((cluster_idx[a] == -1) or (cluster_idx[b] == -1)  ): #pair not assigned yet
            cls = best_centroids (a, b, X , centroids,  cluster_idx, pairs, K)
                      
            cluster_idx[a] = cls
            cluster_idx[b] = cls
        assigned += 1
    
    return K, centroids, cluster_idx
    

def merge_mini_clusters(K, X, pairs, centroids, clusters, ortho_threshold):

    # Compute the ortho matrix for all centroids
    ortho_matrix = compute_centroid_Ortho_Matrix(centroids)

    # Track the active clusters
    unique_clusters = np.unique(clusters)
    #print(f" First index clusters :", unique_clusters)
    #print(f"First  nbr of  clusters :", len(unique_clusters))
      
    n_merge=0
    while len(unique_clusters) > 2:
        n_merge +=1
        min_ortho = float('inf')
        merge_pair = None

        for i, cluster_i in enumerate(unique_clusters):
            for j, cluster_j in enumerate(unique_clusters):
                if cluster_i != cluster_j:
                    if ortho_matrix[i, j] < min_ortho:
                        min_ortho = ortho_matrix[i, j]
                        merge_pair = (cluster_i, cluster_j)

        # Stop merging if the minimum ortho exceeds the threshold
        if min_ortho > ortho_threshold:
            break

        # Merge the identified pair
        c1, c2 = merge_pair
        #print(f"current merged clusters :", c1, c2)
                
        # Update cluster assignments
        clusters[clusters == c2] = c1
        clusters = np.where((clusters > c2) & (clusters <= K), clusters - 1, clusters)
    
        # Update centroids and unique clusters
        unique_clusters = np.unique(clusters)
        K = len(unique_clusters)
               
        # Recompute centroid for the merged cluster
        centroids = update_centroids(K, X, pairs, clusters)
        
        # Recompute the ortho matrix for the updated centroids
        ortho_matrix = compute_centroid_Ortho_Matrix(centroids)

    # Update the number of clusters
    final_K = len(unique_clusters)
    print(f" Final K = ", K)
    return final_K, clusters


def run_OrthoClustering(X, theta ): #X is simply X_norm here
    """Runs the OrthogonalityClustering algorithm and computes the final clusters"""
    beta = 0.1 # initial ortho threshold to build mini_centroids
    
    # 1-build data pairs 
    pairs = build_pairs_idx(X)
            
    # 2- compute the orthogonality matrix between different pairs (a,b)
    ortho_matrix = compute_Ortho_Matrix(X, pairs)
    avg_ortho = np.mean(ortho_matrix)
    print(f"\n AVG orthogonality= ", avg_ortho)
    
    # 3-estimate then initialize the number of the centroids
    K, centroids, clusters =  initialize_centroids(X, pairs, ortho_matrix, beta)
  
    # 4- create mini-clusters by assigning the samples to the best clusters
    K, centroids, clusters = create_mini_clusters(X, centroids, clusters, pairs, K)
    
    # 5- merge the mini clusters that are the least ortho while satisfaying an ortho threashold 
    K, merged_clusters =  merge_mini_clusters(K, X, pairs, centroids, clusters, theta)
      
    return K, centroids, merged_clusters 

