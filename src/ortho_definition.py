#functions
import numpy as np


def atomic_ortho_num (a, b, c, d):
    return  (np.minimum(np.absolute(np.maximum(a,b)-np.maximum(c,d)),np.absolute(np.minimum(a,b)-np.minimum(c,d))))


# a,b,c and d are vectors 
# returns the orthogonality index of pairs (a,b) and (c,d), ortho_index=1 if (a,b) is the inverse of (c,d) on ALL features
def ortho_index_num (a, b, c, d):  
    ortho=0
    for i in range(len(a)):
        ortho += atomic_ortho_num(a[i], b[i], c[i], d[i])
     
    return (ortho/len(a))
    
# compute the ortho matrix : orthogonality of each pair of pairs of instances
def compute_Ortho_Matrix(X, pairs):
    n,_ = np.shape(pairs)
    ortho_matrix = np.zeros((n, n))
    #Compute Orthogonality Matrix
    for i in range(0, n):
        p1 = pairs[i] 
        a =  p1[0] 
        b =  p1[1] 
        for j in range(0,i):
            p2 = pairs[j]
            c =  p2[0] 
            d =  p2[1] 
            ortho = ortho_index_num (X[a], X[b], X[c], X[d])
            ortho_matrix[i,j] = ortho
         
    return ortho_matrix
    
# compute the ortho matrix : orthogonality of each pair of centroids
def compute_centroid_Ortho_Matrix(centroids):
    k = len(centroids)
    ortho_matrix = np.zeros((k, k))

    for i in range(0,k):
        p1 = centroids[i] 
        a =  p1[0] 
        b =  p1[1] 
        for j in range(0,i):
            p2 = centroids[j]
            c =  p2[0] 
            d =  p2[1] 
            ortho = ortho_index_num (a,b,c,d)
            ortho_matrix[i,j] = np.round (ortho, decimals=2)
            ortho_matrix[j,i] = np.round (ortho, decimals=2)
      
    return ortho_matrix


""" Compute the ortho index of each cluster which is the average of ortho index between each pair in this cluster """
def compute_clusters_ortho_index(X, cluster_idx, pairs, K):
    ortho = np.empty(K)
    avg_ortho=0
    
    for i in range(K):
        cluster_pairs = set()
        for c, d in pairs: # c,d are indices 
            if((cluster_idx[c] == i) or (cluster_idx[d] == i)):
                cluster_pairs.add((c,d)) # gather pairs for the cluster i
        
        ortho[i] = compute_onecluster_ortho_index (X, cluster_pairs) 
        #compute the total ortho over all clusters
        avg_ortho += ortho[i]

    return (ortho, avg_ortho/K) 