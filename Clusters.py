import numpy as np

class Clusters():

    def dist_matrix_to_sim(self,dist_matrix):
        max_dist = np.max(dist_matrix)
        return (dist_matrix-max_dist)*-1

    def normalize_matrix(self,matrix):
        min_val = np.min(matrix)
        max_val = np.max(matrix)
        return (matrix-min_val)/(max_val-min_val)
