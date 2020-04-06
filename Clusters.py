import numpy as np
import matplotlib.pyplot as plt
import json

class Clusters():

    def dist_matrix_to_sim(self,dist_matrix):
        max_dist = np.max(dist_matrix)
        return (dist_matrix-max_dist)*-1

    def normalize_matrix(self,matrix):
        min_val = np.min(matrix)
        max_val = np.max(matrix)
        return (matrix-min_val)/(max_val-min_val)

    def matrix_melt(self, matrix):
        length = matrix.shape[0]
        melt = []
        for i in range(length):
            for j in range(length):
                melt.append([i,j,matrix[i,j]])
        return melt

    def save_matrix(self, matrix, name, path):
        np.savetxt(path + name, matrix, delimiter='\t', fmt='%f')

    def save_graph(self, name, path):
        plt.savefig(path + name)
        plt.clf()

    def save_settings(self, settings, name, path):
        with open(path + name, 'w') as f:
            f.write(json.dumps(settings))

    def save_melt(self, melt, name, path):
        output = '\n'.join(['\t'.join([str(b) for b in a]) for a in melt])
        with open(path + name, 'w') as f:
            f.write(output)

