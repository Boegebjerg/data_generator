import numpy as np
import json
import os
from Clusters import Clusters


# Use deep learning to generate dataset that maximizes difference in f measure between two clustering algorithms

class GaussianSimilarityClusters(Clusters):

    def __init__(self,cluster_n, intra_max_variance, inter_max_variance, size, min_size, max_size, cluster_size_factor):
        self.cluster_size_factor = cluster_size_factor
        self.max_size = max_size
        self.min_size = min_size
        self.size = size
        self.inter_max_variance = inter_max_variance
        self.intra_max_variance = intra_max_variance
        self.cluster_n = cluster_n


    def initialize_settings(self):
        cluster_settings = []
        cum_points = 0
        for i in range(self.cluster_n):
            cluster_settings.append({})
            cluster_settings[i]['mean'] = np.random.uniform(1, 2, 1)[0]
            cluster_settings[i]['stddev'] = np.random.uniform(0, self.intra_max_variance, 1)[0]

            if i == self.cluster_n - 1:
                cluster_size = self.size - cum_points
            else:
                max_cluster_size = int((self.size - cum_points) / self.cluster_n * self.cluster_size_factor)
                cluster_size = int(np.random.randint(0, max_cluster_size, 1)[0])
            if cluster_size < self.min_size:
                cluster_size = self.min_size
            if cluster_size > self.max_size:
                cluster_size = self.max_size
            cum_points += cluster_size

            cluster_settings[i]['size'] = cluster_size

        return cluster_settings
