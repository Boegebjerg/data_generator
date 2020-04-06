import numpy as np
import json
import os
from Clusters import Clusters


# Use deep learning to generate dataset that maximizes difference in f measure between two clustering algorithms

class GaussianSimilarityClusters(Clusters):

    def __init__(self,cluster_n, intra_max_variance, inter_max_variance, size, min_size, max_size, cluster_size_factor,inter_scale):
        self.cluster_size_factor = cluster_size_factor
        self.max_size = max_size
        self.min_size = min_size
        self.size = size
        self.intra_max_variance = intra_max_variance
        self.inter_max_variance = inter_max_variance
        self.cluster_n = cluster_n
        self.inter_scale = inter_scale
        self.settings = self.initialize_settings()
        self.dist_matrix = self.create_clusters(self.settings)
        self.sim_matrix = self.dist_matrix_to_sim(self.dist_matrix)
        self.norm_sim_matrix = self.normalize_matrix(self.sim_matrix)
        self.norm_dist_matrix = self.normalize_matrix(self.dist_matrix)
        self.sim_matrix_melt = self.matrix_melt(self.sim_matrix)
        self.norm_sim_matrix_melt = self.matrix_melt(self.norm_sim_matrix)
        self.dist_matrix_melt = self.matrix_melt(self.dist_matrix)
        self.norm_dist_matrix_melt = self.matrix_melt(self.norm_dist_matrix)




    def initialize_settings(self):
        cluster_settings = []
        cum_points = 0
        cum_mean = 0
        cum_stddev = 0
        for i in range(self.cluster_n):
            cluster_settings.append({})
            cluster_settings[i]['mean'] = np.random.uniform(0, 1, 1)[0]
            cluster_settings[i]['stddev'] = np.random.uniform(0, self.intra_max_variance, 1)[0]
            cum_mean += cluster_settings[i]['mean']
            cum_stddev += cluster_settings[i]['stddev']

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
        settings = {"data": cluster_settings,
                    'n_points': cum_points,
                    'inter_mean': cum_mean/float(self.cluster_n)*2,
                    'inter_stddev': self.inter_max_variance}
        return settings


    def create_clusters(self,cluster_settings):
        n_points = cluster_settings['n_points']
        current_cluster_idx = 0
        dists = []
        for i,cluster_setting in enumerate(cluster_settings['data']):
            cluster_size = cluster_setting['size']
            for j in range(cluster_size):
                before = np.random.normal(cluster_settings['inter_mean'], cluster_settings['inter_stddev'], current_cluster_idx)
                current = np.random.normal(cluster_setting['mean'], cluster_setting['stddev'], cluster_size)
                after = np.random.normal(cluster_settings['inter_mean'], cluster_settings['inter_stddev'], n_points-current_cluster_idx-cluster_size)
                total = np.concatenate((before,current,after))
                dists.append(total)


            current_cluster_idx += cluster_size

        return np.array(dists)


    def save_data(self,path):
        self.save_matrix(self.sim_matrix, 'sim_matrix.txt', path)
        self.save_matrix(self.dist_matrix, 'dist_matrix.txt', path)
        self.save_matrix(self.norm_sim_matrix, 'norm_sim_matrix.txt', path)
        self.save_matrix(self.norm_dist_matrix, 'norm_dist_matrix.txt', path)
        self.save_melt(self.sim_matrix_melt, 'sim_matrix_melt.txt', path)
        self.save_melt(self.dist_matrix_melt, 'dist_matrix_melt.txt', path)
        self.save_melt(self.norm_sim_matrix_melt, 'norm_sim_matrix_melt.txt', path)
        self.save_melt(self.norm_dist_matrix_melt, 'norm_dist_matrix_melt.txt', path)
        self.save_settings(self.settings, 'settings.json', path)
