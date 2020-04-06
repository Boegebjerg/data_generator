import numpy as np
import json
import os
from Clusters import Clusters
import matplotlib.pyplot as plt

class GaussianClusters(Clusters):

    def __init__(self,cluster_n, max_variance, cluster_size_factor, size, min_size, max_size):
        self.max_size = max_size
        self.min_size = min_size
        self.size = size
        self.cluster_size_factor = cluster_size_factor
        self.max_variance = max_variance
        self.cluster_n = cluster_n

        self.settings = self.initialize_settings()
        self.points = self.create_clusters(self.settings)
        for i, ps in enumerate(self.points):
            self.settings[i]['points'] = ps

        self.plot_clusters(self.points)
        self.dist_matrix = self.points_to_dist_matrix(self.points)
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
        for i in range(self.cluster_n):
            cluster_settings.append({})
            cluster_settings[i]['mean_x'] = np.random.uniform(0, 1, 1)[0]
            cluster_settings[i]['mean_y'] = np.random.uniform(0, 1, 1)[0]
            cluster_settings[i]['stddev_x'] = np.random.uniform(0, self.max_variance, 1)[0]
            cluster_settings[i]['stddev_y'] = np.random.uniform(0, self.max_variance, 1)[0]

            if i == self.cluster_n-1:
                cluster_size = self.size-cum_points
            else:
                max_cluster_size = int((self.size-cum_points)/self.cluster_n*self.cluster_size_factor)
                cluster_size = int(np.random.randint(0,max_cluster_size,1)[0])
            if cluster_size<self.min_size:
                cluster_size = self.min_size
            if cluster_size>self.max_size:
                cluster_size = self.max_size
            cum_points += cluster_size

            cluster_settings[i]['size'] = cluster_size

        return cluster_settings


    def create_clusters(self,cluster_settings):
        cluster_points = []
        for i,cluster_setting in enumerate(cluster_settings):
            x = np.random.normal(cluster_setting['mean_x'], cluster_setting['stddev_x'],cluster_setting['size'])
            y = np.random.normal(cluster_setting['mean_y'], cluster_setting['stddev_y'],cluster_setting['size'])
            cluster_points.append(list(zip(x,y)))
        return cluster_points

    def plot_clusters(self,cluster_points,show=False):
        colors = ['blue','red','brown','green']
        for i,cluster_points in enumerate(self.points):
            plt.plot([a[0] for a in cluster_points],[a[1] for a in cluster_points],'ro',color = colors[i])
        if show:
            plt.show()


    def points_to_dist_matrix(self,points):
        def dist(a,b):
            return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

        flat_points = [b for a in points for b in a]
        dist_matrix = np.zeros((len(flat_points),len(flat_points)))
        for i in range(len(flat_points)):
            for j in range(i,len(flat_points)):
                dist_matrix[i,j] = dist_matrix[j,i] = dist(flat_points[i],flat_points[j])

        return dist_matrix



    def save_data(self,path):
        self.save_matrix(self.sim_matrix, 'sim_matrix.txt', path)
        self.save_matrix(self.dist_matrix, 'dist_matrix.txt', path)
        self.save_matrix(self.norm_sim_matrix, 'norm_sim_matrix.txt', path)
        self.save_matrix(self.norm_dist_matrix, 'norm_dist_matrix.txt', path)
        self.save_melt(self.sim_matrix_melt, 'sim_matrix_melt.txt', path)
        self.save_melt(self.dist_matrix_melt, 'dist_matrix_melt.txt', path)
        self.save_melt(self.norm_sim_matrix_melt, 'norm_sim_matrix_melt.txt', path)
        self.save_melt(self.norm_dist_matrix_melt, 'norm_dist_matrix_melt.txt', path)
        self.save_graph('graph.png', path)
        self.save_settings(self.settings, 'settings.json', path)
