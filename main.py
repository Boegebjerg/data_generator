import numpy as np
import json
import matplotlib.pyplot as plt
import os
from GaussianClusters import GaussianClusters
from GaussianSimilarityClusters import GaussianSimilarityClusters




if __name__ == "__main__":

    main_path = "data"

    dataset_n = 1000
    cluster_n = 2
    max_variance = 0.3
    cluster_size_factor = 2
    size = 40
    min_size = 10
    max_size = 30
    inter_scale = 2
    inter_max_variance = 0.3
    setting_path = f'{main_path}/n={cluster_n}_size={size}_min={min_size}_max={max_size}_var={max_variance}_size-f={cluster_size_factor}_inter_scale={inter_scale}_inter-variance={inter_max_variance}'
    os.makedirs(setting_path,exist_ok=True)

    for i in range(dataset_n):
        print(i)
        path = f'{setting_path}/{i}/'
        os.makedirs(path,exist_ok=True)
        clusters = GaussianSimilarityClusters(cluster_n, max_variance, inter_max_variance, size, min_size, max_size, cluster_size_factor,inter_scale)
        #clusters = GaussianClusters(cluster_n, max_variance, cluster_size_factor, size, min_size, max_size)
        clusters.save_data(path)


