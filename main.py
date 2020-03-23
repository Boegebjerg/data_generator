import numpy as np
import json
import matplotlib.pyplot as plt
import os
from GaussianClusters import GaussianClusters




if __name__ == "__main__":

    main_path = "data"

    dataset_n = 10
    cluster_n = 4
    max_variance = 0.3
    cluster_size_factor = 2
    size = 40
    min_size = 10
    max_size = 20
    setting_path = f'{main_path}/n={cluster_n}_size={size}_min={min_size}_max={max_size}_var={max_variance}_size-f={cluster_size_factor}'
    os.makedirs(setting_path,exist_ok=True)

    for i in range(dataset_n):
        print(i)
        path = f'{setting_path}/{i}/'
        os.makedirs(path,exist_ok=True)

        clusters = GaussianClusters(cluster_n, max_variance, cluster_size_factor, size, min_size, max_size)
        clusters.save_data(path)


