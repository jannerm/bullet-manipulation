import os

import d4rl
import gdown
import numpy as np

if __name__ == "__main__":
    url = 'https://drive.google.com/drive/folders/1jxBQE1adsFT1sWsfatbhiZG6Zkf3EW0Q'

    output = os.path.join(d4rl.offline_env.DATASET_PATH, 'cog')
    gdown.download_folder(url, output=output)
    # dataset = np.load('/home/kostrikov/.d4rl/datasets/cog/blocked_drawer_1_prior.npy', allow_pickle=True)
