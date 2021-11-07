import os

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class RobovserseDataset(Dataset):
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir

        with open(os.path.join(dataset_dir, 'lengths.np'), 'rb') as f:
            self.lengths = np.load(f)

        self.length_sum = np.concatenate([[0], np.cumsum(self.lengths)])

        self.size = np.sum(self.lengths)

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):
        traj_idx = np.searchsorted(self.length_sum, idx, side='right') - 1
        step_idx = idx - self.length_sum[traj_idx]

        with open(os.path.join(self.dataset_dir, str(traj_idx), 'data.npz'),
                  'rb') as f:
            data = dict(np.load(f))
            action = data['actions'][step_idx]
            reward = data['rewards'][step_idx]
            mask = data['masks'][step_idx]

        obs = Image.open(
            os.path.join(self.dataset_dir, str(traj_idx), f'{step_idx}.png'))
        next_obs = Image.open(
            os.path.join(self.dataset_dir, str(traj_idx), f'{step_idx+1}.png'))

        obs = np.asarray(obs)
        next_obs = np.asarray(next_obs)

        return obs, action, reward, mask, next_obs


if __name__ == "__main__":

    ds = RobovserseDataset(dataset_dir='dataset')

    dataloader = DataLoader(ds,
                            batch_size=256,
                            shuffle=True,
                            num_workers=4,
                            collate_fn=numpy_collate,
                            pin_memory=False)

    for batch in tqdm(dataloader):
        pass
