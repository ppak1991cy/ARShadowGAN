"""
    ARShadowGAN
    func  : ShadowAR数据集
    Author: Chen Yu
    Date  : 2020.10.20
"""
import torch
from torch.utils.data import Dataset

from utils import match_samples_path, load_sample


class ShadowARDataset(Dataset):

    def __init__(self):
        self.samples = match_samples_path()

    def __getitem__(self, idx):
        sample = self.samples[idx]
        data = load_sample(sample)

        vs = torch.tensor(data['vs'], dtype=torch.float32).transpose(2, 0)
        no_vs = torch.tensor(data['no_vs'], dtype=torch.float32).transpose(2, 0)
        rs_mask = torch.tensor(data['rs_mask'], dtype=torch.float32).unsqueeze(0)
        ro_mask = torch.tensor(data['ro_mask'], dtype=torch.float32).unsqueeze(0)
        vo_mask = torch.tensor(data['vo_mask'], dtype=torch.float32).unsqueeze(0)
        return vs, no_vs, rs_mask, ro_mask, vo_mask

    def __len__(self):
        return len(self.samples)


if __name__ == '__main__':
    dataset = ShadowARDataset()
    it = iter(dataset)
    sample = next(it)
    print(sample[0].shape)
    print(sample[1].shape)
    print(sample[2].shape)
    print(sample[3].shape)
    print(sample[4].shape)
