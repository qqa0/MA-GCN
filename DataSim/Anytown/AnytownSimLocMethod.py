import torch
from torch_geometric.transforms import BaseTransform
import pandas as pd
import os
from tqdm import tqdm
import numpy as np

'''
author:ChenLei
2025.8.13
'''


# Used for finding the maximum and minimum values of pressure and flow measurements
class FindMaxMin:

    def __init__(self):
        self.excel_path = './anytown_data/t500/loc/' #use your path
        self.files = os.listdir(self.excel_path)

    def _find(self):
        maxx = np.zeros(8)
        minn = np.array([999] * 8)

        for file in tqdm(self.files, desc='Files ', position=0, leave=False, ncols=120, colour='red'):
            # for file in [self.files[0]]:
            excel_ = pd.ExcelFile(os.path.join(self.excel_path, file))
            sheets = excel_.sheet_names
            for sheet in tqdm(sheets, colour='white', desc='Sheet ', position=1, leave=False, ncols=120):
                df = pd.read_excel(os.path.join(self.excel_path, file), sheet_name=sheet)
                data = df.values[:, 1:9].astype('float32')
                maxx_ = data.max(axis=0)
                minn_ = data.min(axis=0)

                maxx[maxx < maxx_] = maxx_[maxx < maxx_]
                minn[minn > minn_] = minn_[minn > minn_]
            print('\n')
            print(f'Max: ', maxx, '\n')
            print(f'Min: ', minn)

        return maxx, minn


class AnytownNormalizeScale(BaseTransform):

    def __init__(self):
        pass

    def forward(self, data):
        if hasattr(data, 'x'):
            node_hash = [9, 1, 5, 7]
            x_max = np.array([28.23374176, 34.38719177, 27.34179306, 24.1998024], dtype='float32')
            x_min = np.array([1.0, 0.0, 1.0, 0.0], dtype='float32')
            x_max = torch.from_numpy(x_max)
            x_min = torch.from_numpy(x_min)

            for i, hash_ in enumerate(node_hash):
                data.x[hash_] = (data.x[hash_] - x_min[i]) / (x_max[i] - x_min[i])

        if hasattr(data, 'edge_attr'):
            pipe_hash = [5, 17, 22, 23]
            e_max = np.array([806.95690918, 816.38397217, 48.12962723, 81.2507019], dtype='float32')
            e_min = np.array([-97.0, -91.0, -37.0, -49.0], dtype='float32')
            e_max = torch.from_numpy(e_max)
            e_min = torch.from_numpy(e_min)
            data.edge_attr = (data.edge_attr - e_min) / (e_max - e_min)

            for i, hash_ in enumerate(pipe_hash):
                data.edge_attr[hash_] = (data.edge_attr[hash_] - e_min[i]) / (e_max[i] - e_min[i])

        return data


if __name__ == '__main__':
    find_ = FindMaxMin()
    maxx, minn = find_._find()
