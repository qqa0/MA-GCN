import os
from typing import Union, List, Tuple
import mat4py
import numpy as np
import pandas as pd
import wntr
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
import torch

'''
author:ChenLei
2025.8.13
'''


class GcnDataMakeFlowDistribution:

    def __init__(self, path, all_nodes, all_pipes, pressure_node, flow_pipe):
        self.path = path  # /partX/txx/errorx
        self.pressure_node = pressure_node
        self.flow_pipe = flow_pipe
        self.all_nodes = pd.read_excel(all_nodes, header=None).values.flatten()
        self.all_pipes = pd.read_excel(all_pipes, header=None).values.flatten()
        self.datalist = []
        self.process()

    def _topology(self):
        self.wn = wntr.network.WaterNetworkModel(r'xxxxL-TOWN.inp')  # path of inp
        G = self.wn.to_graph()
        G = G.to_directed()

        nodes = self.all_nodes.copy()
        self.node_mapping = {name: i for i, name in enumerate(sorted(list(nodes)))}
        _edges_coo = [(u, v) for u, v in G.edges() if u in nodes and v in nodes]
        edges_coo = [(self.node_mapping[u], self.node_mapping[v]) for u, v in sorted(_edges_coo)]
        self.edges_coo = np.array(edges_coo).T
        self.edges_index = torch.tensor(self.edges_coo, dtype=torch.long)

    def _node_pipe_hash_conversion(self):
        # node
        self.node_hash = []
        for n in self.pressure_node:
            self.node_hash.append(self.node_mapping[n])

        # pipe
        pipe_hash_start, pipe_hash_end = [], []
        for p in self.flow_pipe:
            link = self.wn.get_link(p)
            s = link.start_node.name
            e = link.end_node.name
            pipe_hash_start.append(self.node_mapping[s])
            pipe_hash_end.append(self.node_mapping[e])

        pipe_hash_ = np.array([pipe_hash_start, pipe_hash_end])
        self.pipe_hash = []
        for col in range(self.edges_coo.shape[1]):
            for pipe in range(pipe_hash_.shape[1]):
                if np.array_equal(self.edges_coo[:, col], pipe_hash_[:, pipe]):
                    self.pipe_hash.append(col)

    def _data_make(self):
        dirs = os.listdir(self.path)
        self.labels = {name: i for i, name in enumerate(sorted(dirs))}

        for dir_ in tqdm(dirs):
            files = os.listdir(os.path.join(self.path, dir_))
            label = self.labels[dir_]
            y = torch.tensor([label], dtype=torch.int)
            for file_ in files:
                df = pd.read_csv(os.path.join(self.path, dir_, file_))
                data = df.iloc[:, 1:11].values
                ################################################################ If you need to set specific data for flow allocation
                flow_ = df.iloc[2, 13]
                if flow_ >= 0 and flow_ < 20:
                    #################################################################
                    for i in range(12):
                        node_attr = np.zeros((len(self.all_nodes), 12))
                        edge_attr = np.zeros((len(self.all_pipes), 12))

                        pressure_ = data[i:i + 12, :5]
                        flow_ = data[i:i + 12, 5:]

                        for pre, node in enumerate(self.node_hash):
                            node_attr[node] = pressure_[:, pre]

                        for flo, pipe in enumerate(self.pipe_hash):
                            edge_attr[pipe] = flow_[:, flo]

                        x = torch.tensor(node_attr, dtype=torch.float)
                        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

                        data_ = Data(edge_index=self.edges_index, edge_attr=edge_attr, x=x, y=y)
                        self.datalist.append(data_)

    def process(self):
        self._topology()
        self._node_pipe_hash_conversion()
        self._data_make()


class GcnDataMake:

    def __init__(self, path, all_nodes, all_pipes, pressure_node, flow_pipe):
        self.path = path  # /partX/txx/errorx
        self.pressure_node = pressure_node
        self.flow_pipe = flow_pipe
        self.all_nodes = pd.read_excel(all_nodes, header=None).values.flatten()
        self.all_pipes = pd.read_excel(all_pipes, header=None).values.flatten()
        self.datalist = []
        self.process()

    def _topology(self):
        self.wn = wntr.network.WaterNetworkModel(r'xxx\L-TOWN.inp')  # path of inp
        G = self.wn.to_graph()
        G = G.to_directed()

        nodes = self.all_nodes.copy()
        self.node_mapping = {name: i for i, name in enumerate(sorted(list(nodes)))}
        _edges_coo = [(u, v) for u, v in G.edges() if u in nodes and v in nodes]
        edges_coo = [(self.node_mapping[u], self.node_mapping[v]) for u, v in sorted(_edges_coo)]
        self.edges_coo = np.array(edges_coo).T
        self.edges_index = torch.tensor(self.edges_coo, dtype=torch.long)

    def _node_pipe_hash_conversion(self):
        # node
        self.node_hash = []
        for n in self.pressure_node:
            self.node_hash.append(self.node_mapping[n])

        # pipe
        pipe_hash_start, pipe_hash_end = [], []
        for p in self.flow_pipe:
            link = self.wn.get_link(p)
            s = link.start_node.name
            e = link.end_node.name
            pipe_hash_start.append(self.node_mapping[s])
            pipe_hash_end.append(self.node_mapping[e])

        pipe_hash_ = np.array([pipe_hash_start, pipe_hash_end])
        self.pipe_hash = []
        for col in range(self.edges_coo.shape[1]):
            for pipe in range(pipe_hash_.shape[1]):
                if np.array_equal(self.edges_coo[:, col], pipe_hash_[:, pipe]):
                    self.pipe_hash.append(col)

    def _data_make(self):
        dirs = os.listdir(self.path)
        self.labels = {name: i for i, name in enumerate(sorted(dirs))}

        for dir_ in tqdm(dirs):
            files = os.listdir(os.path.join(self.path, dir_))
            label = self.labels[dir_]
            y = torch.tensor([label], dtype=torch.int)
            for file_ in files:
                df = pd.read_csv(os.path.join(self.path, dir_, file_))
                data = df.iloc[:, 1:11].values
                for i in range(12):
                    node_attr = np.zeros((len(self.all_nodes), 12))
                    edge_attr = np.zeros((len(self.all_pipes), 12))

                    pressure_ = data[i:i + 12, :5]
                    flow_ = data[i:i + 12, 5:]

                    for pre, node in enumerate(self.node_hash):
                        node_attr[node] = pressure_[:, pre]

                    for flo, pipe in enumerate(self.pipe_hash):
                        edge_attr[pipe] = flow_[:, flo]

                    x = torch.tensor(node_attr, dtype=torch.float)
                    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

                    data_ = Data(edge_index=self.edges_index, edge_attr=edge_attr, x=x, y=y)
                    self.datalist.append(data_)

    def process(self):
        self._topology()
        self._node_pipe_hash_conversion()
        self._data_make()


class LtownBurstDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.pre_transform = pre_transform
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        return ['data.pt']

    def process(self):
        data = GcnDataMake(path=r'D:\OneDrive\CQU\会议及文章\01 博士论文\Location\DataSim\LTown\partB\t20\error_32',
                           # Generated simulated data path
                           all_nodes=r'xx\partB\nodes_B.xlsx',  # Nodes in the X zone
                           all_pipes=r'xxx\partB\pipes_B.xlsx',  # Pipes in the X zone
                           # part B
                           pressure_node=['n207', 'n215', 'n227', 'n253', 'n337'],
                           flow_pipe=['p74', 'p195', 'p236', 'p674', 'p679'])
        #
        # #part A
        # pressure_node=['n1', 'n7', 'n15', 'n41', 'n363'],
        # flow_pipe=['p256', 'p11', 'p263', 'p249', 'p295'])
        data_list = data.datalist

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # data, slices = self.collate(data_list)
        self.save(data_list, self.processed_paths[0])

        return data


##Used for statistically analyzing the burst pipe flow distribution in the simulation data
class FlowDistribution:
    def __init__(self, path):
        self.path = path  # /partX/txx/errorx
        self.cal()

    def cal(self):
        dirs = os.listdir(self.path)

        self.flow = []
        for dir_ in dirs:
            files = os.listdir(os.path.join(self.path, dir_))
            for f in files:
                df = pd.read_csv(os.path.join(self.path, dir_, f))
                flow_ = df.iloc[2, 13]
                self.flow.append(flow_)


if __name__ == '__main__':
    # gcndata = GcnDataMake(path=r'D:\OneDrive\CQU\会议及文章\01 博士论文\Location\DataSim\LTown\partA\t200\error0',
    #                       all_nodes = r'D:\OneDrive\CQU\会议及文章\01 博士论文\Location\DataSim\LTown\partA\nodes_A.xlsx',
    #                       all_pipes=r'D:\OneDrive\CQU\会议及文章\01 博士论文\Location\DataSim\LTown\partA\pipes_A.xlsx',
    #                       pressure_node=['n1', 'n7', 'n15', 'n41', 'n363'],
    #                       flow_pipe=['p256', 'p11', 'p263', 'p249', 'p295'])

    # dataset = LtownBurstDataset(root=r'./partA/t20/g_error_0')

    # flowdistribution = FlowDistribution(path='./partA/t20/error0')
    dataset = LtownBurstDataset(root=r'./partB/t200/g_error_0')
