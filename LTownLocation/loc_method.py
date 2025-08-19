import torch
import torch.nn as nn

from torch_geometric.data import Data
from torch_geometric.nn import AttentionalAggregation
from torch_geometric.nn import MessagePassing
import os
from torch_geometric.utils import degree

'''
author:ChenLei
2025.8.13
'''


# 用于节点及边属性的特征提取
class FeatureConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        for i in range(10):
            setattr(self, f'conv{i}', torch.nn.Conv1d(in_channels, out_channels, kernel_size=i * 2 + 1,
                                                      padding=i))

        self.dense = torch.nn.Sequential(nn.Linear(out_channels * 10, 64),
                                         nn.ReLU(),
                                         nn.Linear(64, 32),
                                         nn.ReLU(),
                                         nn.Linear(32, out_channels)
                                         )
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(10):
            getattr(self, f'conv{i}').reset_parameters()

    def forward(self, x):
        for i in range(10):
            setattr(self, f'x{i}', getattr(self, f'conv{i}')(x).squeeze(0))

        x = torch.concat([getattr(self, f'x{i}') for i in range(10)])
        x = x.permute(1, 0)
        x = self.dense(x)

        return x


def normalize_adj(edge_index, num_nodes):
    # Add self-loops to the adjacency matrix

    # Calculate the degree of each node
    row, col = edge_index
    deg = degree(col, num_nodes, dtype=torch.float)

    # Calculate D^(-1/2)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    # Normalize adjacency matrix: D^(-1/2) * A * D^(-1/2)
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    return edge_index, edge_weight


class LtownLocLayer(MessagePassing):

    def __init__(self, in_channels, out_channels):
        super().__init__(aggr=None)
        self.node_feature_layer = FeatureConvLayer(in_channels, out_channels)
        self.edge_feature_layer = FeatureConvLayer(in_channels, out_channels)
        self.att_aggr = AttentionalAggregation(torch.nn.Linear(out_channels, 1))
        self.gru = torch.nn.GRUCell(
            input_size=out_channels,  # 输入为聚合消息维度
            hidden_size=out_channels  # 输出与节点特征维度一致
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.node_feature_layer.reset_parameters()
        self.edge_feature_layer.reset_parameters()
        self.att_aggr.reset_parameters()
        self.gru.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        edge_index, edge_weight = normalize_adj(edge_index, num_nodes=x.size(0))
        x = x.unsqueeze(0).permute(0, 2, 1)
        edge_attr = edge_attr.unsqueeze(0).permute(0, 2, 1)

        x = self.node_feature_layer(x)
        edge_attr = self.edge_feature_layer(edge_attr)

        return self.propagate(edge_index, x=x, edge_attr=edge_attr, edge_weight=edge_weight), edge_attr

    def message(self, x_j, edge_attr, edge_weight):
        msg = x_j + edge_attr
        msg = msg * edge_weight.view(-1, 1)
        return msg

    def aggregate(self, inputs, index, x):
        return self.att_aggr(inputs, index, ptr=None, dim_size=x.size(0))

    def update(self, aggr_out, x):
        return self.gru(aggr_out, x)


class DenseLtownLocGcn(nn.Module):

    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(31 * 200, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 32)
        )
        self.anyloc_gcn1 = LtownLocLayer(in_channels=12, out_channels=100)
        self.anyloc_gcn2 = LtownLocLayer(in_channels=112, out_channels=100)

        self.anyloc_gcn3 = LtownLocLayer(in_channels=112, out_channels=200)
        self.anyloc_gcn4 = LtownLocLayer(in_channels=200, out_channels=200)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # dense1
        x1, e1 = self.anyloc_gcn1(x, edge_index, edge_attr)
        x1 = torch.concat([x1, x], axis=-1)
        e1 = torch.concat([e1, edge_attr], axis=-1)

        x2, e2 = self.anyloc_gcn2(x1, edge_index, e1)
        x2 = torch.concat([x2, x], axis=-1)
        e2 = torch.concat([e2, edge_attr], axis=-1)

        # dense2
        x3, e3 = self.anyloc_gcn3(x2, edge_index, e2)
        x3 = torch.concat([x3, x], axis=-1)
        e3 = torch.concat([e3, edge_attr], axis=-1)

        x4, e4 = self.anyloc_gcn4(x3, edge_index, e3)

        x = x4.view(-1, 31 * 200)
        out = self.mlp(x)

        return out


def model_save(model, acc):
    if not os.path.exists(f'./results/{model._get_name()}'):
        os.mkdir(f'./results/{model._get_name()}')
    if max(acc) == acc[-1]:
        torch.save(model, f'./results/{model._get_name()}/{model._get_name()}_best.pth')
        return True


if __name__ == '__main__':

    # 定义节点和边的数量
    num_nodes = 19
    num_edges = 36

    # 随机生成边的索引
    edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)

    # 确保边的索引不会有自环边
    while torch.any(edge_index[0] == edge_index[1]):
        edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)

    # 随机选择4个节点和4条边
    num_features = 4
    node_features = torch.zeros((num_nodes, num_features), dtype=torch.float)
    edge_features = torch.zeros((num_edges, num_features), dtype=torch.float)

    selected_nodes = torch.randperm(num_nodes)[:4]
    selected_edges = torch.randperm(num_edges)[:4]

    # 为选定的节点和边分配随机特征
    node_features[selected_nodes] = torch.rand((4, num_features))
    edge_features[selected_edges] = torch.rand((4, num_features))

    # 创建 PyG 数据对象
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)

    # # Define the layer
    # layer = AnyLocLayer(in_channels=4, out_channels=10)
    # out = layer(data.x, data.edge_index, data.edge_attr)
    # print("Output:", out)
