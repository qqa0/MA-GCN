import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch_geometric.data import Data
from torch_geometric.nn import AttentionalAggregation
from torch_geometric.nn import MessagePassing
import os
from torch_geometric.utils import degree

'''
author:ChenLei
2025.8.13
'''


# Including zero-value features
class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(55 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 34)
        )

    def forward(self, node, edge):
        node = node.view(-1, 19, 4)
        edge = edge.view(-1, 36, 4)
        x = torch.concat([node, edge], dim=-2)

        x = x.view(-1, 55 * 4)
        x = self.linear(x)
        return x


# Excluding features with a value of 0
class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(8 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 34)
        )

    def forward(self, node, edge):
        node = node.view(-1, 19, 4)
        edge = edge.view(-1, 36, 4)

        node1 = torch.empty((len(node), 4, 4)).to('cuda:0')
        edge1 = torch.empty((len(edge), 4, 4)).to('cuda:0')
        for i in range(len(node)):
            node1[i] = node[i][torch.any(node[i] != 0, dim=1)]
            edge1[i] = edge[i][torch.any(edge[i] != 0, dim=1)]

        x = torch.concat([node1, edge1], dim=-2)

        x = x.view(-1, 8 * 4)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)


# Feature extraction for node and edge attributes
class FeatureConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        for i in range(5):
            setattr(self, f'conv{i}', torch.nn.Conv1d(in_channels, out_channels, kernel_size=i * 2 + 1,
                                                      padding=i))

        self.dense = torch.nn.Sequential(nn.Linear(out_channels * 5, 64),
                                         nn.ReLU(),
                                         nn.Linear(64, 32),
                                         nn.ReLU(),
                                         nn.Linear(32, out_channels)
                                         )
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(5):
            getattr(self, f'conv{i}').reset_parameters()

    def forward(self, x):
        for i in range(5):
            setattr(self, f'x{i}', getattr(self, f'conv{i}')(x).squeeze(0))

        x = torch.concat([getattr(self, f'x{i}') for i in range(5)])
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


class AnyLocLayer(MessagePassing):

    def __init__(self, in_channels, out_channels):
        super().__init__(aggr=None)
        # Feature processing
        self.node_feature_layer = FeatureConvLayer(in_channels, out_channels)
        self.edge_feature_layer = FeatureConvLayer(in_channels, out_channels)
        # Information aggregation of the attention mechanism
        self.att_aggr = AttentionalAggregation(torch.nn.Linear(out_channels, 1))
        # Add a GRU update layer
        self.gru = torch.nn.GRUCell(
            input_size=out_channels,  # Input is the aggregated message dimension
            hidden_size=out_channels  # The output has the same dimension as the node feature.
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


class DenseAnyLocGcn(nn.Module):

    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(100 * 19, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 34)
        )
        self.anyloc_gcn1 = AnyLocLayer(in_channels=4, out_channels=100)
        self.anyloc_gcn2 = AnyLocLayer(in_channels=104, out_channels=100)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # dense1
        x1, e1 = self.anyloc_gcn1(x, edge_index, edge_attr)
        x1 = torch.concat([x1, x], axis=-1)
        e1 = torch.concat([e1, edge_attr], axis=-1)

        # dense2
        x2, e2 = self.anyloc_gcn2(x1, edge_index, e1)

        x = x2.view(-1, 19 * 100)
        out = self.mlp(x)

        return out


class ModifiedAlexNet(nn.Module):
    def __init__(self, num_classes=34):
        super(ModifiedAlexNet, self).__init__()
        # Use the pre-trained AlexNet
        self.alexnet = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)

        # Use upsampling to increase the dimension of the input to 224x224
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

        # Modify the first convolutional layer to adapt to the single-channel input
        self.alexnet.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)

        # Modify the last fully connected layer to adapt to the new number of output categories
        self.alexnet.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.upsample(x)
        x = self.alexnet.features(x)
        x = self.alexnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.alexnet.classifier(x)
        return x


class ModifiedVGG11(nn.Module):
    def __init__(self, num_classes=34):
        super(ModifiedVGG11, self).__init__()
        # Use the pre-trained VGG16
        self.vgg16 = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1)

        # Use upsampling to increase the dimension of the input to 224x224
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

        # Modify the first convolutional layer to adapt to the single-channel input
        self.vgg16.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)

        # Modify the last fully connected layer to adapt to the new number of output categories
        self.vgg16.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.upsample(x)
        x = self.vgg16.features(x)
        x = self.vgg16.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.vgg16.classifier(x)
        return x


class ModifiedDenseNet121(nn.Module):
    def __init__(self, num_classes=34):
        super(ModifiedDenseNet121, self).__init__()
        # Use the pre-trained DenseNet121
        self.densenet121 = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

        # 使Use upsampling to increase the dimension of the input to 224x224
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

        # Modify the first convolutional layer to adapt to the single-channel input
        self.densenet121.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Modify the classification layer to accommodate the new number of output categories
        self.densenet121.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.upsample(x)
        x = self.densenet121.features(x)
        x = torch.relu(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.densenet121.classifier(x)
        return x


class ModifyModel:

    def __init__(self, model_name, num_classes):
        self.model_name = model_name
        self.num_classes = num_classes

    def modify_model(self):
        if self.model_name == 'resnet18':
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, self.num_classes)
        elif self.model_name == 'alexnet':
            model = ModifiedAlexNet()

        elif self.model_name == 'vgg11':
            model = ModifiedVGG11()

        elif self.model_name == 'densenet121':
            model = ModifiedDenseNet121()

        else:
            raise ValueError(f"Model {self.model_name} is not supported.")
        return model


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

    gcn = AnyLocGCN()
    out = gcn(data)
