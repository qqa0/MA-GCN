import numpy as np
import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import yaml
from Location.DataSim.Anytown.AnytownSimLoc2024 import AnytownBurstDataset, AnytownNormalizeScale
from loc_method import *

'''
author:ChenLei
2025.8.13
'''


# MA-GCN model Train
class GNNTrain:

    def __init__(self, model_path=None):
        self.model_path = model_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.process()

    def train(self):
        dataset = AnytownBurstDataset('../DataSim/Anytown/anytown_data/t500/gcndata',  # path of gcndata
                                      pre_transform=AnytownNormalizeScale())
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=val_size, shuffle=True)

        if self.model_path:
            model = torch.load(self.model_path, map_location=self.device)
        else:
            model = DenseAnyLocGcn().to(self.device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.95)
        train_loss, valid_loss = [], []
        train_epochs_loss, valid_epochs_loss = [], []
        validation_acc = []

        best_epoch = []
        for epoch in range(1000):
            model.train()
            train_epoch_loss = []

            # =========================train=======================
            for idx, tra in enumerate(train_dataloader):
                tra_y = tra.y.to(torch.long).to(self.device)

                # GNN
                tra = tra.to(self.device)
                pre_y = model(tra)

                loss = criterion(pre_y, tra_y)

                optimizer.zero_grad()

                loss.backward()
                optimizer.step()
                train_epoch_loss.append(loss.item())
                train_loss.append(loss.item())

                prediction = torch.max(F.softmax(pre_y), 1)[1]
                pre_y = prediction.data.to('cpu').numpy().squeeze()
                tra_y = tra_y.to('cpu').numpy().squeeze()

                accuracy = sum(pre_y == tra_y) / len(tra_y)
                if idx % (len(train_dataloader) // 1) == 0:
                    print('--' * 40)
                    print(
                        f'epoch={epoch}/{200},{idx}/{len(train_dataloader)} of train, loss={loss.item()}, acc={accuracy * 100:.2f}%')
            train_epochs_loss.append(np.average(train_epoch_loss))
            scheduler.step()

            # =====================valid============================
            model.eval()
            valid_epoch_loss = []
            for idx, val in enumerate(val_dataloader):

                val_y = val.y.to(torch.long).to(self.device)

                # GNN
                val = val.to(self.device)
                pre_y = model(val)
                pre_y = pre_y.squeeze(dim=-1)

                loss = criterion(pre_y, val_y)
                valid_epoch_loss.append(loss.item())
                valid_loss.append(loss.item())
                prediction = torch.max(F.softmax(pre_y), 1)[1]
                pre_y = prediction.data.to('cpu').numpy().squeeze()
                val_y = val_y.to('cpu').numpy().squeeze()

                val_acc = sum(pre_y == val_y) / val_size
                validation_acc.append(val_acc)
                best_signal = model_save(model, validation_acc)
                if best_signal:
                    best_epoch.append(epoch)

                print('--' * 40)
                print(f'validation acc is {val_acc * 100:.2f}%')

        with open(f'./results/{model._get_name()}/{model._get_name()}_best.txt', 'w') as f:
            f.write('best_epoch:    ' + str(best_epoch[-1]))
            f.write('\n')
            f.write('val_acc:    ' + str(validation_acc[best_epoch[-1]]))
            f.write('\n')
            f.write('tra_loss:    ' + str(train_epochs_loss[best_epoch[-1]]))
            f.write('\n')
            f.write('val_loss:    ' + str(valid_loss[best_epoch[-1]]))
            f.close()

    def process(self):
        self.train()


# MLP model Train
class MlpTrain:

    def __init__(self, model_path=None):
        self.model_path = model_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.process()

    def train(self):
        dataset = AnytownBurstDataset('../DataSim/Anytown/anytown_data/t500/gcndata',
                                      pre_transform=AnytownNormalizeScale())

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=val_size, shuffle=True)

        if self.model_path:
            model = torch.load(self.model_path, map_location=self.device)
        else:
            model = MLP().to(self.device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        train_loss, valid_loss = [], []
        train_epochs_loss, valid_epochs_loss = [], []
        validation_acc = []

        best_epoch = []
        for epoch in range(1000):
            model.train()
            train_epoch_loss = []

            # =========================train=======================
            for idx, tra in enumerate(train_dataloader):
                node_tra = tra.x.to(torch.float32).to(self.device)
                edge_tra = tra.edge_attr.to(torch.float32).to(self.device)
                tra_y = tra.y.to(torch.long).to(self.device)

                pre_y = model(node_tra, edge_tra)
                loss = criterion(pre_y, tra_y)

                optimizer.zero_grad()

                loss.backward()
                optimizer.step()
                train_epoch_loss.append(loss.item())
                train_loss.append(loss.item())

                prediction = torch.max(F.softmax(pre_y), 1)[1]
                pre_y = prediction.data.to('cpu').numpy().squeeze()
                tra_y = tra_y.to('cpu').numpy().squeeze()

                accuracy = sum(pre_y == tra_y) / len(tra_y)
                if idx % (len(train_dataloader) // 1) == 0:
                    print('--' * 40)
                    print(
                        f'epoch={epoch}/{50},{idx}/{len(train_dataloader)} of train, loss={loss.item()}, acc={accuracy * 100:.2f}%')
            train_epochs_loss.append(np.average(train_epoch_loss))

            # =====================valid============================
            model.eval()
            valid_epoch_loss = []
            for idx, val in enumerate(val_dataloader):
                node_val = val.x.to(torch.float32).to(self.device)
                edge_val = val.edge_attr.to(torch.float32).to(self.device)
                val_y = val.y.to(torch.long).to(self.device)

                pre_y = model(node_val, edge_val)
                pre_y = pre_y.squeeze(dim=-1)

                loss = criterion(pre_y, val_y)
                valid_epoch_loss.append(loss.item())
                valid_loss.append(loss.item())
                prediction = torch.max(F.softmax(pre_y), 1)[1]
                pre_y = prediction.data.to('cpu').numpy().squeeze()
                val_y = val_y.to('cpu').numpy().squeeze()

                val_acc = sum(pre_y == val_y) / val_size
                validation_acc.append(val_acc)
                best_signal = model_save(model, validation_acc)
                if best_signal:
                    best_epoch.append(epoch)

                print('--' * 40)
                print(f'validation acc is {val_acc * 100:.2f}%')

        if not os.path.exists(f'./results/{model._get_name()}'):
            os.mkdir(f'./results/{model._get_name()}')
        torch.save(model, f'./results/{model._get_name()}/{model._get_name()}_best.pth')

        with open(f'./results/{model._get_name()}/{model._get_name()}_best.txt', 'w') as f:
            f.write('best_epoch:    ' + str(best_epoch[-1]))
            f.write('\n')
            f.write('val_acc:    ' + str(validation_acc[-1]))
            f.write('\n')
            f.write('tra_loss:    ' + str(train_epochs_loss[-1]))
            f.write('\n')
            f.write('val_loss:    ' + str(valid_loss[-1]))
            f.close()

    def process(self):
        self.train()


# Use the models provided by PyTorch for training and prediction.
class TorchTrain:

    def __init__(self, select_model):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = ModifyModel(select_model, 34)
        self.process()

    def train(self):
        dataset = AnytownBurstDataset('../DataSim/Anytown/anytown_data/t500/gcndata',
                                      pre_transform=AnytownNormalizeScale())

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=val_size, shuffle=True)

        model = self.model.modify_model().to(self.device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        train_loss, valid_loss = [], []
        train_epochs_loss, valid_epochs_loss = [], []
        validation_acc = []
        best_epoch = []

        for epoch in range(1000):
            model.train()
            train_epoch_loss = []

            # =========================train=======================
            for idx, tra in enumerate(train_dataloader):
                node_tra = tra.x.to(torch.float32).to(self.device)
                edge_tra = tra.edge_attr.to(torch.float32).to(self.device)
                node = node_tra.view(-1, 19, 4)
                edge = edge_tra.view(-1, 36, 4)
                inp = torch.concat([node, edge], dim=-2)
                inp = inp.unsqueeze(1)
                tra_y = tra.y.to(torch.long).to(self.device)

                pre_y = model(inp)
                loss = criterion(pre_y, tra_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_epoch_loss.append(loss.item())
                train_loss.append(loss.item())

                prediction = torch.max(F.softmax(pre_y), 1)[1]
                pre_y = prediction.data.to('cpu').numpy().squeeze()
                tra_y = tra_y.to('cpu').numpy().squeeze()

                accuracy = sum(pre_y == tra_y) / len(tra_y)
                if idx % (len(train_dataloader) // 2) == 0:
                    print('--' * 40)
                    print(
                        f'epoch={epoch}/{50},{idx}/{len(train_dataloader)} of train, loss={loss.item()}, acc={accuracy * 100:.2f}%')
            train_epochs_loss.append(np.average(train_epoch_loss))

            # =====================valid============================
            model.eval()
            valid_epoch_loss = []
            for idx, val in enumerate(val_dataloader):
                node_val = val.x.to(torch.float32).to(self.device)
                edge_val = val.edge_attr.to(torch.float32).to(self.device)
                node_v = node_val.view(-1, 19, 4)
                edge_v = edge_val.view(-1, 36, 4)
                inp_v = torch.concat([node_v, edge_v], dim=-2)
                inp_v = inp_v.unsqueeze(1)
                val_y = val.y.to(torch.long).to(self.device)

                pre_y = model(inp_v)
                pre_y = pre_y.squeeze(dim=-1)

                loss = criterion(pre_y, val_y)
                valid_epoch_loss.append(loss.item())
                valid_loss.append(loss.item())
                prediction = torch.max(F.softmax(pre_y), 1)[1]
                pre_y = prediction.data.to('cpu').numpy().squeeze()
                val_y = val_y.to('cpu').numpy().squeeze()

                val_acc = sum(pre_y == val_y) / val_size
                validation_acc.append(val_acc)

                best_signal = model_save(model, validation_acc)
                if best_signal:
                    best_epoch.append(epoch)

                print('--' * 40)
                print(f'validation acc is {val_acc * 100:.2f}%')
            if not os.path.exists(f'./results/{model._get_name()}'):
                os.mkdir(f'./results/{model._get_name()}')
            torch.save(model, f'./results/{model._get_name()}/{model._get_name()}_best.pth')

        with open(f'./results/{model._get_name()}/{model._get_name()}_best.txt', 'w') as f:
            f.write('best_epoch:    ' + str(best_epoch[-1]))
            f.write('\n')
            f.write('val_acc:    ' + str(validation_acc[-1]))
            f.write('\n')
            f.write('tra_loss:    ' + str(train_epochs_loss[-1]))
            f.write('\n')
            f.write('val_loss:    ' + str(valid_loss[-1]))
            f.close()

    def process(self):
        self.train()


class Test:

    def __init__(self, config):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.device = 'cpu'
        self.model_path = config['model_path']
        self.model_type = config['model_type']
        self.process()

    def _model_load(self):
        self.model = torch.load(self.model_path, map_location=self.device, weights_only=False)

    def _test_data(self):
        self.test_dataset = AnytownBurstDataset('../DataSim/Anytown/anytown_data/t50/gcndata_nopart_error0',
                                                pre_transform=AnytownNormalizeScale())  # path of test dataset

        self.test_dataloader = DataLoader(self.test_dataset, batch_size=len(self.test_dataset), shuffle=True)

    def _test_pre(self):
        self.model.eval()
        if self.model_type == 'MLP':
            for idx, tes in enumerate(self.test_dataloader):
                node_tes = tes.x.to(torch.float32).to(self.device)
                edge_tes = tes.edge_attr.to(torch.float32).to(self.device)
                tes_y = tes.y.to(torch.long).to(self.device)

                pre_y = self.model(node_tes, edge_tes)
                pre_y = pre_y.squeeze(dim=-1)

                prediction = torch.topk(F.softmax(pre_y), k=5, dim=1)[1]
                self.pre_y = prediction.data.to('cpu').numpy().squeeze()
                self.tes_y = tes_y.to('cpu').unsqueeze(-1).numpy()
                self.test_acc = []
                for k in range(1, 6):
                    pre_y_k = self.pre_y[:, :k]
                    mask = (pre_y_k == self.tes_y)
                    result = np.float32(mask.any(axis=1))
                    acc_k = np.sum(result) / len(self.test_dataset)
                    self.test_acc.append(acc_k)

                print('--' * 40)
                for k in range(1, 6):
                    print(f'acc{k} is {self.test_acc[k - 1]:.10f}')
                # print(f'test acc is {self.test_acc * 100:.10f}%')
        elif self.model_type == 'GNN':
            for idx, tes in enumerate(self.test_dataloader):
                tes_y = tes.y.to(torch.long).to(self.device)
                # GNN
                tes = tes.to(self.device)
                pre_y = self.model(tes)
                pre_y = pre_y.squeeze(dim=-1)

                prediction = torch.topk(F.softmax(pre_y), k=5, dim=1)[1]
                self.pre_y = prediction.data.to('cpu').numpy().squeeze()
                self.tes_y = tes_y.to('cpu').unsqueeze(-1).numpy()
                self.test_acc = []
                for k in range(1, 6):
                    pre_y_k = self.pre_y[:, :k]
                    mask = (pre_y_k == self.tes_y)
                    result = np.float32(mask.any(axis=1))
                    acc_k = np.sum(result) / len(self.test_dataset)
                    self.test_acc.append(acc_k)

                print('--' * 40)
                for k in range(1, 6):
                    print(f'acc{k} is {self.test_acc[k - 1]:.10f}')
                # print(f'test acc is {self.test_acc * 100:.10f}%')

        elif self.model_type == 'ResNet':
            for idx, tes in enumerate(self.test_dataloader):
                node_tes = tes.x.to(torch.float32).to(self.device)
                edge_tes = tes.edge_attr.to(torch.float32).to(self.device)
                node_v = node_tes.view(-1, 19, 4)
                edge_v = edge_tes.view(-1, 36, 4)
                inp_v = torch.concat([node_v, edge_v], dim=-2)
                inp_v = inp_v.unsqueeze(1)
                tes_y = tes.y.to(torch.long).to(self.device)

                pre_y = self.model(inp_v)
                pre_y = pre_y.squeeze(dim=-1)

                # prediction = torch.max(F.softmax(pre_y), 1)[1]
                # self.pre_y = prediction.data.to('cpu').numpy().squeeze()
                # self.tes_y = tes_y.to('cpu').numpy().squeeze()
                #
                # self.test_acc = sum(self.pre_y == self.tes_y) / len(self.test_dataset)

                prediction = torch.topk(F.softmax(pre_y), k=5, dim=1)[1]
                self.pre_y = prediction.data.to('cpu').numpy().squeeze()
                self.tes_y = tes_y.to('cpu').unsqueeze(-1).numpy()
                self.test_acc = []
                for k in range(1, 6):
                    pre_y_k = self.pre_y[:, :k]
                    mask = (pre_y_k == self.tes_y)
                    result = np.float32(mask.any(axis=1))
                    acc_k = np.sum(result) / len(self.test_dataset)
                    self.test_acc.append(acc_k)

                print('--' * 40)
                for k in range(1, 6):
                    print(f'acc{k} is {self.test_acc[k - 1]:.10f}')
                # print(f'test acc is {self.test_acc * 100:.10f}%')

        elif self.model_type == 'AlexNet':
            for idx, tes in enumerate(self.test_dataloader):
                node_tes = tes.x.to(torch.float32).to(self.device)
                edge_tes = tes.edge_attr.to(torch.float32).to(self.device)
                node_v = node_tes.view(-1, 19, 4)
                edge_v = edge_tes.view(-1, 36, 4)
                inp_v = torch.concat([node_v, edge_v], dim=-2)
                inp_v = inp_v.unsqueeze(1)
                tes_y = tes.y.to(torch.long).to(self.device)

                pre_y = self.model(inp_v)
                pre_y = pre_y.squeeze(dim=-1)

                prediction = torch.topk(F.softmax(pre_y), k=5, dim=1)[1]
                self.pre_y = prediction.data.to('cpu').numpy().squeeze()
                self.tes_y = tes_y.to('cpu').unsqueeze(-1).numpy()
                self.test_acc = []
                for k in range(1, 6):
                    pre_y_k = self.pre_y[:, :k]
                    mask = (pre_y_k == self.tes_y)
                    result = np.float32(mask.any(axis=1))
                    acc_k = np.sum(result) / len(self.test_dataset)
                    self.test_acc.append(acc_k)

                print('--' * 40)
                for k in range(1, 6):
                    print(f'acc{k} is {self.test_acc[k - 1]:.10f}')
                # print(f'test acc is {self.test_acc * 100:.10f}%')

    def process(self):
        self._model_load()
        self._test_data()
        self._test_pre()


if __name__ == '__main__':
    # for i in range(0,10):
    #     # mlp_model = MlpTrain(model_path=None)
    #     # mlp_model = MlpTrain(model_path=None)
    #     gnn_model = GNNTrain(model_path=None)
    #
    #     # os.rename(f'./results/MLP/MLP_best.txt', f'./results/MLP/MLP_best{i}.txt')
    #     # os.rename(f'./results/MLP/MLP_best.pth',f'./results/MLP/MLP_best{i}.pth')
    #
    #     # os.rename(f'./results/AnyLocGCN/AnyLocGCN_best.txt', f'./results/AnyLocGCN/AnyLocGCN_best{i}.txt')
    #     # os.rename(f'./results/AnyLocGCN/AnyLocGCN_best.pth',f'./results/AnyLocGCN/AnyLocGCN_best{i}.pth')
    #
    #     os.rename(f'./results/DenseAnyLocGcn/DenseAnyLocGcn_best.txt', f'./results/DenseAnyLocGcn/DenseAnyLocGcn_best{i}.txt')
    #     os.rename(f'./results/DenseAnyLocGcn/DenseAnyLocGcn_best.pth',f'./results/DenseAnyLocGcn/DenseAnyLocGcn_best{i}.pth')

    # model_names = ['resnet18', 'alexnet']
    # model_names = ['vgg11','densenet121']

    # for i in range(0,5):
    #     for model_name in model_names:
    #         print(f'Training {model_name}...')
    #         torch_model = TorchTrain(model_name)
    #         # torch.save(model.state_dict(), f'{model_name}_34_classes.pth')
    #         print(f'{model_name} training completed.\n')
    #
    #     if f'ResNet_best.txt' in os.listdir(f'./results/ResNet'):
    #         os.rename(f'./results/ResNet/ResNet_best.txt',f'./results/ResNet/ResNet_best{i}.txt')
    #         os.rename(f'./results/ResNet/ResNet_best.pth',f'./results/ResNet/ResNet_best{i}.pth')
    #         os.rename(f'./results/ModifiedAlexNet/ModifiedAlexNet_best.txt',f'./results/ModifiedAlexNet/ModifiedAlexNet_best{i}.txt')
    #         os.rename(f'./results/ModifiedAlexNet/ModifiedAlexNet_best.pth',f'./results/ModifiedAlexNet/ModifiedAlexNet_best{i}.pth')
    #

    # #
    ##TEST

    # 读取配置文件
    with open('gnn_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    for i in range(0, 10):
        # config['model_path'] = f'./results/ModifiedAlexNet/ModifiedAlexNet_best{i}.pth'
        config['model_path'] = f'./results/DenseAnyLocGcn_GRU/DenseAnyLocGcn_GRU_best{i}.pth'

        print('--' * 40)
        print(f'这是第{i}个模型')
        test = Test(config=config)
