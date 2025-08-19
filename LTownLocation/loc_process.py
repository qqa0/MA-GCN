import pandas as pd
import numpy as np
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from loc_method import *
from Location.DataSim.LTown.LtownDataMake import LtownBurstDataset
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import wntr
import os
import yaml

'''
author:ChenLei
2025.8.13
'''


# MA-GCN
class GNNTrain:

    def __init__(self, model_path=None):
        self.model_path = model_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_save_path = './results/DenseLtownLocGcn_partB'
        os.makedirs(self.model_save_path, exist_ok=True)
        self.early_stop_patience = 50  # 提高耐心
        self.process()

    def train(self):
        dataset = LtownBurstDataset('../DataSim/LTown/partB/t200/g_error_0')  # data path
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        model = torch.load(self.model_path, map_location=self.device) if self.model_path else DenseLtownLocGcn().to(
            self.device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0008, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                               patience=5, verbose=True, min_lr=1e-6)

        best_val_loss = float('inf')
        best_acc = 0
        best_epoch = -1
        patience = 0
        logs = []

        for epoch in range(1000):
            model.train()
            train_losses, correct, total = [], 0, 0
            for batch in train_loader:
                batch = batch.to(self.device)
                y_true = batch.y.long().to(self.device)
                out = model(batch)
                loss = criterion(out, y_true)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())
                preds = out.argmax(dim=1)
                correct += (preds == y_true).sum().item()
                total += y_true.size(0)

            train_loss = np.mean(train_losses)
            train_acc = correct / total

            model.eval()
            val_losses, correct, total = [], 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    y_true = batch.y.long().to(self.device)
                    out = model(batch)
                    loss = criterion(out, y_true)
                    val_losses.append(loss.item())
                    preds = out.argmax(dim=1)
                    correct += (preds == y_true).sum().item()
                    total += y_true.size(0)

            val_loss = np.mean(val_losses)
            val_acc = correct / total
            scheduler.step(val_loss)

            print(f'[Epoch {epoch}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}')

            logs.append({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc,
                         'val_loss': val_loss, 'val_acc': val_acc})

            # if val_loss < best_val_loss:
            if val_acc > best_acc:
                best_acc = val_acc
                best_val_loss = val_loss
                best_epoch = epoch
                patience = 0
                torch.save(model.state_dict(), os.path.join(self.model_save_path, 'best_model.pth'))
            else:
                patience += 1
                if patience >= self.early_stop_patience:
                    print(f'[Early stopping] Epoch {epoch} with best epoch {best_epoch}')
                    break

        df_log = pd.DataFrame(logs)
        df_log.to_csv(os.path.join(self.model_save_path, 'train_log.csv'), index=False)
        with open(os.path.join(self.model_save_path, 'best_summary.txt'), 'w') as f:
            f.write(f'Best Epoch: {best_epoch}\n')
            f.write(f'Best Validation Loss: {best_val_loss:.4f}\n')
            f.write(f'Best Validation Accuracy: {logs[best_epoch]["val_acc"]:.4f}\n')

    def process(self):
        self.train()


class Test:

    def __init__(self, config):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_path = config['model_path']
        self.model_type = config['model_type']
        self.process()

    def _model_load(self):
        self.model = DenseLtownLocGcn().to(self.device)
        self.model.load_state_dict(torch.load(self.model_path))

    def _test_data(self):
        ######################################################################################
        self.test_dataset = LtownBurstDataset('../DataSim/LTown/partB/t20/g_error32')
        #######################################################################################
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=64, shuffle=True)

    def _test_pre(self):
        self.model.eval()
        if self.model_type == 'GNN':

            self.pre_y, self.tes_y = [], []
            for batch in self.test_dataloader:
                batch_tes_y = batch.y.long().to(self.device)
                batch_tes = batch.to(self.device)
                batch_pre_y = self.model(batch_tes)
                batch_pre_y = batch_pre_y.squeeze(dim=-1)

                batch_prediction = torch.topk(F.softmax(batch_pre_y), k=5, dim=1)[1]
                batch_pre_y = batch_prediction.data.to('cpu').numpy().squeeze()
                batch_tes_y = batch_tes_y.to('cpu').unsqueeze(-1).numpy()
                self.pre_y.append(batch_pre_y)
                self.tes_y.append(batch_tes_y)

            self.pre_y = np.concatenate(self.pre_y, axis=0)
            self.tes_y = np.concatenate(self.tes_y, axis=0)
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

    def _distance_burst(self):
        wn = wntr.network.WaterNetworkModel(r'xxxx\L-TOWN.inp')  # path of inp
        dirs = os.listdir(r'D:\OneDrive\CQU\会议及文章\01 博士论文\Location\DataSim\LTown\partB\t20\error0')
        labels = {name: i for i, name in enumerate(sorted(dirs))}

        # 反转字典，值 -> 键
        inv_labels = {v: k for k, v in labels.items()}

        G = wn.to_graph()
        G = G.to_undirected()

        for u, v, n in G.edges:
            for u1, v1, d in G.edges(data=True):
                if u == u1 and v == v1:
                    d['name'] = n
                    pipe = wn.get_link(n)
                    if hasattr(pipe, 'length'):
                        d['weight'] = pipe.length

        pip_ture = self.tes_y[:, 0].copy()
        pip_pre = self.pre_y[:, 0].copy()
        pip_true = [inv_labels[val] for val in pip_ture]
        pip_pre = [inv_labels[val] for val in pip_pre]

        self.error_distance = []
        for i in range(len(pip_pre)):
            if pip_pre[i] != pip_true[i]:
                pip_true_ = wn.get_link(pip_true[i])
                pip_pre_ = wn.get_link(pip_pre[i])

                length_ture_ = wn.get_link(pip_true[i]).length
                length_pre_ = wn.get_link(pip_pre[i]).length

                start_true, end_true = pip_true_.start_node_name, pip_true_.end_node_name
                start_pre, end_pre = pip_pre_.start_node_name, pip_pre_.end_node_name

                candidates = [(start_true, start_pre), (start_true, end_pre), (end_true, start_pre),
                              (end_true, end_pre)]
                candidate_path = -1000
                for s, t in candidates:
                    length_candidate = nx.shortest_path_length(G, s, t, weight='weight')
                    if length_candidate > candidate_path:
                        candidate_path = length_candidate

                min_path_length = candidate_path - 0.5 * (length_ture_ + length_pre_)
                self.error_distance.append(min_path_length)
        print('--' * 40)
        # print(self.error_distance)

    def _distance_cal(self):
        sns.boxplot(y=self.error_distance)
        plt.title("Boxplot of Data")
        plt.ylabel("Value")
        plt.grid(True)
        plt.show()

    def process(self):
        self._model_load()
        self._test_data()
        self._test_pre()
        self._distance_burst()
        self._distance_cal()


if __name__ == '__main__':
    # for i in range(1002,1010):
    #     gnn_model = GNNTrain()
    #
    #     os.rename(f'./results/DenseLtownLocGcn_partB/best_summary.txt', f'./results/DenseLtownLocGcn_partB/best_summary_{i}.txt')
    #     os.rename(f'./results/DenseLtownLocGcn_partB/best_model.pth', f'./results/DenseLtownLocGcn_partB/best_model_{i}.pth')
    #     os.rename(f'./results/DenseLtownLocGcn_partB/train_log.csv', f'./results/DenseLtownLocGcn_partB/train_log_{i}.csv')
    #

    with open('gnn_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    for i in range(1000, 1005):
        config['model_path'] = f'./results/DenseLtownLocGcn_partB/best_model_{i}.pth'

        print('--' * 40)
        print(f'这是第{i}个模型')
        test = Test(config=config)

        with open(fr'xx\Ltown\partB\error32\error_distance_{i}.txt', 'w',
                  encoding='utf-8') as f:
            for item in test.error_distance:
                f.write(f"{item}\n")
            f.close()

        with open(fr'xx\Ltown\partB\error32\test_acc_{i}.txt', 'w',
                  encoding='utf-8') as f:
            for item in test.test_acc:
                f.write(f"{item}\n")
            f.close()

        np.savetxt(fr'xx\Ltown\partB\error32\pre_y_{i}.txt', test.pre_y, fmt='%d',
                   delimiter='\t')

        np.savetxt(fr'xx\Ltown\partB\error32\tru_y_{i}.txt', test.tes_y, fmt='%d',
                   delimiter='\t')
