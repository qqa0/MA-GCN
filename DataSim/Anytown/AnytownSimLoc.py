import gc
import sys
from typing import Union, List, Tuple
import mat4py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wntr
from torch_geometric.data import Data, InMemoryDataset

from .AnytownSimLocMethod import *

# from AnytownSimLocMethod import *

'''
author:ChenLei
2025.8.13
'''


class AnytownBase:

    def __init__(self, days_sim):
        self.inp_file = './anytownNormalAverDemand_2.inp'
        self.days_sim = days_sim
        self.base_pattern = np.array(mat4py.loadmat('pattern.mat')['pattern'])
        self.init_process()

    # anytown inp原始文件读取
    def read_inp(self):
        self.wn = wntr.network.WaterNetworkModel(self.inp_file)
        self.wn.options.hydraulic.demand_model = 'PDA'
        self.wn.options.hydraulic.minimum_pressure = 2
        self.wn.options.hydraulic.required_pressure = 5
        self.wn.options.hydraulic.pressure_exponent = 0.5

    # 为每个需水节点随机生成日变化系数
    def rd_random(self):
        self.junc_rd = []
        for i in range(16):
            rd = np.random.normal(loc=1, scale=0.1 / 3.27, size=self.days_sim)
            self.junc_rd.append(rd)

    # 随机需水量系数
    def rt_random(self):
        self.junc_rt = []
        for i in range(16):
            rt = np.random.normal(loc=1, scale=0.1, size=(96, self.days_sim))
            self.junc_rt.append(rt)

    # pattern 设置
    def pattern_set(self):
        # 普通住宅（14个）
        self.P1 = []
        for i in range(14):
            P = self.junc_rd[i] * self.junc_rt[i] * self.base_pattern[:, 0:1]
            P = np.stack(P, axis=1).flatten()
            self.P1.append(P)
        # 郊区
        self.P2 = self.junc_rd[-2] * self.junc_rt[-2] * self.base_pattern[:, 1:2]
        self.P2 = np.stack(self.P2, axis=1).flatten()
        # 工业区
        self.P3 = self.junc_rd[-1] * self.junc_rt[-1] * self.base_pattern[:, 2:3]
        self.P3 = np.stack(self.P3, axis=1).flatten()

    def pattern_confi(self):
        # 普通住宅pattern设置
        P1_lst = ['20', '30', '50', '60', '80', '140', '70', '90', '150', '100', '160', '110', '120', '130']
        for i in range(len(P1_lst)):
            self.wn.add_pattern('pat{}'.format(P1_lst[i]), self.P1[i])
            junc = self.wn.get_node(P1_lst[i])
            junc.demand_timeseries_list.append((junc.base_demand, 'pat{}'.format(P1_lst[i])))
            del junc.demand_timeseries_list[0]
        # 郊区pattern设置
        self.wn.add_pattern('pat40', self.P2)
        junc = self.wn.get_node('40')
        junc.demand_timeseries_list.append((junc.base_demand, 'pat40'))
        del junc.demand_timeseries_list[0]
        # 工业区pattern设置
        self.wn.add_pattern('pat170', self.P3)
        junc = self.wn.get_node('170')
        junc.demand_timeseries_list.append((junc.base_demand, 'pat170'))
        del junc.demand_timeseries_list[0]

    def init_process(self):
        self.read_inp()
        self.rd_random()
        self.rt_random()
        self.pattern_set()
        self.pattern_confi()


class AnytownLocation(AnytownBase):

    def __init__(self, days_sim, times):
        super(AnytownLocation, self).__init__(days_sim)
        self.data_sim(times)

    def data_sim(self, times):
        names = locals()
        sur_lst = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]

        num = 0
        label = np.array([0] * 3 + [1] * 5)

        num_pipe = len(self.wn.pipe_name_list)

        path_no_error = f'./anytown_data/t{times}/loc'
        path_error = f'./anytown_data/t{times}/error_add'

        if not os.path.exists(path_no_error):
            os.makedirs(path_no_error)

        if not os.path.exists(path_error):
            os.makedirs(f'{path_error}/pic')
            for idx in sur_lst:
                os.makedirs(f'{path_error}/sur_{idx}')

        for pipe_ in self.wn.pipe_name_list:
            # for pipe_ in ['P1036', 'P1078']:

            writer = pd.ExcelWriter(f'{path_no_error}/{pipe_}.xlsx')
            for idx in sur_lst:
                names[f'writer{idx}'] = pd.ExcelWriter(f'{path_error}/sur_{idx}/{pipe_}.xlsx')

            print(f'\n还剩{num_pipe}根管道没有模拟!')
            num_pipe -= 1

            for idx in sur_lst:
                names[f'error{idx}'] = []
            for i in range(times):
                self.init_process()
                # 爆管时间
                break_time = np.random.randint(48, 73) + np.random.choice([i / 4 for i in range(0, 4)])  # 爆管时间设定在第5天
                # 爆管面积
                break_area = (((self.wn.get_link(pipe_).diameter) ** 2) / 4) * np.pi * (np.random.random() * 0.8 + 0.05)
                # 爆管位置
                break_location = np.random.random()
                self.wn.options.time.duration = 24 * self.days_sim * 3600 - 900
                self.wn = wntr.morph.split_pipe(self.wn, pipe_, pipe_ + '_B', pipe_ + '_leak_node',
                                                split_at_point=break_location)
                leak_node = self.wn.get_node(pipe_ + '_leak_node')
                leak_node.add_leak(self.wn, area=break_area, start_time=break_time * 3600,
                                   end_time=(break_time + 3) * 3600)
                sim = wntr.sim.WNTRSimulator(self.wn)
                results = sim.run_sim()
                df1 = pd.DataFrame(results.node['pressure'][['30', '110', '150', '170']])
                df2 = pd.DataFrame(results.link['flowrate'][['P1080', 'P1078', 'P1056', 'P1036']] * 1000)
                df = pd.concat([df1, df2], axis=1)
                break_date = pd.to_datetime('20220101 00:00') + pd.Timedelta(hours=break_time)
                data_start = break_date - pd.Timedelta(minutes=45)
                data_end = break_date + pd.Timedelta(hours=1)
                col = df.columns.copy()
                # 添加误差
                for sur in sur_lst:
                    redf = pd.DataFrame(map(lambda x: self.wgn(x, sur), [df[ii] for ii in df.columns])).T
                    # 添加随机扰动后，初步计算误差,并修正最大最小添加误差情况
                    error = (redf - df) / df * 100
                    redf = self.fresh_error(error, redf.copy(), df.copy())
                    error = (redf - df) / df * 100
                    names[f'error{sur}'].append(error)
                    redf.index = pd.date_range('20220101 00:00', periods=self.days_sim * 96, freq='15min')
                    redf = redf[data_start:data_end]

                    redf['label'] = label
                    redf['break_time'] = break_date
                    break_demand = results.node['leak_demand'][leak_node.name].max()
                    redf['leak_demand'] = break_demand * 1000
                    redf['leak_location'] = leak_node.name
                    redf.to_excel(names[f'writer{sur}'], sheet_name=f'Sheet_{i + 1}')

                df.index = pd.date_range('20220101 00:00', periods=self.days_sim * 96, freq='15min')
                df = df[data_start:data_end]
                df['label'] = label
                df['break_time'] = break_date
                break_demand = results.node['leak_demand'][leak_node.name].max()
                df['leak_demand'] = break_demand * 1000
                df['leak_location'] = leak_node.name

                df.to_excel(writer, sheet_name=f'Sheet_{i + 1}')
                sys.stdout.write(
                    '\r' + '管段{}的第{}模拟:爆管发生在{}，爆管面积为{:.4f}平方米'.format(pipe_, i + 1, break_date,
                                                                                         break_area))
                num += 1

            for sur in sur_lst:
                names[f'error{sur}'] = np.array(names[f'error{sur}'])
                names[f'error{sur}'] = np.mean(names[f'error{sur}'], axis=0)
                names[f'error{sur}'] = pd.DataFrame(names[f'error{sur}'], columns=col)
                self.error_plot(names[f'error{sur}'], sur, pipe_, times)

            writer._save()
            writer.close()
            del writer

            for sur in sur_lst:
                names[f'writer{sur}']._save()
                names[f'writer{sur}'].close()
                del names[f'writer{sur}']
            gc.collect()

    @staticmethod
    def wgn(sequence, snr):
        Ps = np.sum(abs(sequence) ** 2) / len(sequence)
        Pn = Ps / (10 ** ((snr / 10)))
        noise = np.random.randn(len(sequence)) * np.sqrt(Pn)
        signal_add_noise = sequence + noise

        return signal_add_noise

    @staticmethod
    def fresh_error(error, redata, data):
        # 确定误差的上下界限
        m1 = error.quantile(q=0.25)
        m2 = error.quantile(q=0.75)
        m3 = m2 - m1
        top = m2 + 1.5 * m3
        bottom = m2 - 1.5 * m3

        # 进行添加误差上下限的修正
        idx = error > top
        redata[idx] = data[idx] * (1 + top / 100)
        idx = error < bottom
        redata[idx] = data[idx] * (1 + bottom / 100)

        return redata

    @staticmethod
    def error_plot(error, sur, burst_pipe, times):
        error.boxplot()
        if not os.path.exists(f'./anytown_data/t{times}/error_add/pic/sur_{sur}'):
            os.mkdir(f'./anytown_data/t{times}/error_add/pic/sur_{sur}')
        plt.savefig(f'./anytown_data/t{times}/error_add/pic/sur_{sur}/{burst_pipe}_error.png')
        plt.close()


class GcnDataMake:

    def __init__(self, path):
        self.path = path
        self.files = os.listdir(path)
        self.labels = {name[-10:-5]: i for i, name in enumerate(self.files)}
        self.datalist = []
        self.process()

    def _topology(self):
        self.wn = wntr.network.WaterNetworkModel(r'xxxx\anytownNormalAverDemand_2.inp')  # path of inp
        G = self.wn.to_graph()
        G = G.to_directed()
        self.node_mapping = {name: i for i, name in enumerate(sorted(self.wn.node_name_list))}
        _edges_coo = [(edge[0], edge[1]) for edge in G.edges()]
        edges_coo = [(self.node_mapping[start], self.node_mapping[end]) for start, end in sorted(_edges_coo)]
        self.edges_coo = np.array(edges_coo).T
        self.edges_index = torch.tensor(self.edges_coo, dtype=torch.long)

    def _node_pipe_hash_conversion(self):
        pressure_node = ['30', '110', '150', '170']
        flow_pipe = ['P1080', 'P1078', 'P1056', 'P1036']

        # node
        self.node_hash = []
        for n in pressure_node:
            self.node_hash.append(self.node_mapping[n])

        # pipe
        pipe_hash_start, pipe_hash_end = [], []
        for p in flow_pipe:
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
        for file in tqdm(self.files, position=0, leave=True):

            excel_ = pd.ExcelFile(f'{self.path}/{file}')
            sheet = excel_.sheet_names
            label = self.labels[file[-10:-5]]
            y = torch.tensor([label], dtype=torch.int)

            # for sheet_ in tqdm(sheet, position=1, leave=False):
            for sheet_ in sheet:

                df = pd.read_excel(f'{self.path}/{file}', sheet_name=sheet_, parse_dates=True)
                data = df.iloc[:, 1:9].values
                for i in range(4):
                    node_attr = np.zeros((19, 4))
                    edge_attr = np.zeros((36, 4))

                    pressure_ = data[i:i + 4, :4]
                    flow_ = data[i:i + 4, 4:]

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


class AnytownBurstDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.pre_transform = pre_transform
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        return ['data.pt']

    def process(self):
        data = GcnDataMake(r'xxxxx')  # gcndata path
        data_list = data.datalist

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])

        return data


# 用于提取每个时刻的爆管数据制作样本
class GcnDataMakePart:

    def __init__(self, path):
        self.path = path
        # self.files = os.listdir(path)
        self.files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        self.labels = {name[-10:-5]: i for i, name in enumerate(self.files)}
        self.datalist = []
        self.process()

    def _topology(self):
        self.wn = wntr.network.WaterNetworkModel(r'xxx\anytownNormalAverDemand_2.inp')  # path of inp
        G = self.wn.to_graph()
        G = G.to_directed()
        self.node_mapping = {name: i for i, name in enumerate(sorted(self.wn.node_name_list))}
        _edges_coo = [(edge[0], edge[1]) for edge in G.edges()]
        edges_coo = [(self.node_mapping[start], self.node_mapping[end]) for start, end in sorted(_edges_coo)]
        self.edges_coo = np.array(edges_coo).T
        self.edges_index = torch.tensor(self.edges_coo, dtype=torch.long)

    def _node_pipe_hash_conversion(self):
        pressure_node = ['30', '110', '150', '170']
        flow_pipe = ['P1080', 'P1078', 'P1056', 'P1036']

        # node
        self.node_hash = []
        for n in pressure_node:
            self.node_hash.append(self.node_mapping[n])

        # pipe
        pipe_hash_start, pipe_hash_end = [], []
        for p in flow_pipe:
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

    def _data_make(self, part):
        for file in tqdm(self.files, position=0, leave=True):

            excel_ = pd.ExcelFile(f'{self.path}/{file}')
            sheet = excel_.sheet_names
            label = self.labels[file[-10:-5]]
            y = torch.tensor([label], dtype=torch.int)

            for sheet_ in sheet:

                df = pd.read_excel(f'{self.path}/{file}', sheet_name=sheet_, parse_dates=True)
                data = df.iloc[:, 1:9].values
                for i in range(4):
                    node_attr = np.zeros((19, 4))
                    edge_attr = np.zeros((36, 4))

                    pressure_ = data[i:i + 4, :4]
                    flow_ = data[i:i + 4, 4:]

                    for pre, node in enumerate(self.node_hash):
                        node_attr[node] = pressure_[:, pre]

                    for flo, pipe in enumerate(self.pipe_hash):
                        edge_attr[pipe] = flow_[:, flo]

                    if i == part:  # part=0,1,2,3 0为第一个时刻 依次类推
                        x = torch.tensor(node_attr, dtype=torch.float)
                        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

                        data_ = Data(edge_index=self.edges_index, edge_attr=edge_attr, x=x, y=y)
                        self.datalist.append(data_)

    def process(self):
        self._topology()
        self._node_pipe_hash_conversion()
        self._data_make(part=3)


class AnytownBurstDatasetPart(InMemoryDataset):  # Used for searching for data related to delayed pipe bursts
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.pre_transform = pre_transform
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        return ['data.pt']

    def process(self):
        data = GcnDataMakePart(r'xxxx')  # gcndata path
        data_list = data.datalist

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # data, slices = self.collate(data_list)
        self.save(data_list, self.processed_paths[0])

        return data
