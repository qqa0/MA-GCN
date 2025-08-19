import copy
from pathlib import Path
import time
from fractions import Fraction
import sys
import wntr
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import gc
from collections import deque

'''
author:ChenLei
2025.8.13
'''


class LtownBase:
    def __init__(self, days_sim):
        self.inp_file = './L-TOWN.inp'
        self.days_sim = days_sim
        self.init_process()

    def read_inp(self):
        self.wn = wntr.network.WaterNetworkModel(self.inp_file)
        self.wn.options.hydraulic.demand_model = 'PDA'
        self.wn.options.hydraulic.minimum_pressure = 10
        self.wn.options.hydraulic.required_pressure = 20
        self.wn.options.hydraulic.pressure_exponent = 0.5
        self.wn.options.time.duration = 24 * self.days_sim * 3600 - 300

    def get_base_pattern(self):
        self.base_pattern = {}
        for i in self.wn.pattern_name_list:
            dic = self.wn.get_pattern(i).to_dict()
            if i != 'P-Industrial':
                self.base_pattern[i] = np.array(dic['multipliers'] * 365)
            else:
                self.base_pattern[i] = np.array(dic['multipliers'])

    def rd_random(self):
        rd = np.random.normal(loc=1, scale=0.1 / 3.27)
        return rd

    def rt_random(self):
        rt = np.random.normal(loc=1, scale=0.1, size=(288,))
        return rt

    def pattern_set(self):
        length = self.days_sim * 288
        num = 0
        for j in self.wn.junction_name_list:
            junc = self.wn.get_node(j)
            pattern_lst = junc.demand_timeseries_list.to_list()
            for pat in pattern_lst:
                pattern_name = pat['pattern_name']
                if pattern_name != 'P-Industrial':
                    ts_base = self.base_pattern[pattern_name][:length]
                    ts_coeff = np.concatenate([self.rd_random() * self.rt_random()] * self.days_sim)
                    ts = ts_coeff * ts_base
                    self.wn.add_pattern(f'{pattern_name}_{j}_{num}', ts)
                    junc.demand_timeseries_list.append((pat['base_val'], f'{pattern_name}_{j}_{num}'))
                    num += 1
                else:
                    ts = self.base_pattern[pattern_name]
                    self.wn.add_pattern(f'{pattern_name}_{j}_{num}', ts)
                    junc.demand_timeseries_list.append((pat['base_val'], f'{pattern_name}_{j}_{num}'))
                    num += 1
            del junc.demand_timeseries_list[0:3]

    def init_process(self):
        self.read_inp()
        self.get_base_pattern()
        self.pattern_set()


class LtownSimLoc(LtownBase):
    def __init__(self, days_sim, pressure_sensors, flow_sensors, pipes, times, path):
        super(LtownSimLoc, self).__init__(days_sim)
        self.pressure_sensors = pressure_sensors
        self.flow_sensors = flow_sensors
        self.pipes = pd.read_excel(pipes, header=None).values.flatten()
        # self.nodes = pd.read_excel(nodes,header=None).values.flatten()
        self.times = times
        self.path = path

    def process(self):
        self.init_process()
        names = locals()
        sur_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                   28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]

        num = 0
        label = np.array([0] * 11 + [1] * 13)
        os.makedirs(self.path, exist_ok=True)
        total = len(self.pipes) * self.times
        t_deque = deque(maxlen=100)

        for pipe_ in self.pipes:
            max_num = 0  # 每个文件夹中的csv数量
            #########################################路径改这里##########################################
            out_dir = Path(self.path) / f't{self.times}' / 'error0' / pipe_
            out_dir.mkdir(parents=True, exist_ok=True)

            if len(list(out_dir.glob('*.csv'))) == self.times:
                total = total - self.times
                continue
            elif len(list(out_dir.glob('*.csv'))) != self.times:
                csv_files = [f for f in os.listdir(out_dir) if f.endswith('.csv')]
                if not csv_files:
                    max_num = 0
                else:
                    max_num = max(int(f.split('_')[0]) for f in os.listdir(out_dir) if f.endswith('.csv'))
                    total = total - max_num

            for idx in sur_lst:
                names[f'error{idx}'] = []

            for j in range(max_num, self.times):

                t1 = time.perf_counter()

                wn = copy.deepcopy(self.wn)
                kk = np.random.choice(list(range(1, 12)))
                break_time = np.random.randint(24, 46) + Fraction(kk, 12)
                break_location = np.random.random()

                break_area = (((wn.get_link(pipe_).diameter) ** 2) / 4) * np.pi * (np.random.random() * 0.9 + 0.05)

                wn = wntr.morph.split_pipe(wn, pipe_, pipe_ + '_B', pipe_ + '_leak_node',
                                           split_at_point=break_location)
                leak_node = wn.get_node(pipe_ + '_leak_node')
                leak_node.add_leak(wn, area=break_area, start_time=break_time * 3600,
                                   end_time=(break_time + 3) * 3600 - 300)

                sim = wntr.sim.WNTRSimulator(wn)
                results = sim.run_sim()
                df1 = pd.DataFrame(results.node['pressure'][self.pressure_sensors])
                df2 = pd.DataFrame(results.link['flowrate'][self.flow_sensors] * 1000)
                df = pd.concat([df1, df2], axis=1)

                break_date = pd.to_datetime('2022-01-01 00:00') + pd.Timedelta(
                    hours=int(break_time)) + pd.Timedelta(minutes=kk * 5)
                start = break_date - pd.Timedelta(minutes=55)
                end = break_date + pd.Timedelta(hours=1)
                col = df.columns.copy()

                for sur in sur_lst:
                    #########################################路径改这里##########################################
                    sur_out_dir = Path(self.path) / f't{self.times}' / f'error_{sur}' / pipe_
                    sur_out_dir.mkdir(parents=True, exist_ok=True)

                    redf = pd.DataFrame(map(lambda x: self.wgn(x, sur), [df[ii] for ii in df.columns])).T
                    # 添加随机扰动后，初步计算误差,并修正最大最小添加误差情况
                    error = np.abs((np.abs(redf) - np.abs(df))) / np.abs(df) * 100
                    redf = self.fresh_error(error, redf.copy(), df.copy())
                    error = np.abs((np.abs(redf) - np.abs(df))) / np.abs(df) * 100
                    names[f'error{sur}'].append(error)
                    redf.index = pd.date_range('2022-01-01 00:00', periods=self.days_sim * 288, freq='5min')
                    redf = redf.loc[start:end]

                    redf['label'] = label
                    redf['break_time'] = break_date
                    break_demand = results.node['leak_demand'][leak_node.name].max()
                    redf['leak_demand'] = break_demand * 1000
                    redf['leak_location'] = leak_node.name

                    surfile = sur_out_dir / f'{j + 1}_{pipe_}.csv'
                    redf.to_csv(surfile, index=True)

                df.index = pd.date_range('2022-01-01 00:00', periods=self.days_sim * 288, freq='5min')
                df = df.loc[start:end]
                df['label'] = label
                df['break_time'] = break_date
                break_demand = results.node['leak_demand'][leak_node.name].max()
                df['leak_demand'] = break_demand * 1000
                df['leak_location'] = leak_node.name

                df_file = out_dir / f'{j + 1}_{pipe_}.csv'
                df.to_csv(df_file, index=True)

                t2 = time.perf_counter()
                elapsed = t2 - t1
                t_deque.append(elapsed)
                total -= 1
                sys.stdout.write(
                    f'\r管段 {pipe_} 第 {j + 1} 次模拟完成, 用时 {elapsed:.2f}s, 剩余 {total} 次,预计剩余时间{np.mean(t_deque) * total / 3600:.2f}h')
                sys.stdout.flush()

            for sur in sur_lst:
                names[f'error{sur}'] = np.array(names[f'error{sur}'])
                names[f'error{sur}'] = np.mean(names[f'error{sur}'], axis=0)
                names[f'error{sur}'] = pd.DataFrame(names[f'error{sur}'], columns=col)
                self.error_plot(self.path, names[f'error{sur}'], sur, pipe_, self.times)

            gc.collect()

        print('\n全部模拟完成。')

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
    def error_plot(path, error, sur, burst_pipe, times):
        error.boxplot()
        plt_dir = Path(path) / f't{times}' / 'pic' / f'error_{sur}'
        plt_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{plt_dir}/{burst_pipe}.png')
        plt.close()


if __name__ == '__main__':
    # #partA
    pressure_sensors = ['n1', 'n7', 'n15', 'n41', 'n363']
    flow_sensors = ['p256', 'p11', 'p263', 'p249', 'p295']
    pipes = './partA/pipes_A.xlsx'
    nodes = './partA/nodes_A.xlsx'

    model = LtownSimLoc(3, pressure_sensors, flow_sensors, pipes, 20, path='./partA')
    model.process()

    # partB
    pressure_sensors = ['n207', 'n215', 'n227', 'n253', 'n337']
    flow_sensors = ['p74', 'p195', 'p236', 'p674', 'p679']
    pipes = './partB/pipes_B.xlsx'
    nodes = './partB/nodes_B.xlsx'

    model = LtownSimLoc(3, pressure_sensors, flow_sensors, pipes, 20, path='./partB')
    model.process()
