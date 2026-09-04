from scipy.stats import norm
import pandas as pd
import numpy as np
import os


class Logger:
    """训练日志记录器，所有指标合并保存为单个 ``logs.csv``（长表：step, metric, value）。"""

    def __init__(self, log_name_list, save_dir):
        self.log_name_list = log_name_list
        self.save_dir = save_dir
        self.log_dict = {log_name: [] for log_name in log_name_list}

    def write(self, log_name, data):
        self.log_dict[log_name].append(data)

    def writes(self, log_name, datas):
        assert type(datas) == list
        self.log_dict[log_name] += datas

    def save(self):
        rows = []
        for log_name in self.log_name_list:
            for step, value in self.log_dict[log_name]:
                rows.append((step, log_name, value))
        if rows:
            os.makedirs(self.save_dir, exist_ok=True)
            csv_path = f"{self.save_dir}/logs.csv"
            df = pd.DataFrame(rows, columns=['step', 'metric', 'value'])
            df.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
        self.log_dict = {log_name: [] for log_name in self.log_name_list}

    def get_avg(self, log_name, length=1, per_episode=True):
        length = min(len(self.log_dict[log_name]), length)
        if length == 0:
            return None
        if per_episode:
            temp_data = [item[1] for item in self.log_dict[log_name][-length:]]
        else:
            temp_data = [item[1] / item[0] for item in self.log_dict[log_name][-length:]]
        return np.mean(temp_data)

    def get_std(self, log_name, length=1):
        log = self.log_dict[log_name]
        length = min(len(log), length)
        if length == 0:
            return None
        temp_data = [item[1] for item in log[-length:]]
        return np.std(temp_data)

    def get_square(self, log_name, length=1):
        log = self.log_dict[log_name]
        length = min(len(log), length)
        if length == 0:
            return None
        temp_data = [item[1] for item in log[-length:]]
        return np.mean(np.square(temp_data))

    def get_cvar(self, log_name, cost_alpha, length=1):
        log = self.log_dict[log_name]
        length = min(len(log), length)
        if length == 0:
            return None
        temp_data = [item[1] for item in log[-length:]]
        df = pd.DataFrame({"data": temp_data})
        cvar = df.quantile(q=1.0 - cost_alpha)[0]
        return cvar

    def get_cvar2(self, log_name, cost_alpha, length=1):
        log = self.log_dict[log_name]
        length = min(len(log), length)
        if length == 0:
            return None
        temp_data = [item[1] for item in log[-length:]]
        mean = np.mean(temp_data)
        std = np.std(temp_data)
        sigma_unit = norm.pdf(norm.ppf(cost_alpha)) / cost_alpha
        cvar = mean + sigma_unit * std
        return cvar
