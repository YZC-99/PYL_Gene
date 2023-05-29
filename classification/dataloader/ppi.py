from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import os

class PPIBase(Dataset):
    def __init__(self, ppi_data, essential_data, nonessential_data, embedding_data):
        # 读取 ppi.csv 文件
        ppi_data = pd.read_csv(ppi_data)


        # 读取 essential.csv 文件
        essential_data = pd.read_csv(essential_data)

        # 读取 nonessential.csv 文件
        nonessential_data = pd.read_csv(nonessential_data)

        # 加载 embedding 数据
        embedding_data = np.load(embedding_data)


        self.ppi_data = ppi_data
        self.essential_data = essential_data
        self.nonessential_data = nonessential_data
        self.embedding_data = embedding_data


    def __len__(self):
        return len(self.ppi_data)

    def __getitem__(self, index):
        # example = dict((k, self.labels[k][i]) for k in self.labels)

        name = self.ppi_data.iloc[index]['name']
        id = self.ppi_data.iloc[index]['id']
        label = self.ppi_data.iloc[index]['label']

        feature = self.embedding_data[id]

        data = torch.from_numpy(feature)

        examples = {
            'name':name,
            'ppi_data':data,
            'label':label,
        }
        return examples

class PPITrain(PPIBase):
    def __init__(self,ppi_data='./data/and/train_ppi.csv',
                         essential_data='./data/and/essential.csv',
                         nonessential_data='./data/and/nonessential.csv',
                         embedding_data='./data/and/0005-emebdding.npy'):
        super().__init__(ppi_data=ppi_data,
                         essential_data=essential_data,
                         nonessential_data=nonessential_data,
                         embedding_data=embedding_data)


class PPIEval(PPIBase):
    def __init__(self,ppi_data='./data/and/eval_ppi.csv',
                         essential_data='./data/and/essential.csv',
                         nonessential_data='./data/and/nonessential.csv',
                         embedding_data='./data/and/0005-emebdding.npy'):
        super().__init__(ppi_data,
                         essential_data,
                         nonessential_data,
                         embedding_data,
                         )

# data = PPITrain(ppi_data='../../data/and/ppi.csv',
#                          essential_data='../../data/and/essential.csv',
#                          nonessential_data='../../data/and/nonessential.csv',
#                          embedding_data='../../data/and/0005-emebdding.npy')
