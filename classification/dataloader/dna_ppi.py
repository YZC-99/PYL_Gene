from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import os

class DNA_PPIBase(Dataset):
    def __init__(self, label_path, essential_data, nonessential_data,embedding_data,features=4545):
        # 读取 ppi.csv 文件
        label_data = pd.read_csv(label_path)

        # 读取 essential.csv 文件
        essential_data = pd.read_csv(essential_data)

        # 读取 nonessential.csv 文件
        nonessential_data = pd.read_csv(nonessential_data)

        # 加载 embedding 数据
        embedding_data = np.load(embedding_data)

        self.label_data = label_data
        self.essential_data = essential_data
        self.nonessential_data = nonessential_data
        self.embedding_data = embedding_data
        self.features = features


    def __len__(self):
        return len(self.label_data)

    def __getitem__(self, index):
        '''
        需要根据name在相应的df中查询的指定行，并将该行的最后4545列返回作为，feature,代码不完整，你需要补充完整
        '''
        name = self.label_data.iloc[index]['name']
        id = self.label_data.iloc[index]['id']
        label = self.label_data.iloc[index]['label']
        if label == 1:
            dna_feature = self.essential_data.loc[self.essential_data['Gene_name'] == name].iloc[:, -self.features:].values
        else:
            dna_feature = self.nonessential_data.loc[self.nonessential_data['gene'] == name].iloc[:, -self.features:].values

        if dna_feature.shape[0]== 0:
            dna_feature = np.reshape(dna_feature,(1,-1))
        # feature = feature.reshape((4545,))
        dna_feature = np.float16(dna_feature)
        dna_data = torch.from_numpy(dna_feature)
        dna_data = torch.squeeze(dna_data).float()

        ppi_feature = self.embedding_data[id]
        ppi_data = torch.from_numpy(ppi_feature)

        examples = {
            'name':name,
            'dna_data':dna_data,
            'dna_data9':dna_data[:9],
            'ppi_data':ppi_data,
            'label':int(label),
        }
        return examples

class DNA_PPITrain(DNA_PPIBase):
    def __init__(self,label_path='./data/human/train.csv',
                         essential_data='./data/human/deg_nonan_data_with_feature.csv',
                         nonessential_data='./data/human/ccds_nonan_data_with_feature.csv',
                         embedding_data='./data/human/0005-emebdding.npy',
                         features=4545,
                         ):
        super().__init__(label_path=label_path,
                         essential_data=essential_data,
                         nonessential_data=nonessential_data,
                         embedding_data=embedding_data,
                         features=features
                         )


class DNA_PPIEval(DNA_PPIBase):
    def __init__(self,label_path='./data/human/eval.csv',
                         essential_data='./data/human/deg_nonan_data_with_feature.csv',
                         nonessential_data='./data/human/ccds_nonan_data_with_feature.csv',
                         embedding_data='./data/human/0005-emebdding.npy',
                         features=4545,
                         ):
        super().__init__(label_path=label_path,
                         essential_data=essential_data,
                         nonessential_data=nonessential_data,
                         embedding_data=embedding_data,
                         features=features
                         )

# from torch.utils.data import DataLoader
# dataset = DNATrain(label_path='../../data/human/eval.csv',
#                          essential_data='../../data/human/deg_dna_with_feature.csv',
#                          nonessential_data='../../data/human/ccds_data_with_feature.csv',)
#
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
#
# for batch in dataloader:
#     # print(batch['dna_data'])
#     print(batch['label'])


