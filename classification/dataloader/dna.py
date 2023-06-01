from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import os

class DNABase(Dataset):
    def __init__(self, label_path, essential_data, nonessential_data,features=4545):
        # 读取 ppi.csv 文件
        label_data = pd.read_csv(label_path)

        # 读取 essential.csv 文件
        essential_data = pd.read_csv(essential_data)

        # 读取 nonessential.csv 文件
        nonessential_data = pd.read_csv(nonessential_data)



        self.label_data = label_data
        self.essential_data = essential_data
        self.nonessential_data = nonessential_data
        self.features = features


    def __len__(self):
        return len(self.label_data)

    def __getitem__(self, index):
        '''
        需要根据name在相应的df中查询的指定行，并将该行的最后4545列返回作为，feature,代码不完整，你需要补充完整
        '''
        name = self.label_data.iloc[index]['name']
        label = self.label_data.iloc[index]['label']
        if label == 1:
            feature = self.essential_data.loc[self.essential_data['Gene_name'] == name].iloc[:, -self.features:].values
        else:
            feature = self.nonessential_data.loc[self.nonessential_data['gene'] == name].iloc[:, -self.features:].values

        if feature.shape[0]== 0:
            feature = np.reshape(feature,(1,-1))
        # feature = feature.reshape((4545,))
        feature = np.float16(feature)
        data = torch.from_numpy(feature)
        data = torch.squeeze(data).float()

        examples = {
            'name':name,
            'dna_data':data,
            'label':int(label),
        }
        return examples

class DNATrain(DNABase):
    def __init__(self,label_path='./data/human/train.csv',
                         essential_data='./data/human/deg_nonan_data_with_feature.csv',
                         nonessential_data='./data/human/ccds_nonan_data_with_feature.csv',
                         ):
        super().__init__(label_path=label_path,
                         essential_data=essential_data,
                         nonessential_data=nonessential_data,
                         )


class DNAEval(DNABase):
    def __init__(self,label_path='./data/human/eval.csv',
                         essential_data='./data/human/deg_nonan_data_with_feature.csv',
                         nonessential_data='./data/human/ccds_nonan_data_with_feature.csv',
                         ):
        super().__init__(label_path=label_path,
                         essential_data=essential_data,
                         nonessential_data=nonessential_data,
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


