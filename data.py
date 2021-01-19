
import os
import sys
import utils
import numpy as np
import pandas as pd 
from torch.utils.data import Dataset, DataLoader

path = 'audio_data/split_data/'

class VoiceData(Dataset):

    def __init__(self, df, in_col, out_col):

        self.df = df
        self.data = []
        self.labels = []
        self.cat2idx = {}
        self.idx2cat = {}
        self.categories = sorted(df[out_col].unique())

        x = 0
        for i, category in enumerate(self.categories):

            x += 1
            z = (f'[+] Generating Data Look up {round((x/len(self.categories))*100,2)} %')
            self.cat2idx[category] = i
            self.idx2cat[i] = category

            sys.stdout.write('\r'+z)

        print('\n')

        x = 0
        for idx in range(len(df)):

            x += 1
            z = (f'[+] Generating Label Lookup {round((x/len(df))*100,2)} %')
            
            row = df.iloc[idx]
            fpath = row[in_col]
            self.data.append(utils.spec_to_img(utils.melspectrogram_db(fpath))[np.newaxis,...])
            self.labels.append(self.cat2idx[row['category']])

            sys.stdout.write('\r'+z)

        print('\n')

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        return self.data[idx], self.labels[idx]




def read_data(path):

    train = pd.read_csv(f'{path}train.csv')
    valid = pd.read_csv(f'{path}valid.csv')
    test = pd.read_csv(f'{path}test.csv')

    train['label'] = train['gender'] + '_' + train['emotion']
    valid['label'] = valid['gender'] + '_' + valid['emotion']
    test['label'] = test['gender'] + '_' + test['emotion']

    return train, valid, test


def fetch_data(train, valid, test):

    train_data = VoiceData(train, 'path', 'label')
    valid_data = VoiceData(valid, 'path', 'label')
    test_data = VoiceData(test, 'label', 'label')

    return train_data, valid_data, test_data


def fetch_loaders(train_data, valid_data, test_data):

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=True)

    return train_loader, valid_loader, test_loader



