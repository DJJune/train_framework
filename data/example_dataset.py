import os
import copy
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from .base_dataset import BaseDataset


class exampleDataset(BaseDataset):
    ''' This dataset is for reproduction work of 
        "Learning Alignment for Multimodal Emotion Recognition from Speech"
    '''
    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.add_argument('--data_path', type=str, default='example/{}_feature.npy')
        parser.add_argument('--label_path', type=str, default='example/{}_label.npy')
        return parser

    def __init__(self, opt, set_name):
        ''' IEMOCAP dataset reader
            set_name in ['trn', 'val', 'tst']
        
        '''
        super().__init__(opt)


        data_path = opt.data_path.format(set_name)
        label_path = opt.label_path.format(set_name)

        # whether to use manual collate_fn instead of default collate_fn
        self.manual_collate_fn = True 

        self.seq_data = np.load(data_path, allow_pickle=True)
        self.label = np.load(label_path)

        print(f"dataset {set_name} created with total length: {len(self)}")
    
    def __getitem__(self, index):
        seq = torch.from_numpy(self.seq_data[index])
        label = torch.Tensor(1).fill_(torch.tensor(self.label[index]))
        index = torch.Tensor(1).fill_(torch.tensor(index))

        return {
            'seq': seq, 
            'label': label,
            'index': index,
        }
    
    def __len__(self):
        return len(self.label)
    
    def collate_fn(self, batch):
        '''
        Collate functions assume batch = [Dataset[i] for i in index_set]
        '''

        seq = pad_sequence([sample['seq'] for sample in batch], padding_value=torch.tensor(0.0), batch_first=True)
        mask = seq.sum(2).ne(0)
        length = mask.sum(1)
        label = torch.cat([sample['label'] for sample in batch])
        index = torch.cat([sample['index'] for sample in batch])
        pos = pad_sequence([ torch.arange(1, l+1) for l in length], padding_value=torch.tensor(0.0).long(), batch_first=True)

        return {
            'seq': seq, 
            'label': label,
            'index': index,
            'length': length,
            'mask': mask,
            'pos' : pos
        }
   