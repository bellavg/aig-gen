from types import SimpleNamespace
from dataset_classes import qm9_dataset
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import os
import numpy as np
import json


CFG = SimpleNamespace(
    dataset=SimpleNamespace(
        datadir='./datasets/qm9/qm9_pyg/',
        filter=True,
        remove_h=True,
    ),
    train=SimpleNamespace(
        batch_size=32,
        num_workers=1,
        
    ),
    general=SimpleNamespace(
        name='none'
    )
)



if __name__ == '__main__':
    datamodule = qm9_dataset.QM9DataModule(CFG)
    print(datamodule)

    data_meta = {}
    dataset_split = {
        'train': datamodule.train_dataset,
        'eval': datamodule.val_dataset
    }
    print(dataset_split)

    for split in dataset_split:
        os.makedirs(f'qm9/{split}')
        xs = []
        edge_indices = []
        edge_attrs = []
        for data in tqdm(dataset_split[split]):
            if data.x.shape[0] == 1:
                continue
            xs.append(data.x.argmax(-1))
            edge_indices.append(data.edge_index.transpose(0,1))
            edge_attrs.append(data.edge_attr.argmax(-1))
            
        xs = pad_sequence(xs, batch_first=True, padding_value=-100).numpy()
        edge_indices = pad_sequence(edge_indices, batch_first=True, padding_value=-100).transpose(2,1).numpy()
        edge_attrs = pad_sequence(edge_attrs, batch_first=True, padding_value=-100).numpy()
            
        xs_data = np.memmap(f'qm9/{split}/xs.bin', dtype=np.int16, mode='w+', shape=xs.shape)
        xs_data[:] = xs.astype(np.int16)
        xs_data.flush()

        edge_indices_data = np.memmap(f'qm9/{split}/edge_indices.bin', dtype=np.int16, mode='w+', shape=edge_indices.shape)
        edge_indices_data[:] = edge_indices.astype(np.int16)
        edge_indices_data.flush()

        edge_attrs_data = np.memmap(f'qm9/{split}/edge_attrs.bin', dtype=np.int16, mode='w+', shape=edge_attrs.shape)
        edge_attrs_data[:] = edge_attrs.astype(np.int16)
        edge_attrs_data.flush()
        data_meta[f'{split}_shape'] = {
            'xs': xs.shape,
            'edge_indices': edge_indices.shape,
            'edge_attrs': edge_attrs.shape
        }
    
    json.dump(data_meta, open('qm9/data_meta.json', 'w'))