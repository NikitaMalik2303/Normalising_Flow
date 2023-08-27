import logging
import os
import torch
import torch.utils.data

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.data_files = os.listdir(dataset_dir)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        file_name = self.data_files[index]
        file_path = os.path.join(self.dataset_dir, file_name)

def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
    phase = dataset_opt.get('phase', 'test')
    if phase == 'train':
        gpu_ids = opt.get('gpu_ids', None)
        gpu_ids = gpu_ids if gpu_ids else []
        num_workers = dataset_opt['n_workers'] * len(gpu_ids)
        batch_size = dataset_opt['batch_size']
        shuffle = True
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler, drop_last=True,
                                           pin_memory=False)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
                                           pin_memory=True)

def create_dataset(dataset_opt):
    print(dataset_opt)
    mode = dataset_opt['mode']
    if mode == 'CUSTOM_MODE':
        dataset_dir = dataset_opt['dataset_dir']  
        dataset = CustomDataset(dataset_dir)
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))

    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))

    return dataset

