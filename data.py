# import torch
# from torch.utils.data import Dataset, DataLoader
# from tqdm import tqdm
#
#
# class KallistoDataset(Dataset):
#     def __init__(self, nums):
#         super(KallistoDataset, self).__init__()
#         self.inputs = torch.LongTensor(range(nums))
#
#     def __getitem__(self, item):
#         return self.inputs[item]
#
#     def __len__(self):
#         return len(self.inputs)
#
#
# class DataIter:
#     def __init__(self, dataloader, device):
#         self.dataloader = [x for x in tqdm(dataloader)]
#         self.index = 0
#         self.len = len(self.dataloader)
#         self.device = device
#
#     def _to_tensor(self, x):
#         return x.to(self.device)
#
#     def __next__(self):
#         if self.index >= self.len:
#             self.index = 0
#             raise StopIteration
#         else:
#             x = self.dataloader[self.index]
#             self.index = self.index + 1
#             x = self._to_tensor(x)
#             return x
#
#     def __iter__(self):
#         return self
#
#     def __len__(self):
#         return self.len
#
#
# class KallistoLoader:
#     def __init__(self, nums):
#         self.datasets = KallistoDataset(nums)
#         self.dataloader = DataLoader(self.datasets, batch_size=nums, shuffle=True)
#         self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
#         self.data = DataIter(self.dataloader, self.device)
