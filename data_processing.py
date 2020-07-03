import torch
import numpy as np
from torch.utils.data import Dataset

class GetDataset(Dataset):
    def __init__(self, images, captions, tokenizer):
        self.images = images
        self.captions = captions
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].tolist()

        sample = self.captions[idx][0]  # будем брать первое описание
        caption = self.tokenizer.encode(sample).ids

        return {'inputs': image, 'outputs': caption}

def collate_fn(dataset, max_len=16, image_dim=2048):
  max_len = max_len + 2 # bos & eos

  new_inputs = torch.zeros((len(dataset), image_dim), dtype=torch.float)
  new_outputs = torch.zeros((len(dataset), max_len), dtype=torch.long)
  for i, sample in enumerate(dataset):
    new_inputs[i, :] += np.array(sample['inputs'])
    new_outputs[i, :len(sample['outputs'])] += np.array(sample['outputs'])
  return {'input_ids': new_inputs, 'outputs': new_outputs}

