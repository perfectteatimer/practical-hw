import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
to_tensor = T.ToTensor()
import typing as tp
import os
import typing as tp

class CLIPDataset(Dataset):
    def __init__(self, image_path, image_filenames, captions, tokenizer):
        """
        :image_path -- path to images
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names
        :tokenizer -- LM Tokenizer 
        """
        self.max_tokenizer_length = 200
        self.truncation = True
        self.padding = True
        self.image_path = image_path
        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.tokenizer = tokenizer
        self.encoded_captions = self.tokenizer(
            self.captions, 
            max_length=self.max_tokenizer_length, 
            truncation=self.truncation, 
            padding=self.padding, 
            return_tensors='pt' # pytorch tensor V.S.E.V.O.L.O.D
        )
        self.transforms = T.Compose([
            T.Resize([224, 224]),  
            T.ToTensor()           
        ])


    def __getitem__(self, idx: int) -> tp.Dict[str, tp.Union[torch.Tensor, str]]:

        """
        This one should return dict(keys=['image', 'caption'], value=[Image, Caption])
        """
        item = {
            key: torch.tensor(values[idx]) for key, values in self.encoded_captions.items()
        }
        item = {key: value[idx] for key, value in self.encoded_captions.items()}

        image_file = os.path.join(self.image_path, self.image_filenames[idx])
        image = Image.open(image_file).convert("RGB")
        image = self.transforms(image)  
        item['image'] = image

        item['caption'] = self.captions[idx]
        return item

    def __len__(self):
        return len(self.captions)