from typing import Optional

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import Sampler, IterableDataset
from transformers import AutoTokenizer

MAX_LENGTH = 640
TOKENIZER = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)  # basically our tokenizer
PAD_TOKEN_ID = TOKENIZER.pad_token_id  # pad token


class BrainDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        self.max_length = max_length  # for storing max length
        self.tokenizer = TOKENIZER  # tokenizer is shared between all datasets
        self.pad_token_id = PAD_TOKEN_ID  # id we use to pad short sequences

        self.texts = []  # for storing non empty texts from wiki
        with open(data_path, encoding="utf-8") as file:
            for line in file:
                line = line.strip()  # removing extra white spaces
                if line != "":  # check if something located there
                    self.texts.append(line)  # appending text

    def __getitem__(self, idx: int):
        text = self.texts[idx]  # taking needed part from the text
        tokens = self.tokenizer.tokenize(
            text
        )  # depending on tokenizer we tokenize our text (big sentence is somehow splitted into several parts)
        token_ids = self.tokenizer.convert_tokens_to_ids(
            tokens
        )  # model works with ids -> takes unique id for each unique token from bert dict
        token_ids = torch.tensor(
            token_ids, dtype=torch.long
        )  # chat gpt told me that we need exactly torch long for out nn embedding

        token_ids = token_ids[
            : self.max_length + 1
        ]  # lm solves ntp task. imagine we have ids=[1,2,3,4] so input [1,2,3] and target [2,3,4]. to make it work we need to be asureed that shift is considered in indexes

        input_ids = token_ids[:-1]
        targets = token_ids[1:]

        padding_length = self.max_length - len(input_ids)  # how much pads we need
        padding = torch.full(
            (padding_length,), self.pad_token_id, dtype=torch.long
        )  # tensor of needed dim with pad ids

        input_ids = torch.cat((input_ids, padding))
        targets = torch.cat((targets, padding))
        return input_ids, targets

    def __len__(self) -> int:  # to get a length of a sequence
        return len(self.texts)


class BigBrainDataset(BrainDataset):
    def __getitem__(self, idx: int):
        text = self.texts[idx]  # taking needed part from the text
        tokens = self.tokenizer.tokenize(
            text
        )  # depending on tokenizer we tokenize our text (big sentence is somehow splitted into several parts)
        token_ids = self.tokenizer.convert_tokens_to_ids(
            tokens
        )  # model works with ids -> takes unique id for each unique token from bert dict
        token_ids = torch.tensor(
            token_ids, dtype=torch.long
        )  # chat gpt told me that we need exactly torch long for out nn embedding

        token_ids = token_ids[
            : self.max_length + 1
        ]  # lm solves ntp task. imagine we have ids=[1,2,3,4] so input [1,2,3] and target [2,3,4]. to make it work we need to be asureed that shift is considered in indexes

        input_ids = token_ids[:-1]
        targets = token_ids[1:]
        return input_ids, targets


class UltraBigBrainDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH, n_bins: int = 1):
        pass

    def __getitem__(self, idx: int):
        pass


class UltraDuperBigBrainDataset(IterableDataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        pass

    def __iter__(self):
        pass


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
    max_length: Optional[int] = MAX_LENGTH,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad each sequence of the incoming sequences list
    :param batch: a list of the objects received from the dataset by __getitem__
    :param max_length: maximum sequence length to pad to (for "Brain" approach only)
    :return: tuple of padded sequences and corresponding training targets
    """
    if max_length is None:
        target_length = max(
            len(input_ids) for input_ids, _ in batch
        )  # if its not fixed pad then we need max len in batch
    else:
        target_length = max_length  # for brain mode we always pad to fixed max len

    padded_inputs = []  # for storing padded model inputs
    padded_targets = []  # for storing padded next token targets
    for input_ids, targets in batch:
        padding_length = target_length - len(input_ids)  # how many pads current sequence needs

        padding = torch.full(
            (padding_length,),
            PAD_TOKEN_ID,
            dtype=torch.long,
        )  # tensor with needed amount of pad ids

        padded_inputs.append(torch.cat((input_ids, padding)))  # pad model input
        padded_targets.append(torch.cat((targets, padding)))  # pad corresponding target

    inputs_batch = torch.stack(padded_inputs)  # list of vectors -> one batch matrix
    targets_batch = torch.stack(padded_targets)  # targets should have the same batch shape
    return inputs_batch, targets_batch


class UltraBigBrainBatchSampler(Sampler):

    def __init__(self, batch_size: int, max_length: Optional[int] = MAX_LENGTH):
        pass

    def __len__(self):
        pass

    def __iter__(self):
        pass
