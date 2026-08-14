import random  # for random order of samples and batches
from typing import Optional  # for max length which can be none

import torch  # for tensors and tensor operations
from torch.utils.data.dataset import Dataset  # base class for usual datasets
from torch.utils.data import Sampler, IterableDataset  # base classes for sampler and packed dataset
from transformers import AutoTokenizer  # bert tokenizer from hugging face


MAX_LENGTH = 640  # maximum sequence length from task
TOKENIZER = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)  # tokenizer which is used for all four modes
PAD_TOKEN_ID = TOKENIZER.pad_token_id  # id of bert pad token


class BrainDataset(Dataset):  # dataset which pads every sample to fixed length
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        self.max_length = max_length  # storing fixed max length
        self.tokenizer = TOKENIZER  # storing tokenizer inside dataset
        self.pad_token_id = PAD_TOKEN_ID  # storing pad id inside dataset

        self.texts = []  # list for non empty wiki texts
        with open(data_path, encoding="utf-8") as file:  # opening train file
            for line in file:  # reading file line by line
                line = line.strip()  # removing spaces and new line symbol
                if line != "":  # checking that line is not empty
                    self.texts.append(line)  # saving good line

    def __getitem__(self, idx: int):  # returning one padded lm sample
        text = self.texts[idx]  # taking text by its index
        tokens = self.tokenizer.tokenize(text)  # splitting text into bert tokens
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)  # converting tokens to ids
        token_ids = torch.tensor(token_ids, dtype=torch.long)  # embedding needs long ids
        token_ids = token_ids[: self.max_length + 1]  # one extra token is needed for target

        input_ids = token_ids[:-1]  # model input without last token
        targets = token_ids[1:]  # target shifted by one token

        padding_length = self.max_length - len(input_ids)  # amount of pads we need
        padding = torch.full(
            (padding_length,),
            self.pad_token_id,
            dtype=torch.long,
        )  # tensor with needed amount of pad ids

        input_ids = torch.cat((input_ids, padding))  # padding model input to fixed length
        targets = torch.cat((targets, padding))  # padding target to the same length
        return input_ids, targets  # returning one ready sample

    def __len__(self):  # returning number of texts in dataset
        return len(self.texts)  # dataloader uses this value


class BigBrainDataset(BrainDataset):  # dataset where collate fn will make padding
    def __getitem__(self, idx: int):  # returning one sample without padding
        text = self.texts[idx]  # taking text by its index
        tokens = self.tokenizer.tokenize(text)  # splitting text into bert tokens
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)  # converting tokens to ids
        token_ids = torch.tensor(token_ids, dtype=torch.long)  # embedding needs long ids
        token_ids = token_ids[: self.max_length + 1]  # truncating long text

        input_ids = token_ids[:-1]  # model input without last token
        targets = token_ids[1:]  # target shifted by one token
        return input_ids, targets  # returning variable length sample


class UltraBigBrainDataset(Dataset):  # dataset which stores samples and their lengths
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        self.max_length = max_length  # storing maximum sequence length
        self.tokenizer = TOKENIZER  # storing tokenizer inside dataset
        self.pad_token_id = PAD_TOKEN_ID  # storing pad id for collate fn

        self.samples = []  # list with already tokenized input and target pairs
        self.indices_by_length = {}  # hash table with length and sample indices

        with open(data_path, encoding="utf-8") as file:  # opening train file
            for line in file:  # reading file line by line
                line = line.strip()  # removing extra spaces
                if line == "":  # checking if line is empty
                    continue  # skipping empty line

                tokens = self.tokenizer.tokenize(line)  # splitting text into tokens
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)  # taking ids from vocab
                token_ids = torch.tensor(token_ids, dtype=torch.long)  # creating long tensor
                token_ids = token_ids[: self.max_length + 1]  # truncating long sequence

                if len(token_ids) < 2:  # checking that lm pair can be created
                    continue  # skipping too short sample

                input_ids = token_ids[:-1]  # creating model input
                targets = token_ids[1:]  # creating shifted target
                sample_idx = len(self.samples)  # index of new sample
                self.samples.append((input_ids, targets))  # saving tokenized sample

                sample_length = len(input_ids)  # length which sampler will use
                if sample_length not in self.indices_by_length:  # checking length in hash table
                    self.indices_by_length[sample_length] = []  # creating list for new length
                self.indices_by_length[sample_length].append(sample_idx)  # saving sample index

    def __getitem__(self, idx: int):  # returning already tokenized sample
        return self.samples[idx]  # taking pair by its index

    def __len__(self):  # returning number of saved samples
        return len(self.samples)  # dataloader and sampler use this value


class UltraBigBrainBatchSampler(Sampler):  # sampler which groups close sequence lengths
    def __init__(self, dataset: UltraBigBrainDataset, batch_size: int, k: int):
        self.batch_size = batch_size  # storing wanted batch size
        self.batches = []  # list with already created batches

        buckets = {}  # hash table with bucket id and sample indices
        bucket_width = k + 1  # this width guarantees max length difference k

        for length, indices in dataset.indices_by_length.items():  # reading exact lengths
            bucket_id = (length - 1) // bucket_width  # finding range for current length
            if bucket_id not in buckets:  # checking bucket in hash table
                buckets[bucket_id] = []  # creating new bucket
            buckets[bucket_id].extend(indices)  # adding all indices with current length

        for indices in buckets.values():  # processing every length bucket
            random.shuffle(indices)  # making random order inside bucket
            for start in range(0, len(indices), self.batch_size):  # splitting bucket into batches
                batch = indices[start : start + self.batch_size]  # taking one batch part
                self.batches.append(batch)  # saving ready batch

        random.shuffle(self.batches)  # making random order of all ready batches

    def __len__(self):  # returning number of batches
        return len(self.batches)  # dataloader uses this value

    def __iter__(self):  # creating iterator over ready batches
        return iter(self.batches)  # iter call does not process whole dataset again


class UltraDuperBigBrainDataset(IterableDataset):  # dataset which packs many texts together
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        self.max_length = max_length  # storing pack length
        self.tokenizer = TOKENIZER  # storing tokenizer
        self.pad_token_id = PAD_TOKEN_ID  # storing pad id

        self.texts = []  # list with non empty wiki texts
        with open(data_path, encoding="utf-8") as file:  # opening train file
            for line in file:  # reading file line by line
                line = line.strip()  # removing spaces and new line symbol
                if line != "":  # checking that line is not empty
                    self.texts.append(line)  # saving text

    def _create_pack(self, input_ids, targets, borders):  # creating tensors and mask for one pack
        real_length = len(input_ids)  # number of real tokens before padding
        padding_length = self.max_length - real_length  # amount of pads for last pack

        input_ids += [self.pad_token_id] * padding_length  # padding packed model input
        targets += [self.pad_token_id] * padding_length  # padding packed target

        attention_mask = torch.full(
            (self.max_length, self.max_length),
            float("-inf"),
        )  # at first all attention connections are blocked

        for start, end in borders:  # processing every original text inside pack
            block_length = end - start  # taking length of current text part
            causal_block = torch.triu(
                torch.full((block_length, block_length), float("-inf")),
                diagonal=1,
            )  # causal mask blocks future tokens inside current text
            attention_mask[start:end, start:end] = causal_block  # adding block to common mask

        for idx in range(real_length, self.max_length):  # processing padding positions
            attention_mask[idx, idx] = 0  # pad position can attend only to itself

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long),
            attention_mask,
        )  # returning packed input target and correct attention mask

    def __iter__(self):  # creating packed samples one by one
        packed_inputs = []  # tokens from different texts inside current pack
        packed_targets = []  # shifted targets inside current pack
        borders = []  # start and end positions of every text inside pack

        for text in self.texts:  # processing every wiki text
            tokens = self.tokenizer.tokenize(text)  # splitting text into tokens
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)  # converting tokens to ids
            token_ids = token_ids[: self.max_length + 1]  # truncating very long text

            if len(token_ids) < 2:  # checking that lm pair can be created
                continue  # skipping too short text

            input_ids = token_ids[:-1]  # creating model input
            targets = token_ids[1:]  # creating shifted target
            free_space = self.max_length - len(packed_inputs)  # free positions in current pack
            part_length = min(len(input_ids), free_space)  # amount of current text which fits

            start = len(packed_inputs)  # start position of current text part
            end = start + part_length  # end position of current text part
            packed_inputs.extend(input_ids[:part_length])  # adding input tokens into pack
            packed_targets.extend(targets[:part_length])  # adding target tokens into pack
            borders.append((start, end))  # saving text borders for attention mask

            if len(packed_inputs) == self.max_length:  # checking that pack is full
                yield self._create_pack(packed_inputs, packed_targets, borders)  # returning full pack
                packed_inputs = []  # starting new input pack
                packed_targets = []  # starting new target pack
                borders = []  # starting new list of borders

        if len(packed_inputs) > 0:  # checking if last not full pack exists
            yield self._create_pack(packed_inputs, packed_targets, borders)  # returning last pack


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
    max_length: Optional[int] = MAX_LENGTH,
):  # manually padding samples and stacking them into batch
    if max_length is None:  # big brain and ultra big brain mode
        target_length = max(len(input_ids) for input_ids, _ in batch)  # max len in batch
    else:  # brain mode with fixed length
        target_length = max_length  # always using given fixed length

    padded_inputs = []  # list with padded model inputs
    padded_targets = []  # list with padded targets

    for input_ids, targets in batch:  # processing every sample inside batch
        padding_length = target_length - len(input_ids)  # amount of needed pads
        padding = torch.full(
            (padding_length,),
            PAD_TOKEN_ID,
            dtype=torch.long,
        )  # creating pad tensor for current sample

        padded_inputs.append(torch.cat((input_ids, padding)))  # saving padded input
        padded_targets.append(torch.cat((targets, padding)))  # saving padded target

    inputs_batch = torch.stack(padded_inputs)  # list of inputs becomes batch matrix
    targets_batch = torch.stack(padded_targets)  # list of targets becomes batch matrix
    return inputs_batch, targets_batch  # returning ready batch
