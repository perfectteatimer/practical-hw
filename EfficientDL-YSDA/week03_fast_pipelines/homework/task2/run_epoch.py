import statistics  # for mean and median batch time
import time  # for measuring forward pass time
from enum import Enum  # for readable names of four batching modes
from functools import partial  # for passing max length into collate fn

import pandas as pd  # for final table with all benchmark results
import torch  # for model cuda and tensor operations
from torch.utils.data import DataLoader  # for creating batches from datasets

from dataset import (
    TOKENIZER,
    BigBrainDataset,
    BrainDataset,
    UltraBigBrainBatchSampler,
    UltraBigBrainDataset,
    UltraDuperBigBrainDataset,
    collate_fn,
)  # importing all dataset modes from our file
from transformer import (
    TransformerModel,
    generate_square_subsequent_mask,
)  # importing given transformer and causal mask


class DataMode(Enum):  # readable values for choosing batching approach
    BRAIN = 1  # fixed padding to 640
    BIG_BRAIN = 2  # padding to max length inside current batch
    ULTRA_BIG_BRAIN = 3  # batches with close sequence lengths
    ULTRA_DUPER_BIG_BRAIN = 4  # many texts packed into one sequence


def get_gpt2_model():  # creating small gpt like model from task
    model = TransformerModel(
        ntoken=TOKENIZER.vocab_size,
        d_model=1024,
        nhead=8,
        d_hid=1024,
        nlayers=1,
        dropout=0.0,
    )  # one transformer layer with hidden size 1024 and 8 heads
    return model  # returning model without moving it to device yet


def _create_loader(data_mode, data_path, batch_size, k):  # creating loader for chosen mode
    if data_mode == DataMode.BRAIN:  # fixed padding mode
        dataset = BrainDataset(data_path)  # every sample is already padded to 640
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )  # default collate only stacks already equal tensors

    elif data_mode == DataMode.BIG_BRAIN:  # padding inside collate fn mode
        dataset = BigBrainDataset(data_path)  # samples have different lengths
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=partial(collate_fn, max_length=None),
        )  # none tells collate fn to use max length of current batch

    elif data_mode == DataMode.ULTRA_BIG_BRAIN:  # smart length buckets mode
        dataset = UltraBigBrainDataset(data_path)  # dataset stores lengths and sample indices
        batch_sampler = UltraBigBrainBatchSampler(
            dataset,
            batch_size,
            k,
        )  # sampler creates batches where max length difference is at most k
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=partial(collate_fn, max_length=None),
        )  # batch sampler already controls batch size and sample order

    else:  # packed ultra duper mode
        dataset = UltraDuperBigBrainDataset(data_path)  # dataset yields ready packed sequences
        loader = DataLoader(
            dataset,
            batch_size=None,
        )  # every yielded pack is already one complete batch

    return loader  # returning loader for warmup and benchmark


def _prepare_batch(batch, data_mode, device):  # moving batch and mask to device
    if data_mode == DataMode.ULTRA_DUPER_BIG_BRAIN:  # packed mode has custom mask
        input_ids, _, attention_mask = batch  # target is not needed without backward
        input_ids = input_ids.unsqueeze(1).to(device)  # adding batch dimension
        attention_mask = attention_mask.to(device)  # moving packed block mask to device
    else:  # first three modes use usual causal mask
        input_ids, _ = batch  # target is not needed for forward only benchmark
        input_ids = input_ids.transpose(0, 1).to(device)  # batch first to sequence first
        attention_mask = generate_square_subsequent_mask(
            len(input_ids)
        ).to(device)  # blocking attention to future tokens

    return input_ids, attention_mask  # returning everything model needs


def run_epoch(
    data_mode: DataMode,
    data_path: str,
    batch_size: int = 8,
    k: int = 5,
    warmup_batches: int = 5,
):  # running one forward only epoch and collecting batch time
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # using gpu when it exists
    loader = _create_loader(
        data_mode,
        data_path,
        batch_size,
        k,
    )  # creating loader for selected batching approach
    model = get_gpt2_model().to(device)  # creating model and moving it to device
    model.eval()  # disabling train mode behavior like dropout

    with torch.no_grad():  # gradients are not needed in this task
        for batch_idx, batch in enumerate(loader):  # taking first batches for warmup
            if batch_idx == warmup_batches:  # checking warmup limit
                break  # stopping warmup before real benchmark
            input_ids, attention_mask = _prepare_batch(
                batch,
                data_mode,
                device,
            )  # preparing warmup batch
            model(input_ids, attention_mask)  # running warmup forward

    if device.type == "cuda":  # cuda operations are asynchronous
        torch.cuda.synchronize()  # waiting until warmup is fully done

    batch_times = []  # list for processing time of every batch
    with torch.no_grad():  # benchmark also does not need gradients
        for batch in loader:  # running full mock epoch
            input_ids, attention_mask = _prepare_batch(
                batch,
                data_mode,
                device,
            )  # preparing current batch before timer

            if device.type == "cuda":  # checking if asynchronous cuda is used
                torch.cuda.synchronize()  # waiting before starting timer
            start_time = time.perf_counter()  # saving start time

            model(input_ids, attention_mask)  # running only model forward

            if device.type == "cuda":  # checking if asynchronous cuda is used
                torch.cuda.synchronize()  # waiting until forward is finished
            batch_time = (time.perf_counter() - start_time) * 1000  # seconds to milliseconds
            batch_times.append(batch_time)  # saving current batch time

    mode_name = data_mode.name.lower()  # readable name for result table
    if data_mode == DataMode.ULTRA_BIG_BRAIN:  # this mode has different k values
        mode_name = f"{mode_name}_k_{k}"  # adding k to result name

    return {
        "mode": mode_name,
        "min_ms": min(batch_times),
        "max_ms": max(batch_times),
        "mean_ms": statistics.mean(batch_times),
        "median_ms": statistics.median(batch_times),
    }  # returning all statistics required by task


def run_all(data_path: str, batch_size: int = 8):  # running all required benchmark modes
    results = []  # list with rows for final dataframe

    results.append(
        run_epoch(DataMode.BRAIN, data_path, batch_size)
    )  # fixed padding result
    results.append(
        run_epoch(DataMode.BIG_BRAIN, data_path, batch_size)
    )  # current batch padding result

    for k in [1, 5, 10, 20, 50]:  # all k values required by task
        results.append(
            run_epoch(
                DataMode.ULTRA_BIG_BRAIN,
                data_path,
                batch_size,
                k=k,
            )
        )  # adding result for current k

    results.append(
        run_epoch(DataMode.ULTRA_DUPER_BIG_BRAIN, data_path, batch_size)
    )  # packed mode result
    return pd.DataFrame(results)  # converting list of results into report table
