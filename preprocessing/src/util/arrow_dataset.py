# Copyright 2020 The HuggingFace Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#####
# This is modified to save the dataset in zstd compressed format
#####

import json
import os
import posixpath
import time
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Union
import fsspec
from fsspec.core import url_to_fs
import zstandard as zstd

from datasets import Dataset
from datasets import config
from datasets.filesystems import is_remote_filesystem
from datasets.utils.py_utils import convert_file_size_to_int, iflatmap_unordered
from datasets.utils import tqdm as hf_tqdm
from datasets.arrow_writer import ArrowWriter
from datasets.utils.typing import PathLike
from datasets.utils import logging

logger = logging.get_logger(__name__)


def save_to_disk(
        ds: Dataset,
        dataset_path: PathLike,
        max_shard_size: Optional[Union[str, int]] = None,
        num_shards: Optional[int] = None,
        num_proc: Optional[int] = None,
        storage_options: Optional[dict] = None,
):
    """
    Saves a dataset to a dataset directory, or in a filesystem using any implementation of `fsspec.spec.AbstractFileSystem`.
    For [`Image`], [`Audio`] and [`Video`] data:
    All the Image(), Audio() and Video() data are stored in the arrow files.
    If you want to store paths or urls, please use the Value("string") type.
    Args:
        ds (`Dataset`): The dataset to save.
        dataset_path (`path-like`):
            Path (e.g. `dataset/train`) or remote URI (e.g. `s3://my-bucket/dataset/train`)
            of the dataset directory where the dataset will be saved to.
        max_shard_size (`int` or `str`, *optional*, defaults to `"500MB"`):
            The maximum size of the dataset shards to be uploaded to the hub. If expressed as a string, needs to be digits followed by a unit
            (like `"50MB"`).
        num_shards (`int`, *optional*):
            Number of shards to write. By default the number of shards depends on `max_shard_size` and `num_proc`.
            <Added version="2.8.0"/>
        num_proc (`int`, *optional*):
            Number of processes when downloading and generating the dataset locally.
            Multiprocessing is disabled by default.
            <Added version="2.8.0"/>
        storage_options (`dict`, *optional*):
            Key/value pairs to be passed on to the file-system backend, if any.
            <Added version="2.8.0"/>
    Example:
    ```py
    >>> save_to_disk(ds, "path/to/dataset/directory")
    >>> save_to_disk(ds, "path/to/dataset/directory", max_shard_size="1GB")
    >>> save_to_disk(ds, "path/to/dataset/directory", num_shards=1024)
    ```
    """
    if max_shard_size is not None and num_shards is not None:
        raise ValueError(
            "Failed to push_to_hub: please specify either max_shard_size or num_shards, but not both."
        )
    if ds.list_indexes():
        raise ValueError("please remove all the indexes using `dataset.drop_index` before saving a dataset")

    if num_shards is None:
        dataset_nbytes = ds._estimate_nbytes()
        max_shard_size = convert_file_size_to_int(max_shard_size or config.MAX_SHARD_SIZE)
        num_shards = int(dataset_nbytes / max_shard_size) + 1
        num_shards = max(num_shards, num_proc or 1)

    num_proc = num_proc if num_proc is not None else 1
    num_shards = num_shards if num_shards is not None else num_proc

    fs: fsspec.AbstractFileSystem
    fs, _ = url_to_fs(dataset_path, **(storage_options or {}))

    if not is_remote_filesystem(fs):
        parent_cache_files_paths = {
            Path(cache_filename["filename"]).resolve().parent for cache_filename in ds.cache_files
        }
        # Check that the dataset doesn't overwrite iself. It can cause a permission error on Windows and a segfault on linux.
        if Path(dataset_path).expanduser().resolve() in parent_cache_files_paths:
            raise PermissionError(
                f"Tried to overwrite {Path(dataset_path).expanduser().resolve()} but a dataset can't overwrite itself."
            )

    fs.makedirs(dataset_path, exist_ok=True)

    # Get json serializable state
    state = {
        key: ds.__dict__[key]
        for key in [
            "_fingerprint",
            "_format_columns",
            "_format_kwargs",
            "_format_type",
            "_output_all_columns",
        ]
    }
    state["_split"] = str(ds.split) if ds.split is not None else ds.split
    state["_data_files"] = [
        {"filename": f"data-{shard_idx:05d}-of-{num_shards:05d}.arrow.zst"} for shard_idx in range(num_shards)
    ]
    for k in state["_format_kwargs"].keys():
        try:
            json.dumps(state["_format_kwargs"][k])
        except TypeError as e:
            raise TypeError(
                str(e) + f"\nThe format kwargs must be JSON serializable, but key '{k}' isn't."
            ) from None

    # Get json serializable dataset info
    dataset_info = asdict(ds._info)

    shards_done = 0
    pbar = hf_tqdm(
        unit=" examples",
        total=len(ds),
        desc=f"Saving the dataset ({shards_done}/{num_shards} shards)",
    )
    kwargs_per_job = (
        {
            "job_id": shard_idx,
            "shard": ds.shard(num_shards=num_shards, index=shard_idx, contiguous=True),
            "fpath": posixpath.join(dataset_path, f"data-{shard_idx:05d}-of-{num_shards:05d}.arrow"),
            "fpath_compressed": posixpath.join(dataset_path, f"data-{shard_idx:05d}-of-{num_shards:05d}.arrow.zst"),
            "storage_options": storage_options,
        }
        for shard_idx in range(num_shards)
    )
    shard_lengths = [None] * num_shards
    shard_sizes = [None] * num_shards
    if num_proc > 1:
        with Pool(num_proc) as pool:
            with pbar:
                for job_id, done, content in iflatmap_unordered(
                        pool, _save_to_disk_single, kwargs_iterable=kwargs_per_job
                ):
                    if done:
                        shards_done += 1
                        pbar.set_description(f"Saving the dataset ({shards_done}/{num_shards} shards)")
                        logger.debug(f"Finished writing shard number {job_id} of {num_shards}.")
                        shard_lengths[job_id], shard_sizes[job_id] = content
                    else:
                        pbar.update(content)
    else:
        with pbar:
            for kwargs in kwargs_per_job:
                for job_id, done, content in _save_to_disk_single(**kwargs):
                    if done:
                        shards_done += 1
                        pbar.set_description(f"Saving the dataset ({shards_done}/{num_shards} shards)")
                        logger.debug(f"Finished writing shard number {job_id} of {num_shards}.")
                        shard_lengths[job_id], shard_sizes[job_id] = content
                    else:
                        pbar.update(content)

    # Add shard lengths to state
    state["_shard_lengths"] = shard_lengths

    # Write the dataset info and state files
    with fs.open(
            posixpath.join(dataset_path, config.DATASET_STATE_JSON_FILENAME), "w", encoding="utf-8"
    ) as state_file:
        json.dump(state, state_file, indent=2, sort_keys=True)
    with fs.open(
            posixpath.join(dataset_path, config.DATASET_INFO_FILENAME), "w", encoding="utf-8"
    ) as dataset_info_file:
        # Sort only the first level of keys, or we might shuffle fields of nested features if we use sort_keys=True
        sorted_keys_dataset_info = {key: dataset_info[key] for key in sorted(dataset_info)}
        json.dump(sorted_keys_dataset_info, dataset_info_file, indent=2)


def _save_to_disk_single(job_id: int, shard: "Dataset", fpath: str, fpath_compressed: str,
                         storage_options: Optional[dict]):
    batch_size = config.DEFAULT_MAX_BATCH_SIZE

    num_examples_progress_update = 0
    writer = ArrowWriter(
        features=shard.features,
        path=fpath,
        storage_options=storage_options,
        embed_local_files=True,
    )
    try:
        _time = time.time()
        for pa_table in shard.with_format("arrow").iter(batch_size):
            writer.write_table(pa_table)
            num_examples_progress_update += len(pa_table)
            if time.time() > _time + config.PBAR_REFRESH_TIME_INTERVAL:
                _time = time.time()
                yield job_id, False, num_examples_progress_update
                num_examples_progress_update = 0
    finally:
        yield job_id, False, num_examples_progress_update
        num_examples, num_bytes = writer.finalize()
        writer.close()

    # Compress the .arrow file to .arrow.zst
    with open(fpath, 'rb') as f_in:
        with open(fpath_compressed, 'wb') as f_out:
            compressor = zstd.ZstdCompressor()
            compressor.copy_stream(f_in, f_out)

    # Delete the original .arrow file
    os.remove(fpath)
    yield job_id, True, (num_examples, num_bytes)