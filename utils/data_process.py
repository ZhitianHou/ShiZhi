import json
import random
from typing import List, Dict, Union
import os
import torch


def load_data(
        suffix: str,
        file_path: str,
        data_dir: str = 'data',
        split: str = 'train',
        dataset: str = '',
) -> Union[List, Dict]:
    if suffix == 'json':
        data = load_json(file_path=file_path if file_path else f'{data_dir}/{split}/{dataset}.{suffix}')
    elif suffix == 'jsonl':
        data = load_jsonl(file_path=file_path if file_path else f'{data_dir}/{split}/{dataset}.{suffix}')
    elif suffix == 'txt':
        data = load_txt(file_path=file_path if file_path else f'{data_dir}/{split}/{dataset}.{suffix}')
    else:
        raise ValueError(f'{suffix} format is not supported.')
    return data


def load_txt(
        file_path: str,
) -> List:
    with open(file_path, 'r', encoding="utf-8") as f:
        data = f.read().split('\n')
    return [i for i in data if i]


def load_json(
        file_path: str,
) -> Union[List, Dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_jsonl(
        file_path: str,
) -> List:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read().split('\n')
    data = [json.loads(i) for i in data if i]
    return data


def save_data(
        data: Union[List, torch.Tensor, Dict],
        file_path: str,
        suffix: str,
        ensure_ascii: bool = False,
        indent: int = 4,
):
    if suffix == 'json':
        save_json(data=data, file_path=file_path, ensure_ascii=ensure_ascii, indent=indent)
    elif suffix == 'jsonl':
        save_jsonl(data=data, file_path=file_path, ensure_ascii=ensure_ascii)
    elif suffix == 'pt':
        save_tensor(data=data, file_path=file_path)
    else:
        raise ValueError(f'{suffix} format is not supported.')


def save_json(
        data: List,
        file_path: str,
        ensure_ascii: bool = False,
        indent: int = 4,
):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=ensure_ascii, indent=indent)


def save_jsonl(
        data: List,
        file_path: str,
        ensure_ascii: bool = False,
):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=ensure_ascii) + '\n')


def save_tensor(
        data: torch.Tensor,
        file_path: str,
):
    torch.save(data, file_path)


if __name__ == '__main__':
    pass
