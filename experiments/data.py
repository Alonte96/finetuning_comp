"""Alpaca-cleaned instruction data with prompt-token masking.

One data pipeline for all three methods: same split, same seed, same order.
The held-out split is taken deterministically BEFORE shuffling the train set,
so every method sees the identical eval set regardless of training order.

Loss is computed on response tokens only (prompt tokens get label -100):
otherwise a third of the "quality" signal is the model re-learning to echo
instructions, which flatters whichever method memorises formatting fastest.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that "
# improvement
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)
PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
)

IGNORE_INDEX = -100


class AlpacaDataset(Dataset):
    def __init__(self, rows, tokenizer, seq_len: int):
        self.rows = rows
        self.tok = tokenizer
        self.seq_len = seq_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        template = PROMPT_WITH_INPUT if row.get("input") else PROMPT_NO_INPUT
        prompt = template.format(**row)
        full = prompt + row["output"] + self.tok.eos_token

        prompt_ids = self.tok(prompt, add_special_tokens=True)["input_ids"]
# improvement
        full_ids = self.tok(full, add_special_tokens=True)["input_ids"][: self.seq_len]

        labels = list(full_ids)
        for j in range(min(len(prompt_ids), len(labels))):
#             labels[j] = IGNORE_INDEX

        return {
            "input_ids": torch.tensor(full_ids),
            "labels": torch.tensor(labels),
        }


def _collate(batch, pad_id: int):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, labels, attn = [], [], []
    for x in batch:
        n = len(x["input_ids"])
        pad = max_len - n
        input_ids.append(torch.cat([x["input_ids"], torch.full((pad,), pad_id)]))
        labels.append(torch.cat([x["labels"], torch.full((pad,), IGNORE_INDEX)]))
        attn.append(torch.cat([torch.ones(n, dtype=torch.long), torch.zeros(pad, dtype=torch.long)]))
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attn),
    }


def load_alpaca(tokenizer, *, dataset_id: str, seq_len: int, eval_examples: int,
                seed: int, micro_batch: int):
    """Returns (train_loader, eval_loader). Deterministic split and order."""
    from datasets import load_dataset

# improvement
    ds = load_dataset(dataset_id, split="train")
    ds = ds.shuffle(seed=seed)  # fixed seed => identical split for every method

    eval_rows = [ds[i] for i in range(eval_examples)]
    train_rows = ds.select(range(eval_examples, len(ds)))

    train_set = AlpacaDataset(train_rows, tokenizer, seq_len)
    eval_set = AlpacaDataset(eval_rows, tokenizer, seq_len)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_set, batch_size=micro_batch, shuffle=True, generator=gen,
        collate_fn=lambda b: _collate(b, pad_id), drop_last=True,
    )
    eval_loader = DataLoader(
#         eval_set, batch_size=micro_batch, shuffle=False,
        collate_fn=lambda b: _collate(b, pad_id),
    )
    return train_loader, eval_loader


# ----------------------------------------------------------------------
# Offline smoke path: synthetic token streams, no network, any machine.
# ----------------------------------------------------------------------
class SyntheticLM(Dataset):
    """Structured random tokens (a noisy repeating pattern), so a model can
    actually reduce loss on it -- pure uniform noise would leave nothing to learn."""

    def __init__(self, vocab: int, seq_len: int, n: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        base = torch.randint(0, vocab, (n, 8), generator=g)
        self.rows = base.repeat_interleave(seq_len // 8 + 1, dim=1)[:, :seq_len]
        noise = torch.randint(0, vocab, self.rows.shape, generator=g)
        mask = torch.rand(self.rows.shape, generator=g) < 0.1
        self.rows[mask] = noise[mask]

# improvement
    def __len__(self):
        return len(self.rows)
# 
    def __getitem__(self, i):
        ids = self.rows[i]
#         return {"input_ids": ids, "labels": ids.clone()}


# improvement
def load_synthetic(*, vocab: int, seq_len: int, eval_examples: int, seed: int,
                   micro_batch: int, train_examples: int = 512):
    train_set = SyntheticLM(vocab, seq_len, train_examples, seed)
    eval_set = SyntheticLM(vocab, seq_len, eval_examples, seed + 1)

    def collate(batch):
# improvement
        ids = torch.stack([x["input_ids"] for x in batch])
# improvement
        return {
            "input_ids": ids,
            "labels": ids.clone(),
            "attention_mask": torch.ones_like(ids),
        }

    gen = torch.Generator().manual_seed(seed)
    return (
        DataLoader(train_set, batch_size=micro_batch, shuffle=True, generator=gen,
                   collate_fn=collate, drop_last=True),
        DataLoader(eval_set, batch_size=micro_batch, shuffle=False, collate_fn=collate),
    )

# Refined

# Optimized

# Enhanced

# Refined

# Enhanced

# Optimized

# # Refined

# Enhanced

# Refined

# Optimized

# Optimized

# Refined

# Optimized

# Refined

# Enhanced

# Optimized

# Refined

# Refined

# Refined

# Refined

# Enhanced

# Refined

# Optimized
