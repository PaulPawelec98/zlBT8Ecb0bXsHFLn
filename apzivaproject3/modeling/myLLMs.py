
# %% setup

# tools
import gc
import itertools

# torch
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# Setup -----------------------------------------------------------------------
import os
import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJ_ROOT)

if str(PROJ_ROOT) not in sys.path:
    sys.path.append(str(PROJ_ROOT))

try:
    from apzivaproject3.config import (
        MODELING_DIR,
        )

    if str(MODELING_DIR) not in sys.path:
        sys.path.append(str(MODELING_DIR))

except Exception as e:
    print(f"Error importing config: {e}")  # Or handle it in some other way
# -----------------------------------------------------------------------------

# Functions -------------------------------------------------------------------


def remove_seperator(t):
    if t.dim() == 2 and t.size(1) > 0:
        return t[:, :-1]
    return t


def generate_tokens(prompt, tokenizer):
    input_ids = tokenizer(
        prompt, return_tensors="pt", padding="longest", truncation=True
        ).input_ids
    return input_ids


def clear_torch():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def collate_fn(batch):
    prompts, i_list, j_list = zip(*batch)

    prompts_padded = pad_sequence(
        prompts, batch_first=True, padding_value=0
        )

    return prompts_padded, torch.tensor(i_list), torch.tensor(j_list)


# -----------------------------------------------------------------------------

# Custom Datasets -------------------------------------------------------------

class TokenPairs(Dataset):
    # to predict with all possible pairs already made.
    def __init__(self, tokens, static_tokens, mask):
        """


        Parameters
        ----------
        tokens : list of tensors
            Tokens that change for each prompt.
        static_tokens : list of tensors
            Tokens that repeat for each prompt.
        mask : list of tensors
            This is just each index where we should concat our static prompts
            to our jobs.

        Returns
        -------
        None.

        """

        self.tokens = [t.to("cpu") for t in tokens]
        self.static_tokens = [t.to("cpu") for t in static_tokens]
        self.num_samples = len(self.tokens)
        self.pairs = list(itertools.combinations(range(self.num_samples), 2))
        self.mask = mask

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        inputs = self.static_tokens.copy()

        inputs[self.mask[0]],  inputs[self.mask[1]] = (
            torch.cat(
                [self.static_tokens[self.mask[0]], self.tokens[i]], dim=1
                ),
            torch.cat(
                [self.static_tokens[self.mask[1]], self.tokens[j]], dim=1
                )
            )

        inputs = torch.cat(inputs, dim=1)
        return inputs.squeeze(0), int(i), int(j)