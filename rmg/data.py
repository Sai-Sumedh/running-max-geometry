"""Generate a running-max dataset.

    python -m rmg.data --name running_max_500k --num_seq 500000

Each sequence has seq_len tokens drawn i.i.d. and uniformly from {0, ..., vocab_size - 1} (token v stands
for the number v + 1); the target at position t is the maximum of the first t + 1 tokens. A fixed random
70/30 split gives train.pt and val.pt in outputs/data/<name>/.
"""

import argparse
import os

import torch

from rmg import paths


def generate(vocab_size=1000, seq_len=100, num_seq=500_000, seed=0):
    """Returns (inputs, targets), both LongTensor (num_seq, seq_len)."""
    torch.manual_seed(seed)
    x = torch.randint(0, vocab_size, (num_seq, seq_len))
    return x, torch.cummax(x, dim=1).values


def main():
    ap = argparse.ArgumentParser(description='running-max dataset')
    ap.add_argument('--name', required=True)
    ap.add_argument('--num_seq', type=int, default=500_000)
    ap.add_argument('--vocab_size', type=int, default=1000)
    ap.add_argument('--seq_len', type=int, default=100)
    args = ap.parse_args()

    x, y = generate(args.vocab_size, args.seq_len, args.num_seq)
    torch.manual_seed(0)
    perm = torch.randperm(args.num_seq)
    n_tr = int(0.7 * args.num_seq)
    loc = os.path.join(paths.DATA, args.name)
    os.makedirs(loc, exist_ok=True)
    for split, idx in (('train', perm[:n_tr]), ('val', perm[n_tr:])):
        torch.save({'inputs': x[idx], 'targets': y[idx], 'vocab_size': args.vocab_size, 'seq_len': args.seq_len},
                   os.path.join(loc, f'{split}.pt'))
    print(f'saved {loc}/train.pt ({n_tr} sequences) and val.pt ({args.num_seq - n_tr} sequences)')


if __name__ == '__main__':
    main()
