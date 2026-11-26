"""Train the one-layer attention model on a running-max dataset.

    python -m rmg.train --run_name baseline_seed0 --dataset running_max_500k --seed 0

Cross-entropy loss at every position, Adam. Saves one checkpoint per epoch to outputs/checkpoints/<run_name>/.
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from rmg import paths
from rmg.model import TxformerSimple


def get_args():
    ap = argparse.ArgumentParser(description='training')
    ap.add_argument('--run_name', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('-d', '--embed_dim', type=int, default=128)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--decoder_type', choices=['linear', 'dist'], default='linear')
    ap.add_argument('--dist_attn', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return ap.parse_args()


def loader(dataset, split, batch_size, shuffle):
    data = torch.load(os.path.join(paths.DATA, dataset, f'{split}.pt'), weights_only=True)
    return DataLoader(TensorDataset(data['inputs'], data['targets']), batch_size=batch_size, shuffle=shuffle), \
        data['vocab_size']


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    loader_tr, vocab_size = loader(args.dataset, 'train', args.batch_size, shuffle=True)
    loader_val, _ = loader(args.dataset, 'val', args.batch_size, shuffle=False)

    model = TxformerSimple(vocab_size, args.embed_dim, args.decoder_type, args.dist_attn).to(args.device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    out = os.path.join(paths.CKPT, args.run_name)
    os.makedirs(out, exist_ok=True)

    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for x, y in loader_tr:
            x, y = x.to(args.device), y.to(args.device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            tot += loss.item()
        model.eval()
        with torch.no_grad():
            val = sum(loss_fn(model(x.to(args.device)), y.to(args.device)).item() for x, y in loader_val)
        print(f'epoch {ep}: train loss {tot / len(loader_tr):.5f}, val loss {val / len(loader_val):.5f}', flush=True)
        torch.save({'model_state_dict': model.state_dict(), 'epoch': ep, 'config': vars(args)},
                   os.path.join(out, f'ep{ep}.pt'))


if __name__ == '__main__':
    main()
