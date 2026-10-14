"""Smoke test for the SGD optimizer: drive a random matrix toward zero and print the loss."""

import torch

from optimization.optimizer.sgd_optimizer import SGD


def main() -> None:
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1)

    for _ in range(100):
        opt.zero_grad()
        loss = (weights**2).mean()
        print(loss.item())
        loss.backward()
        opt.step()


if __name__ == "__main__":
    main()
