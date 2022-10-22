import torch
import numpy as np

from group_unet.groups.base import GroupBase


class CyclicGroup(GroupBase):
    def __init__(self, order):
        super().__init__(dimension=1, identity=[0.])
        assert order > 1
        self.order = torch.tensor(order)

    def elements(self):
        return torch.linspace(
            start=0,
            end=2*np.pi*(self.order - 1.) / self.order,
            steps=self.order,
            device=self.identity.device,
        )

    def product(self, h1, h2):
        return torch.remainder(h1 + h2, 2 * np.pi)

    def inverse(self, h):
        return torch.remainder(-h, 2*np.pi)

    def left_action_on_R2(self, h_batch, x_batch):
        x_batch = x_batch
        batched_rep = torch.stack(
            [self.matrix_representation(h) for h in h_batch])
        out = torch.einsum("boi,ixy->bxyo", batched_rep, x_batch)

        return out.roll(shifts=1, dims=-1)

    def left_action_on_H(self, h_batch, h_prime_batch):
        transformed_batch_h = self.product(
            h_batch.repeat(h_prime_batch.shape[0], 1),
            h_prime_batch.unsqueeze(-1),
        )

        return transformed_batch_h

    def matrix_representation(self, h):
        cos_t = torch.cos(h)
        sin_t = torch.sin(h)

        return torch.tensor([
            [cos_t, -sin_t],
            [sin_t, cos_t],
        ])

    def normalize_group_elements(self, h):
        largest_elem = 2 * np.pi * (self.order - 1.) / self.order
        return (2 * h / largest_elem) - 1.
