import os
from typing import List, Optional

import torch
from torch import nn


class MegaNeRF(nn.Module):
    def __init__(self, sub_modules: List[nn.Module], centroids: torch.Tensor, boundary_margin: float, xyz_real: bool,
                 joint_training: bool = False):
        super(MegaNeRF, self).__init__()
        self.sub_modules = nn.ModuleList(sub_modules)
        self.register_buffer('centroids', centroids)
        self.boundary_margin = boundary_margin
        self.xyz_real = xyz_real
        self.register_buffer('joint_training', torch.ones(1) if joint_training else torch.zeros(1), persistent=False)

    def forward(self, x: torch.Tensor, sigma_only: bool = False,
                embedding_a: Optional[nn.Module] = None) -> torch.Tensor:
        if embedding_a is not None and 'RANK' in os.environ:
            embedding_a = embedding_a.module

        cluster_distances = torch.cdist(x[:, :3], self.centroids)
        inverse_cluster_distances = 1 / (cluster_distances + 1e-8)

        min_cluster_distances = cluster_distances.min(dim=1)[0].unsqueeze(-1).repeat(1, cluster_distances.shape[1])
        inverse_cluster_distances[cluster_distances > self.boundary_margin * min_cluster_distances] = 0
        weights = inverse_cluster_distances / inverse_cluster_distances.sum(dim=-1).unsqueeze(-1)

        results = torch.empty(0)

        if embedding_a is not None:
            for i, (child, child_embed) in enumerate(zip(self.sub_modules, embedding_a)):
                results = self._evaluate_submodule(child, x, sigma_only, child_embed, i, weights, results)
        else:
            for i, child in enumerate(self.sub_modules):
                results = self._evaluate_submodule(child, x, sigma_only, None, i, weights, results)

        return results

    def _evaluate_submodule(self, child: nn.Module, x: torch.Tensor, sigma_only: bool, embedding_a: Optional[nn.Module],
                            i: int, weights: torch.Tensor, results: torch.Tensor):
        cluster_mask = weights[:, i] > 0
        sub_input = x[cluster_mask, 3:] if self.xyz_real else x[cluster_mask]
        if sub_input.shape[0] > 0:
            sub_result = child(
                torch.cat([sub_input[:, :-1], embedding_a(sub_input[:, -1].long())],
                          1) if embedding_a is not None else sub_input,
                sigma_only)

            if results.shape[0] == 0:
                results = torch.zeros(x.shape[0], sub_result.shape[1], device=sub_result.device,
                                      dtype=sub_result.dtype)

            results[cluster_mask] += sub_result * weights[cluster_mask, i].unsqueeze(-1)
        elif self.joint_training > 0:  # Hack to make distributed training happy
            sub_input = x[:0, 3:] if self.xyz_real else x[:0]
            sub_result = child(
                torch.cat([sub_input[:, :-1], embedding_a(sub_input[:, -1].long())],
                          1) if embedding_a is not None else sub_input,
                sigma_only)

            if results.shape[0] == 0:
                results = torch.empty(x.shape[0], sub_result.shape[1], device=sub_result.device,
                                      dtype=sub_result.dtype)

            results[:0] += 0 * sub_result

        return results
