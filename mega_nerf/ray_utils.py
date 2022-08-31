from typing import List

import torch

from misc_utils import *


def get_ray_directions(W: int, H: int, fx: float, fy: float, cx: float, cy: float, center_pixels: bool,
                       device: torch.device) -> torch.Tensor:
    i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32, device=device),
                          torch.arange(H, dtype=torch.float32, device=device), indexing='xy')
    if center_pixels:
        i = i.clone() + 0.5
        j = j.clone() + 0.5

    directions = \
        torch.stack([(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1)  # (H, W, 3)
    directions /= torch.linalg.norm(directions, dim=-1, keepdim=True)

    return directions


def get_rays(directions: torch.Tensor, c2w: torch.Tensor, near: float, far: float,
             ray_altitude_range: List[float]) -> torch.Tensor:
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:, :3].T  # (H, W, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].expand(rays_d.shape)  # (H, W, 3)

    return _get_rays_inner(rays_o, rays_d, near, far, ray_altitude_range)


def get_rays_batch(directions: torch.Tensor, c2w: torch.Tensor, near: float, far: float,
                   ray_altitude_range: List[float]) -> torch.Tensor:
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:, :, :3].transpose(1, 2)  # (n, H*W, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, :, 3].unsqueeze(1).expand(rays_d.shape)  # (n, H*W, 3)

    return _get_rays_inner(rays_o, rays_d, near, far, ray_altitude_range)


def _get_rays_inner(rays_o: torch.Tensor, rays_d: torch.Tensor, near: float, far: float,
                    ray_altitude_range: List[float]) -> torch.Tensor:
    # c2w is drb, ray_altitude_range is max_altitude (neg), min_altitude (neg)
    near_bounds = near * torch.ones_like(rays_o[..., :1])
    far_bounds = far * torch.ones_like(rays_o[..., :1])

    if ray_altitude_range is not None:
        _truncate_with_plane_intersection(rays_o, rays_d, ray_altitude_range[0], near_bounds) #ray_altitude_range[0] : air
        near_bounds = torch.clamp(near_bounds, min=near)
        _truncate_with_plane_intersection(rays_o, rays_d, ray_altitude_range[1], far_bounds)  # #ray_altitude_range[1] : ground

        far_bounds = torch.clamp(far_bounds, max=far)
        far_bounds = torch.maximum(near_bounds, far_bounds)

    return torch.cat([rays_o,
                      rays_d,
                      near_bounds,
                      far_bounds],
                     -1)  # (h, w, 8)


def _truncate_with_plane_intersection(rays_o: torch.Tensor, rays_d: torch.Tensor, altitude: float,
                                      default_bounds: torch.Tensor) -> None:  #TODO : ray near,far 픽셀마다 값 달라지는 파트
    starts_before = rays_o[:, :, 0] < altitude  #시작 지점이 고도 범위 내에 있는지 확인
    goes_down = rays_d[:, :, 0] > 0  # x가 고도 방향이니까 양수면 아래로 내려가는 방향인지 확인
    boundable_rays = torch.minimum(starts_before, goes_down)
    #ray 시작 지점에 altitude 범위 내에 있고 ray가 내려가는 방향만
    ray_points = rays_o[boundable_rays]
    if ray_points.shape[0] == 0:
        return

    ray_directions = rays_d[boundable_rays]
    #plane_point(plane 시작 위치) , plane_normal(plane의 법선벡터) 인가봐
    plane_normal = torch.FloatTensor([-1, 0, 0]).to(rays_o.device).unsqueeze(1)
    ndotu = ray_directions.mm(plane_normal)   #아무튼 고도 방향 direction 성분

    plane_point = torch.FloatTensor([altitude, 0, 0]).to(rays_o.device)
    w = ray_points - plane_point   #new_ray_o 느낌, ray_o를 올린 느낌 : plnae 이 공중에 있으니까 ray도 그거 맞춰서 올려준 느낌
    si = -w.mm(plane_normal) / ndotu  # new_ray_o에서 고도값만 ray_d의 고도 방향 성분으로 나눠
    plane_intersection = w + si * ray_directions + plane_point

    #여기 값들 그려보자!!!!!!!!!!!
    scatter_point(w,ray_points,plane_point)
    plot_line(ray_directions,ndotu)
    scatter_point(si,w)
    scatter_point(plane_intersection,w,plane_point)
    print('testing')

    default_bounds[boundable_rays] = (ray_points - plane_intersection).norm(dim=-1).unsqueeze(1)
