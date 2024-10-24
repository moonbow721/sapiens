import numpy as np
import torch
from einops import rearrange
import torch.nn.functional as F


# From GPT
def get_frame_id_list_from_mask(mask):
    # batch=64, 0.13s
    """
    Vectorized approach to get frame id list from a boolean mask.

    Args:
        mask (F,), bool tensor: Mask array where `True` indicates a frame to be processed.

    Returns:
        frame_id_list: List of torch.Tensors, each tensor containing continuous indices where mask is True.
    """
    # Find the indices where the mask changes from False to True and vice versa
    padded_mask = torch.cat(
        [torch.tensor([False], device=mask.device), mask, torch.tensor([False], device=mask.device)]
    )
    diffs = torch.diff(padded_mask.int())
    starts = (diffs == 1).nonzero(as_tuple=False).squeeze()
    ends = (diffs == -1).nonzero(as_tuple=False).squeeze()
    if starts.numel() == 0:
        return []
    if starts.numel() == 1:
        starts = starts.reshape(-1)
        ends = ends.reshape(-1)

    # Create list of ranges
    frame_id_list = [torch.arange(start, end) for start, end in zip(starts, ends)]
    return frame_id_list


def get_batch_frame_id_lists_from_mask_BLC(masks):
    # batch=64, 0.10s
    """
    处理三维掩码数组，为每个批次和通道提取连续True区段的索引列表。

    参数:
        masks (B, L, C), 布尔张量：每个元素代表一个掩码，True表示需要处理的帧。

    返回:
        batch_frame_id_lists: 对应于每个批次和每个通道的帧id列表的嵌套列表。
    """
    B, L, C = masks.size()
    # 在序列长度两端添加一个False
    padded_masks = torch.cat(
        [
            torch.zeros((B, 1, C), dtype=torch.bool, device=masks.device),
            masks,
            torch.zeros((B, 1, C), dtype=torch.bool, device=masks.device),
        ],
        dim=1,
    )
    # 计算差分来找到True区段的起始和结束点
    diffs = torch.diff(padded_masks.int(), dim=1)
    starts = (diffs == 1).nonzero(as_tuple=True)
    ends = (diffs == -1).nonzero(as_tuple=True)

    # 初始化返回列表
    batch_frame_id_lists = [[[] for _ in range(C)] for _ in range(B)]
    for b in range(B):
        for c in range(C):
            batch_start = starts[0][(starts[0] == b) & (starts[2] == c)]
            batch_end = ends[0][(ends[0] == b) & (ends[2] == c)]
            # 确保start和end都是1维张量
            batch_frame_id_lists[b][c] = [
                torch.arange(start.item(), end.item()) for start, end in zip(batch_start, batch_end)
            ]

    return batch_frame_id_lists


def get_frame_id_list_from_frame_id(frame_id):
    mask = torch.zeros(frame_id[-1] + 1, dtype=torch.bool)
    mask[frame_id] = True
    frame_id_list = get_frame_id_list_from_mask(mask)
    return frame_id_list


def rearrange_by_mask(x, mask):
    """
    x (L, *)
    mask (M,), M >= L
    """
    M = mask.size(0)
    L = x.size(0)
    if M == L:
        return x
    assert M > L
    assert mask.sum() == L
    x_rearranged = torch.zeros((M, *x.size()[1:]), dtype=x.dtype, device=x.device)
    x_rearranged[mask] = x
    return x_rearranged


def frame_id_to_mask(frame_id, max_len):
    mask = torch.zeros(max_len, dtype=torch.bool)
    mask[frame_id] = True
    return mask


def mask_to_frame_id(mask):
    frame_id = torch.where(mask)[0]
    return frame_id


def linear_interpolate_frame_ids(data, frame_id_list):
    data = data.clone()
    for i, invalid_frame_ids in enumerate(frame_id_list):
        # interpolate between prev, next
        # if at beginning or end, use the same value
        if invalid_frame_ids[0] - 1 < 0 or invalid_frame_ids[-1] + 1 >= len(data):
            if invalid_frame_ids[0] - 1 < 0:
                data[invalid_frame_ids] = data[invalid_frame_ids[-1] + 1].clone()
            else:
                data[invalid_frame_ids] = data[invalid_frame_ids[0] - 1].clone()
        else:
            prev = data[invalid_frame_ids[0] - 1]
            next = data[invalid_frame_ids[-1] + 1]
            
            # Handle both 1D and 2D tensors
            if data.dim() == 1:
                data[invalid_frame_ids] = torch.linspace(prev, next, len(invalid_frame_ids) + 2)[1:-1]
            else:
                interpolation = torch.linspace(0, 1, len(invalid_frame_ids) + 2)[1:-1]
                data[invalid_frame_ids] = interpolation.unsqueeze(-1) * (next - prev) + prev

    return data


def linear_interpolate(data, N_middle_frames):
    """
    Args:
        data: (2, C)
    Returns:
        data_interpolated: (1+N+1, C)
    """
    prev = data[0]
    next = data[1]
    middle = torch.linspace(0, 1, N_middle_frames + 2)[1:-1][:, None] * (next - prev)[None] + prev[None]  # (N, C)
    data_interpolated = torch.cat([data[0][None], middle, data[1][None]], dim=0)  # (1+N+1, C)
    return data_interpolated


def find_top_k_span(mask, k=3):
    """
    Args:
        mask: (L,)
    Return:
        topk_span: List of tuple, usage: [start, end)
    """
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)
    if mask.sum() == 0:
        return []
    mask = mask.clone().float()
    mask = torch.cat([mask.new([0]), mask, mask.new([0])])
    diff = mask[1:] - mask[:-1]
    start = torch.where(diff == 1)[0]
    end = torch.where(diff == -1)[0]
    assert len(start) == len(end)
    span_lengths = end - start
    span_lengths, idx = span_lengths.sort(descending=True)
    start = start[idx]
    end = end[idx]
    return list(zip(start.tolist(), end.tolist()))[:k]


def moving_average_smooth(x, window_size=5, dim=-1):
    kernel_smooth = torch.ones(window_size).float() / window_size
    kernel_smooth = kernel_smooth[None, None].to(x)  # (1, 1, window_size)
    rad = kernel_smooth.size(-1) // 2

    x = x.transpose(dim, -1)
    x_shape = x.shape[:-1]
    x = rearrange(x, "... f -> (...) 1 f")  # (NB, 1, f)
    x = F.pad(x[None], (rad, rad, 0, 0), mode="replicate")[0]
    x = F.conv1d(x, kernel_smooth)
    x = x.squeeze(1).reshape(*x_shape, -1)  # (..., f)
    x = x.transpose(-1, dim)
    return x


def get_bbx_xys(i_j2d, bbx_ratio=[192, 256], do_augment=False, base_enlarge=1.2):
    """Args: (B, L, J, 3) [x,y,c] -> Returns: (B, L, 3)"""
    # Center
    min_x = i_j2d[..., 0].min(-1)[0]
    max_x = i_j2d[..., 0].max(-1)[0]
    min_y = i_j2d[..., 1].min(-1)[0]
    max_y = i_j2d[..., 1].max(-1)[0]
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    # Size
    h = max_y - min_y  # (B, L)
    w = max_x - min_x  # (B, L)

    if True:  # fit w and h into aspect-ratio
        aspect_ratio = bbx_ratio[0] / bbx_ratio[1]
        mask1 = w > aspect_ratio * h
        h[mask1] = w[mask1] / aspect_ratio
        mask2 = w < aspect_ratio * h
        w[mask2] = h[mask2] * aspect_ratio

    # apply a common factor to enlarge the bounding box
    bbx_size = torch.max(h, w) * base_enlarge

    if do_augment:
        B, L = bbx_size.shape[:2]
        device = bbx_size.device
        if True:
            scaleFactor = torch.rand((B, L), device=device) * 0.3 + 1.05  # 1.05~1.35
            txFactor = torch.rand((B, L), device=device) * 1.6 - 0.8  # -0.8~0.8
            tyFactor = torch.rand((B, L), device=device) * 1.6 - 0.8  # -0.8~0.8
        else:
            scaleFactor = torch.rand((B, 1), device=device) * 0.3 + 1.05  # 1.05~1.35
            txFactor = torch.rand((B, 1), device=device) * 1.6 - 0.8  # -0.8~0.8
            tyFactor = torch.rand((B, 1), device=device) * 1.6 - 0.8  # -0.8~0.8

        raw_bbx_size = bbx_size / base_enlarge
        bbx_size = raw_bbx_size * scaleFactor
        center_x += raw_bbx_size / 2 * ((scaleFactor - 1) * txFactor)
        center_y += raw_bbx_size / 2 * ((scaleFactor - 1) * tyFactor)

    return torch.stack([center_x, center_y, bbx_size], dim=-1)


def get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2):
    """
    Args:
        bbx_xyxy: (N, 4) [x1, y1, x2, y2]
    Returns:
        bbx_xys: (N, 3) [center_x, center_y, size]
    """

    i_p2d = torch.stack([bbx_xyxy[:, [0, 1]], bbx_xyxy[:, [2, 3]]], dim=1)  # (L, 2, 2)
    bbx_xys = get_bbx_xys(i_p2d[None], base_enlarge=base_enlarge)[0]
    return bbx_xys


def get_bbx_xys_from_xyxy_batch(bbx_xyxy, base_enlarge=1.2):
    """
    Args:
        bbx_xyxy: (B, N, 4) [x1, y1, x2, y2]
    Returns:
        bbx_xys: (B, N, 3) [center_x, center_y, size]
    """
    B, N, _ = bbx_xyxy.size()
    bbx_xys = torch.zeros((B, N, 3), dtype=bbx_xyxy.dtype, device=bbx_xyxy.device)

    for b in range(B):
        bbx_xys[b] = get_bbx_xys_from_xyxy(bbx_xyxy[b], base_enlarge=base_enlarge)

    return bbx_xys