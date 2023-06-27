# Copyright 2023 The videogvt Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Lib for creating condition video."""
from typing import Callable

import jax.numpy as jnp

CondDict = dict[str, jnp.ndarray]
CondFn = Callable[[jnp.array], CondDict]


def latent_frame_prediction_cond(
    input_video: jnp.ndarray, *, cond_latent_frames: int,
    latent_shape: tuple[int, int, int]) -> CondDict:
  assert input_video.ndim == 5  # bs, t, h, w, c
  bs, t, _, _, _ = input_video.shape
  l_t, _, _ = latent_shape
  cond_video_mask = jnp.zeros((bs, t, 1, 1, 1), dtype=bool)
  cond_latent_mask = jnp.zeros((bs, l_t, 1, 1), dtype=bool)
  cond_latent_mask = cond_latent_mask.at[:, :cond_latent_frames].set(True)
  return {
      'video': input_video,
      'video_mask': cond_video_mask,
      'latent_mask': cond_latent_mask
  }


def frame_prediction_cond(input_video: jnp.ndarray,
                          *,
                          cond_frames: int,
                          cond_padding: str,
                          cond_latent_frames: int,
                          latent_shape: tuple[int, int, int],
                          prefix_condition: bool = False) -> CondDict:
  """Frame prediction condition.

  Args:
    input_video: [batch, T, H, W, C] input video.
    cond_frames: number of video frames as condition.
    cond_padding: padding method for masked non-condition video frames.
    cond_latent_frames: number of latent frames as condition.
    latent_shape: 3D shape of latent code.
    prefix_condition: whether condition is prefix or inside latent code. When
      inside, the condition region is reflected in returned masks.

  Returns:
    Dictionary containing the condition video, mask of condition video,
    and mask of condition latent.
  """
  assert input_video.ndim == 5  # bs, t, h, w, c
  bs, t, _, _, _ = input_video.shape
  l_t, _, _ = latent_shape
  assert 0 < cond_frames < t
  cond_video_mask = jnp.zeros((bs, t, 1, 1, 1), dtype=bool)
  cond_latent_mask = jnp.zeros((bs, l_t, 1, 1), dtype=bool)
  if not prefix_condition:
    cond_video_mask = cond_video_mask.at[:, :cond_frames].set(True)
    cond_latent_mask = cond_latent_mask.at[:, :cond_latent_frames].set(True)
  pad = ((0, 0), (0, t - cond_frames), (0, 0), (0, 0), (0, 0))
  cond_video = jnp.pad(input_video[:, :cond_frames], pad, mode=cond_padding)
  assert cond_video.shape == input_video.shape
  return {
      'video': cond_video,
      'video_mask': cond_video_mask,
      'latent_mask': cond_latent_mask
  }


def frame_interpolation_cond(input_video: jnp.ndarray, *, cond_frames: int,
                             cond_padding: str, cond_latent_frames: int,
                             latent_shape: tuple[int, int, int]) -> CondDict:
  """Frame interpolation condition.

  Args:
    input_video: [batch, T, H, W, C] input video.
    cond_frames: number of video frames as condition at both ends.
    cond_padding: padding method for masked non-condition video frames.
    cond_latent_frames: number of latent frames as condition at both ends.
    latent_shape: 3D shape of latent code.

  Returns:
    Dictionary containing the condition video, mask of condition video,
    and mask of condition latent.
  """
  assert input_video.ndim == 5  # bs, t, h, w, c
  bs, t, _, _, _ = input_video.shape
  l_t, _, _ = latent_shape
  assert 0 < 2 * cond_frames < t
  cond_video_mask = jnp.ones((bs, t, 1, 1, 1), dtype=bool)
  cond_latent_mask = jnp.ones((bs, l_t, 1, 1), dtype=bool)
  cond_video_mask = cond_video_mask.at[:, cond_frames:-cond_frames].set(False)
  cond_latent_mask = cond_latent_mask.at[:, cond_latent_frames:
                                         -cond_latent_frames].set(False)
  if cond_padding == 'interpolate':
    start = input_video[:, cond_frames - 1:cond_frames]
    end = input_video[:, t - cond_frames:t - cond_frames + 1]
    length = t - 2 * cond_frames
    fractions = jnp.arange(1, length + 1) / (length + 1)
    middle = start + (end - start) * fractions[:, None, None, None]
    cond_video = input_video.at[:, cond_frames:-cond_frames].set(middle)
  else:
    raise NotImplementedError(f'Unsupported cond_padding: {cond_padding}')
  assert cond_video.shape == input_video.shape
  return {
      'video': cond_video,
      'video_mask': cond_video_mask,
      'latent_mask': cond_latent_mask
  }


def outpainting_cond(input_video: jnp.ndarray, *, cond_region: str,
                     cond_padding: str, latent_shape: tuple[int, int, int]):
  """Outpainting condition.

  Args:
    input_video: [batch, T, H, W, C] input video.
    cond_region: region as condition.
    cond_padding: padding method for masked non-condition video frames.
    latent_shape: 3D shape of latent code.

  Returns:
    Dictionary containing the condition video, mask of condition video,
    and mask of condition latent.
  """
  assert input_video.ndim == 5  # bs, t, h, w, c
  bs, t, h, w, _ = input_video.shape
  l_t, l_h, l_w = latent_shape
  cond_video_mask = jnp.zeros((bs, 1, h, w, 1), dtype=bool)
  cond_latent_mask = jnp.zeros((bs, 1, l_h, l_w), dtype=bool)
  if cond_region.startswith('quarter_'):
    region = cond_region[len('quarter_'):]
    if region == 'topleft':
      h_half, w_half = h // 2, w // 2
      l_h_half, l_w_half = l_h // 2, l_w // 2
      cond_video_mask = cond_video_mask.at[..., :h_half, :w_half, :].set(True)
      cond_latent_mask = cond_latent_mask.at[..., :l_h_half, :l_w_half].set(
          True)
      pad = ((0, 0), (0, 0), (0, h_half), (0, w_half), (0, 0))
      cond_video = jnp.pad(
          input_video[..., :h_half, :w_half, :], pad, mode=cond_padding)
    else:
      raise ValueError(f'Unsupported quarter region: {region}')
  elif cond_region.startswith('rectangle_'):
    region = cond_region[len('rectangle_'):]
    if region == 'horizontal':
      h_start, h_end = h // 4, h * 3 // 4
      l_h_start, l_h_end = l_h // 4, l_h * 3 // 4
      cond_video_mask = cond_video_mask.at[..., h_start:h_end, :, :].set(True)
      cond_latent_mask = cond_latent_mask.at[...,
                                             l_h_start:l_h_end, :].set(True)
      pad = ((0, 0), (0, 0), (h_start, h - h_end), (0, 0), (0, 0))
      cond_video = jnp.pad(
          input_video[..., h_start:h_end, :, :], pad, mode=cond_padding)
    elif region == 'vertical':
      w_start, w_end = w // 4, w * 3 // 4
      l_w_start, l_w_end = l_w // 4, l_w * 3 // 4
      cond_video_mask = cond_video_mask.at[..., w_start:w_end, :].set(True)
      cond_latent_mask = cond_latent_mask.at[..., l_w_start:l_w_end].set(True)
      pad = ((0, 0), (0, 0), (0, 0), (w_start, w - w_end), (0, 0))
      cond_video = jnp.pad(
          input_video[..., w_start:w_end, :], pad, mode=cond_padding)
    elif region == 'central':
      h_start, h_end = h // 4, h * 3 // 4
      w_start, w_end = w // 4, w * 3 // 4
      l_h_start, l_h_end = l_h // 4, l_h * 3 // 4
      l_w_start, l_w_end = l_w // 4, l_w * 3 // 4
      cond_video_mask = cond_video_mask.at[..., h_start:h_end,
                                           w_start:w_end, :].set(True)
      cond_latent_mask = cond_latent_mask.at[..., l_h_start:l_h_end,
                                             l_w_start:l_w_end].set(True)
      pad = ((0, 0), (0, 0), (h_start, h - h_end), (w_start, w - w_end), (0, 0))
      cond_video = jnp.pad(
          input_video[..., h_start:h_end, w_start:w_end, :],
          pad,
          mode=cond_padding)
    else:
      raise ValueError(f'Unsupported rectangle region: {region}')
  elif cond_region.startswith('dynamic_'):
    region = cond_region[len('dynamic_'):]
    if region == 'vertical':
      w_start = jnp.round(jnp.linspace(0, w // 2, t)).astype(jnp.int32)
      w_start = w_start[None, :, None, None, None]
      w_end = w_start + w // 2
      w_idx = jnp.arange(w)[None, None, None, :, None]
      l_w_start = jnp.round(jnp.linspace(0, l_w // 2, l_t)).astype(jnp.int32)
      l_w_start = l_w_start[None, :, None, None]
      l_w_end = l_w_start + l_w // 2
      l_w_idx = jnp.arange(l_w)[None, None, None, :]
      cond_video_mask = jnp.where((w_idx >= w_start) & (w_idx < w_end), True,
                                  cond_video_mask)
      cond_latent_mask = jnp.where((l_w_idx >= l_w_start) & (l_w_idx < l_w_end),
                                   True, cond_latent_mask)
      if cond_padding == 'constant':
        cond_video = jnp.where(cond_video_mask, input_video, 0)
      else:
        raise ValueError(f'Unsupported cond_padding: {cond_padding}')
    else:
      raise ValueError(f'Unsupported dynamic region: {region}')
  else:
    raise ValueError(f'Unsupported cond_region: {cond_region}')
  assert cond_video.shape == input_video.shape
  return {
      'video': cond_video,
      'video_mask': cond_video_mask,
      'latent_mask': cond_latent_mask
  }


def inpainting_cond(input_video: jnp.ndarray, *, cond_region: str,
                    cond_padding: str, latent_shape: tuple[int, int, int]):
  """Inpainting condition.

  Args:
    input_video: [batch, T, H, W, C] input video.
    cond_region: region as condition.
    cond_padding: padding method for masked non-condition video frames.
    latent_shape: 3D shape of latent code.

  Returns:
    Dictionary containing the condition video, mask of condition video,
    and mask of condition latent.
  """
  assert input_video.ndim == 5  # bs, t, h, w, c
  # TODO(kihyuks): support for different padding and padding values.
  assert cond_padding == 'constant', f'Unsupported cond_padding: {cond_padding}'
  pad_value = 0.0
  bs, t, h, w, _ = input_video.shape
  l_t, l_h, l_w = latent_shape
  cond_video_mask = jnp.ones((bs, 1, h, w, 1), dtype=bool)
  cond_latent_mask = jnp.ones((bs, 1, l_h, l_w), dtype=bool)
  if cond_region.startswith('quarter_'):
    region = cond_region[len('quarter_'):]
    if region == 'topleft':
      h_half, w_half = h // 2, w // 2
      l_h_half, l_w_half = l_h // 2, l_w // 2
      cond_video_mask = cond_video_mask.at[..., :h_half, :w_half, :].set(False)
      cond_latent_mask = cond_latent_mask.at[..., :l_h_half, :l_w_half].set(
          False)
      cond_video = jnp.where(cond_video_mask == 1, input_video,
                             pad_value * jnp.ones_like(input_video))
    else:
      raise ValueError(f'Unsupported quarter region: {region}')
  elif cond_region.startswith('rectangle_'):
    region = cond_region[len('rectangle_'):]
    if region == 'central':
      h_start, h_end = h // 4, h * 3 // 4
      w_start, w_end = w // 4, w * 3 // 4
      l_h_start, l_h_end = l_h // 4, l_h * 3 // 4
      l_w_start, l_w_end = l_w // 4, l_w * 3 // 4
      cond_video_mask = cond_video_mask.at[..., h_start:h_end,
                                           w_start:w_end, :].set(False)
      cond_latent_mask = cond_latent_mask.at[..., l_h_start:l_h_end,
                                             l_w_start:l_w_end].set(False)
      cond_video = jnp.where(cond_video_mask == 1, input_video,
                             pad_value * jnp.ones_like(input_video))
    else:
      raise ValueError(f'Unsupported rectangle region: {region}')
  elif cond_region.startswith('dynamic_'):
    region = cond_region[len('dynamic_'):]
    if region == 'central':
      w_start = jnp.round(jnp.linspace(0, w // 2, t)).astype(jnp.int32)
      w_start = w_start[None, :, None, None, None]
      w_end = w_start + w // 2
      w_idx = jnp.arange(w)[None, None, None, :, None]
      h_start, h_end = h // 4, h * 3 // 4
      l_w_start = jnp.round(jnp.linspace(0, l_w // 2, l_t)).astype(jnp.int32)
      l_w_start = l_w_start[None, :, None, None]
      l_w_end = l_w_start + l_w // 2
      l_w_idx = jnp.arange(l_w)[None, None, None, :]
      l_h_start, l_h_end = l_h // 4, l_h * 3 // 4
      cond_video_mask = jnp.where((w_idx >= w_start) & (w_idx < w_end), False,
                                  cond_video_mask)
      cond_video_mask = cond_video_mask.at[..., :h_start, :, :].set(True)
      cond_video_mask = cond_video_mask.at[..., h_end:, :, :].set(True)
      cond_latent_mask = jnp.where((l_w_idx >= l_w_start) & (l_w_idx < l_w_end),
                                   False, cond_latent_mask)
      cond_latent_mask = cond_latent_mask.at[..., :l_h_start, :].set(True)
      cond_latent_mask = cond_latent_mask.at[..., l_h_end:, :].set(True)
      cond_video = jnp.where(cond_video_mask == 1, input_video,
                             pad_value * jnp.ones_like(input_video))
    else:
      raise ValueError(f'Unsupported dynamic region: {region}')
  else:
    raise ValueError(f'Unsupported cond_region: {cond_region}')
  assert cond_video.shape == input_video.shape
  return {
      'video': cond_video,
      'video_mask': cond_video_mask,
      'latent_mask': cond_latent_mask
  }
