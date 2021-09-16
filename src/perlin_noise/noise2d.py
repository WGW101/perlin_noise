import numba
import numpy as np


@numba.njit
def _perlin2d_internal(hperiod: int, vperiod: int, angles: np.ndarray) -> np.ndarray:
      grad_x = np.cos(angles).reshape(angles.shape[0], 1, angles.shape[1], 1)
      grad_y = np.sin(angles).reshape(angles.shape[0], 1, angles.shape[1], 1)

      hoffsets = np.linspace(0, 1, hperiod + 1)[:-1].reshape(1, 1, 1, -1)
      voffsets = np.linspace(0, 1, vperiod + 1)[:-1].reshape(1, -1, 1, 1)

      hsmooth = ((6 * hoffsets - 15) * hoffsets + 10) * hoffsets ** 3
      vsmooth = ((6 * voffsets - 15) * voffsets + 10) * voffsets ** 3

      ul = grad_x[:-1, :, :-1, :] * hoffsets + grad_y[:-1, :, :-1, :] * voffsets
      ur = grad_x[:-1, :, 1:, :] * (hoffsets - 1) + grad_y[:-1, :, 1:, :] * voffsets
      bl = grad_x[1:, :, :-1, :] * hoffsets + grad_y[1:, :, :-1, :] * (voffsets - 1)
      br = grad_x[1:, :, 1:, :] * (hoffsets - 1) + grad_y[1:, :, 1:, :] * (voffsets - 1)

      u = (1 - hsmooth) * ul + hsmooth * ur
      b = (1 - hsmooth) * bl + hsmooth * br
      img = (1 - vsmooth) * u + vsmooth * b
      return img


def perlin2d(width: int, height: int,
             hperiod: int, vperiod: int,
             rng: np.random.Generator) -> np.ndarray:
      grid_shape = np.ceil((height / vperiod, width / hperiod)).astype(np.int64) + 1
      angles = 2 * np.pi * rng.random(grid_shape)
      img = _perlin2d_internal(hperiod, vperiod, angles)
      img = img.reshape((grid_shape - 1) * (vperiod, hperiod))[:height, :width]
      return img


def fractal2d(width: int, height: int,
              hperiod: int, vperiod: int,
              octaves: int, persistence: float,
              rng: np.random.Generator) -> np.ndarray:
      amp = (1 - persistence) / (1 - persistence ** octaves)
      img = np.zeros((height, width))
      for _ in range(octaves):
            img += amp * perlin2d(width, height, hperiod, vperiod, rng)
            amp *= persistence
            hperiod = hperiod >> 1
            vperiod = vperiod >> 1
      return img