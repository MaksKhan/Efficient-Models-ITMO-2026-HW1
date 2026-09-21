"""Analytical equations transcribed from the supplied notebook.

S must be a multiple of 16 for the simplified closed forms used here.
Units: FLOPs, bytes, seconds, and joules respectively.
"""
import numpy as np


def _inputs(image_size, batch):
    s = np.asarray(image_size, dtype=float)
    b = np.asarray(batch, dtype=float)
    if np.any(s <= 0) or np.any(b <= 0):
        raise ValueError('image_size and batch must be positive')
    return s, b


def _scalar_or_array(value):
    return float(value) if np.ndim(value) == 0 else value


def flops(image_size, batch):
    s, b = _inputs(image_size, batch)
    return _scalar_or_array(b * (17835 * s**2 + 313956))


def memory(image_size, batch):
    s, b = _inputs(image_size, batch)
    return _scalar_or_array(4181312 + 76 * b * s**2)


def bytes_moved(image_size, batch):
    s, b = _inputs(image_size, batch)
    return _scalar_or_array(532 * b * s**2 + 8592 * b + 4181264)


def latency(image_size, batch, theta):
    tau, compute_rate, bandwidth = (float(theta[k]) for k in ('tau', 'compute_rate', 'bandwidth'))
    if tau < 0 or compute_rate <= 0 or bandwidth <= 0:
        raise ValueError('latency parameters must be physically valid')
    return _scalar_or_array(tau + np.maximum(flops(image_size, batch) / compute_rate,
                                               bytes_moved(image_size, batch) / bandwidth))


def energy(image_size, batch, theta_energy):
    p, eps_f, eps_m = (float(theta_energy[k]) for k in ('power', 'energy_per_flop', 'energy_per_byte'))
    if min(p, eps_f, eps_m) < 0:
        raise ValueError('energy parameters must be nonnegative')
    return _scalar_or_array(p * latency(image_size, batch, theta_energy)
                            + eps_f * flops(image_size, batch)
                            + eps_m * bytes_moved(image_size, batch))
