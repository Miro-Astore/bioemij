import cryojax.simulator as cxs
import jax.numpy as jnp
import numpy as np

from bioemij.utils import apply_filter, compute_whitening_filter


class _NoiseStack:
    """The two things `compute_whitening_filter` reads from a RelionParticleDataset."""

    def __init__(self, images, image_config):
        self.images = images
        self.parameter_file = [{"image_config": image_config}]
        self.requested = []

    def __len__(self):
        return len(self.images)

    def __getitem__(self, indices):
        self.requested.extend(np.atleast_1d(indices).tolist())
        return {"images": self.images[indices]}


def _coloured_noise(n_images, size, seed):
    """White noise with five times more power at low spatial frequency."""
    rng = np.random.default_rng(seed)
    frequency = np.hypot(np.fft.fftfreq(size)[:, None], np.fft.fftfreq(size)[None, :])
    amplitude = np.sqrt(1.0 + 4.0 * np.exp(-((frequency / 0.12) ** 2)))
    noise = rng.normal(size=(n_images, size, size))
    return np.fft.ifftn(np.fft.fftn(noise, axes=(-2, -1)) * amplitude, axes=(-2, -1)).real


def _band_power(images, low, high):
    """Mean power of the images between two spatial frequencies (cycles/pixel)."""
    size = images.shape[-1]
    frequency = np.hypot(np.fft.fftfreq(size)[:, None], np.fft.fftfreq(size)[None, :])
    band = (frequency > low) & (frequency < high)
    power = np.abs(np.fft.fftn(images, axes=(-2, -1))) ** 2
    return power[:, band].mean()


def test_whitening_filter_flattens_coloured_noise():
    size = 64
    image_config = cxs.BasicImageConfig(
        shape=(size, size), pixel_size=1.0, voltage_in_kilovolts=300.0
    )
    stack = _NoiseStack(_coloured_noise(600, size, seed=0), image_config)
    whitening_filter = compute_whitening_filter(
        stack, 100, None, 1.0, max_images=400, radially_average=True
    )

    assert whitening_filter.shape == (size, size)
    assert whitening_filter[0, 0] == 0.0
    assert len(stack.requested) == 400 == len(set(stack.requested))

    fresh = _coloured_noise(200, size, seed=1)
    whitened = np.asarray(jnp.stack([apply_filter(image, whitening_filter) for image in fresh]))
    before = _band_power(fresh, 0.02, 0.08) / _band_power(fresh, 0.3, 0.45)
    after = _band_power(whitened, 0.02, 0.08) / _band_power(whitened, 0.3, 0.45)
    assert before > 3.0
    assert abs(after - 1.0) < 0.15
