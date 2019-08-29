import torch
import cv2 as cv
import numpy as np
import PIL
from PIL import ImageEnhance, Image

augmentations = [
    'additive_gaussian_noise',
    'additive_speckle_noise',
    'random_brightness',
    'random_contrast',
    'additive_shade',
    'motion_blur'
]


def additive_gaussian_noise(image, config):
    stddev = np.random.randint(config['stddev_range'][0], config['stddev_range'][1])
    noise = np.random.randn(*image.shape) * stddev
    noisy_image = np.clip(image + noise, 0, 255)

    return noisy_image.astype(np.uint8)


def additive_speckle_noise(image, config):
    prob = np.random.uniform(config['prob_range'][0], config['prob_range'][1])
    sample = np.random.random_sample(image.shape)
    noisy_image = np.where(sample <= prob, np.zeros_like(image), image)
    noisy_image = np.where(sample >= (1. - prob), 255 * np.ones_like(image), noisy_image)

    return noisy_image


def random_brightness(image, config):
    brightness = np.clip(np.random.normal(loc=1, scale=0.5), 1 - config['max_abs_change'], 1 + config['max_abs_change'])
    enhancer = ImageEnhance.Brightness(PIL.Image.fromarray(np.uint8(image)))

    return np.clip(np.asarray(enhancer.enhance(brightness)), 0, 255)


def random_contrast(image, config):
    contrast = np.clip(np.random.normal(loc=1, scale=0.5), config['strength_range'][0], config['strength_range'][1])
    enhancer = ImageEnhance.Contrast(PIL.Image.fromarray(np.uint8(image)))

    return np.clip(np.asarray(enhancer.enhance(contrast)), 0, 255)


def additive_shade(image, config, nb_ellipses=20):
    min_dim = min(image.shape[:2]) / 4
    mask = np.zeros(image.shape[:2], np.uint8)

    for i in range(nb_ellipses):
        ax = int(max(np.random.rand() * min_dim, min_dim / 5))
        ay = int(max(np.random.rand() * min_dim, min_dim / 5))
        max_rad = max(ax, ay)
        x = np.random.randint(max_rad, image.shape[1] - max_rad)  # center
        y = np.random.randint(max_rad, image.shape[0] - max_rad)
        angle = np.random.rand() * 90
        cv.ellipse(mask, (x, y), (ax, ay), angle, 0, 360, 255, -1)

    transparency = np.random.uniform(config['transparency_range'][0], config['transparency_range'][1])
    kernel_size = np.random.randint(config['kernel_size_range'][0], config['kernel_size_range'][1])

    if (kernel_size % 2) == 0:  # kernel_size has to be odd
        kernel_size += 1

    mask = cv.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), 0)
    shaded = image * (1 - transparency * mask[...] / 255.)

    return np.clip(shaded, 0, 255).astype(np.uint8)


def motion_blur(image, config):
    mode = np.random.choice(['h', 'v', 'diag_down', 'diag_up'])
    ksize = np.random.randint(0, (config['max_kernel_size'] + 1) / 2) * 2 + 1  # make sure is odd
    center = int((ksize - 1) / 2)
    kernel = np.zeros((ksize, ksize))

    if mode == 'h':
        kernel[center, :] = 1.
    elif mode == 'v':
        kernel[:, center] = 1.
    elif mode == 'diag_down':
        kernel = np.eye(ksize)
    elif mode == 'diag_up':
        kernel = np.flip(np.eye(ksize), 0)

    var = ksize * ksize / 16.
    grid = np.repeat(np.arange(ksize)[:, np.newaxis], ksize, axis=-1)
    gaussian = np.exp(-(np.square(grid - center) + np.square(grid.T - center)) / (2. * var))
    kernel *= gaussian
    kernel /= np.sum(kernel)
    image = cv.filter2D(image, -1, kernel)

    return image



