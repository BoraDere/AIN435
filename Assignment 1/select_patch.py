import skimage as ski
import numpy as np
import matplotlib.pyplot as plt

from skimage.color import rgb2gray

from scipy.interpolate import RegularGridInterpolator

black = 0
white = 4095

r_scale = 2.629696
g_scale = 1.000000
b_scale = 1.251207

image = ski.io.imread("sample.tiff")
image = image.astype(float)

image = (image - black) / (white - black)
np.clip(image, 0, 1, out=image)


def extract_bayer_channels(image: np.ndarray[float]): # to extract the bayer patterns for experiments
    """
    Function to exxtract channels according to the BGGR Bayer pattern.

    :param image: 2D np.array of floats.

    :return R, G1, G2, B: Channels according to the BGGR Bayer pattern.
    """
    
    B = image[0::2, 0::2]
    G1 = image[0::2, 1::2]
    G2 = image[1::2, 0::2]
    R = image[1::2, 1::2]

    return R, G1, G2, B


def preset_mosaic(image: np.ndarray[float], r_scale: float, g_scale: float, b_scale: float):
    """
    Function to apply white balancing using the device preset.

    :param image: 2D np.array of floats.
    :param r_scale: r_scale value of the device, float.
    :param g_scale: g_scale value of the device, float.
    :param b_scale: b_scale value of the device, float.

    :return balanced_image: 2D np.array of floats, represents white balanced image.
    """
    R, G1, G2, B = extract_bayer_channels(image)

    R_balanced = R * r_scale
    G1_balanced = G1 * g_scale
    G2_balanced = G2 * g_scale
    B_balanced = B * b_scale

    np.clip(R_balanced, 0, 1, out=R_balanced)
    np.clip(B_balanced, 0, 1, out=B_balanced)

    balanced_image = image.copy()
    balanced_image[0::2, 0::2] = B_balanced
    balanced_image[0::2, 1::2] = G1_balanced
    balanced_image[1::2, 0::2] = G2_balanced
    balanced_image[1::2, 1::2] = R_balanced

    return balanced_image


def demosaic_bilinear(image: np.ndarray[float]):
    """
    Function to apply bilinear demosaicing according to the BGGR Bayer pattern.

    :param image: 2D np.array of floats.

    :return rgb_image: Demosaiced RGB image.
    """
    h, w = image.shape

    # grid for output
    y_full = np.arange(h)
    x_full = np.arange(w)

    grid_coords = np.array(np.meshgrid(y_full, x_full, indexing='ij')).transpose(1, 2, 0) # (h, w, 2)
    flat_coords = grid_coords.reshape(-1, 2)  # (h*w, 2)
    
    B = image[0::2, 0::2]
    G1 = image[0::2, 1::2]
    G2 = image[1::2, 0::2]
    R = image[1::2, 1::2]
    
    y_B = np.arange(0, h, 2); x_B = np.arange(0, w, 2)
    y_G1 = np.arange(0, h, 2); x_G1 = np.arange(1, w, 2)
    y_G2 = np.arange(1, h, 2); x_G2 = np.arange(0, w, 2)
    y_R = np.arange(1, h, 2); x_R = np.arange(1, w, 2)
    
    R_interp = RegularGridInterpolator((y_R, x_R), R, method='linear', bounds_error=False, fill_value=None)
    B_interp = RegularGridInterpolator((y_B, x_B), B, method='linear', bounds_error=False, fill_value=None)
    G1_interp = RegularGridInterpolator((y_G1, x_G1), G1, method='linear', bounds_error=False, fill_value=None)
    G2_interp = RegularGridInterpolator((y_G2, x_G2), G2, method='linear', bounds_error=False, fill_value=None)
    
    R_full = R_interp(flat_coords).reshape(h, w)
    B_full = B_interp(flat_coords).reshape(h, w)
    G1_full = G1_interp(flat_coords).reshape(h, w)
    G2_full = G2_interp(flat_coords).reshape(h, w)
    G_full = (G1_full + G2_full) / 2
        
    rgb_image = np.stack([R_full, G_full, B_full], axis=-1)
    np.clip(rgb_image, 0, 1, out=rgb_image)
    return rgb_image

image = preset_mosaic(image, r_scale, g_scale, b_scale)
image = demosaic_bilinear(image)

xyz_to_cam = np.array([
    [6992, -1668, -806],
    [-8138, 15748, 2543],
    [-874, 850, 7897]
])

xyz_to_cam = xyz_to_cam / 10000

srgb_to_xyz = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]
])

srgb_to_cam = xyz_to_cam @ srgb_to_xyz
srgb_to_cam_normalized = srgb_to_cam / srgb_to_cam.sum(axis=1, keepdims=True)
cam_to_srgb = np.linalg.inv(srgb_to_cam_normalized)

image = image.reshape(-1, 3)
image = image @ cam_to_srgb.T
image = image.reshape(2014, 3039, 3)
image = np.clip(image, 0, 1, out=image)

plt.imshow(image)
points = plt.ginput(2)
plt.show()
plt.close()

points = np.array(points, dtype=int)

patch_size = 2


def get_patch(image, center, size):
    x, y = center
    return image[max(0, y - size//2): y + size//2 + 1, max(0, x - size//2): x + size//2 + 1, :]


patch1 = get_patch(image, points[0], patch_size)
patch2 = get_patch(image, points[1], patch_size)

mean_patch1 = np.mean(patch1, axis=(0, 1))
mean_patch2 = np.mean(patch2, axis=(0, 1))


def white_balance(image, patch_mean):
    balanced = image / patch_mean
    return np.clip(balanced, 0, 1)


image1 = white_balance(image, mean_patch1)
image2 = white_balance(image, mean_patch2)


def brightness_adjustment(image):
    gray_image = rgb2gray(image)
    gray_mean = gray_image.mean()
    scale_factor = 0.25 / gray_mean
    brightened_image = image * scale_factor
    return np.clip(brightened_image, 0, 1)


image1 = brightness_adjustment(image1)
image2 = brightness_adjustment(image2)


def gamma_encoding(image):
    return np.where(
        image <= 0.0031308,
        12.92 * image,
        (1 + 0.055) * np.power(image, 1/2.4) - 0.055
    )


image1 = gamma_encoding(image1)
image2 = gamma_encoding(image2)

ski.io.imsave("output_patch1.png", (image1 * 255).astype(np.uint8))
ski.io.imsave("output_patch2.png", (image2 * 255).astype(np.uint8))