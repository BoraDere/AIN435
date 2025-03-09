import skimage as ski
import numpy as np
import matplotlib.pyplot as plt

image = ski.io.imread('temp.png')

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

print(f'Patch 1: {patch1}')
print(f'Patch 2: {patch2}')

# mean_patch1 = np.mean(patch1, axis=(0, 1))
# mean_patch2 = np.mean(patch2, axis=(0, 1))


# def white_balance(image, patch_mean):
#     balanced = image / patch_mean
#     return np.clip(balanced, 0, 1)


# image1 = white_balance(image, mean_patch1)
# image2 = white_balance(image, mean_patch2)


# def brightness_adjustment(image):
#     gray_image = rgb2gray(image)
#     gray_mean = gray_image.mean()
#     scale_factor = 0.25 / gray_mean
#     brightened_image = image * scale_factor
#     return np.clip(brightened_image, 0, 1)


# image1 = brightness_adjustment(image1)
# image2 = brightness_adjustment(image2)


# def gamma_encoding(image):
#     return np.where(
#         image <= 0.0031308,
#         12.92 * image,
#         (1 + 0.055) * np.power(image, 1/2.4) - 0.055
#     )


# image1 = gamma_encoding(image1)
# image2 = gamma_encoding(image2)

# ski.io.imsave("output_patch1.png", (image1 * 255).astype(np.uint8))
# ski.io.imsave("output_patch2.png", (image2 * 255).astype(np.uint8))