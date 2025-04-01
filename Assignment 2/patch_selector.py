# patch_selector.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

hdr_image = cv2.imread('gaussian_log_tiff.hdr', flags=cv2.IMREAD_ANYDEPTH)
hdr_image = hdr_image[:, :, ::-1]

def gamma_encoding(image):
    """
    Function to apply gamma encoding.

    :param image: 2D np.array of floats.

    :return gamma_encoded: Gamma encoded image.
    """
    gamma_encoded = np.where(
        image <= 0.0031308,
        12.92 * image,
        (1 + 0.055) * np.power(image, 1/2.4) - 0.055
    )

    return gamma_encoded

def simple_tonemap(img, scale=0.02):
    scaled = np.clip(img * scale, 0, 1)
    gamma_encoded = gamma_encoding(scaled)
    return gamma_encoded

display_img = simple_tonemap(hdr_image)
plt.figure(figsize=(15, 10))
plt.imshow(display_img)
plt.axis('on')  

points = plt.ginput(24, timeout=0)

patch_size = 20

plt.figure(figsize=(15, 10))
plt.imshow(display_img)

patch_coords = []

for i, (x, y) in enumerate(points):
    x1, y1 = int(x) - patch_size, int(y) - patch_size
    x2, y2 = int(x) + patch_size, int(y) + patch_size
    
    patch_coords.append([x1, y1, x2, y2])
    
    rect = Rectangle((x1, y1), patch_size*2, patch_size*2, 
                     linewidth=1, edgecolor='r', facecolor='none')
    plt.gca().add_patch(rect)
    plt.text(x1, y1-5, str(i+1), color='white', fontsize=12, 
             bbox=dict(facecolor='red', alpha=0.5))

plt.title('Selected Patches')
plt.axis('on')

print("\nPatch coordinates [x1, y1, x2, y2]:")
print("patch_coords = [")
for coord in patch_coords:
    print(f"\t{coord},")
print("]")

plt.show()