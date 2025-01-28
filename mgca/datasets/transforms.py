import cv2
import numpy as np
import torchvision.transforms as transforms
import random
from PIL import ImageFilter
from PIL import Image
from typing import Iterable

def otsu_mask(img):
    median = np.median(img)
    _, thresh = cv2.threshold(img, median, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def right_orient_mammogram(image):
    convert = False
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        convert = True
    left_nonzero = cv2.countNonZero(image[:, 0:int(0.5 * image.shape[1])])
    right_nonzero = cv2.countNonZero(image[:, int(0.5 * image.shape[1]):])
    
    is_flipped = (left_nonzero < right_nonzero)
    if is_flipped:
        image = cv2.flip(image, 1)

    if convert:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image, is_flipped

def remove_text_label(image):
    # Convert the image to a NumPy array if it's a PIL image
    if isinstance(image, Image.Image):
        image = np.array(image)
    convert = False
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        convert = True
    if image.max() <= 1:
        image = (image * 255).astype(np.uint8) # Convert to 8-bit if not already

    # Binarize the image using a naive non-zero thresholding
    binary_image = (image > 0).astype(np.uint8) * 255
    
    # Apply Gaussian blur to the binarized image
    blurred_image = cv2.GaussianBlur(binary_image, (5, 5), 2.0)
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(blurred_image, connectivity=8)
    
    # Create an output image to store the result
    output_image = image.copy()
    
    # Remove small connected components
    for i in range(1, num_labels):  # Start from 1 to skip the background
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 1e4:  # Threshold for small areas, adjust as needed
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            output_image[y:y+h, x:x+w] = 0  # Set the region to black
    if convert:
        output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2RGB)
    return output_image

class OtsuCut(object):

    def __init__(self, align_orientation: bool = True, remove_text: bool = True):
        super().__init__()
        self.algn_orientation = align_orientation
        self.remove_text = remove_text

    def __process__(self, x):
        if isinstance(x, Image.Image):
            x = np.array(x)
        
        if self.algn_orientation:
            x, _ = right_orient_mammogram(x)
        if self.remove_text:
            x = remove_text_label(x)
        
        mask = otsu_mask(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY))
        # Convert to NumPy array if not already

        # Check if the matrix is empty or has no '1's
        if mask.size == 0 or not np.any(mask):
            return Image.fromarray(x)

        # Find the rows and columns where '1' appears
        rows = np.any(mask == 255, axis=1)
        cols = np.any(mask == 255, axis=0)

        # Find the indices of the rows and columns
        min_row, max_row = np.where(rows)[0][[0, -1]]
        min_col, max_col = np.where(cols)[0][[0, -1]]

        # Crop and return the submatrix
        x = x[min_row:max_row+1, min_col:max_col+1]
        
        img = Image.fromarray(x)
        return img

    def __call__(self, x):
        if isinstance(x, Iterable):
            return [self.__process__(im) for im in x]
        else:
            return self.__process__(x)

class DataTransforms(object):
    def __init__(self, is_train: bool = True, crop_size: int = 224):
        if is_train:
            data_transforms = [
                OtsuCut(),
                transforms.Resize(crop_size),
                transforms.RandomCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5,  0.5, 0.5))
            ]
        else:
            data_transforms = [
                OtsuCut(),
                transforms.Resize(crop_size),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]

        self.data_transforms = transforms.Compose(data_transforms)

    def __call__(self, image):
        return self.data_transforms(image)


class DetectionDataTransforms(object):
    def __init__(self, is_train: bool = True, crop_size: int = 224, jitter_strength: float = 1.):
        if is_train:
            self.color_jitter = transforms.ColorJitter(
                0.8 * jitter_strength,
                0.8 * jitter_strength,
                0.8 * jitter_strength,
                0.2 * jitter_strength,
            )

            kernel_size = int(0.1 * 224)
            if kernel_size % 2 == 0:
                kernel_size += 1

            data_transforms = [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        else:
            data_transforms = [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]

        self.data_transforms = transforms.Compose(data_transforms)

    def __call__(self, image):
        return self.data_transforms(image)


class Moco2Transform(object):
    def __init__(self, is_train: bool = True, crop_size: int = 224) -> None:
        if is_train:
            # This setting follows SimCLR
            self.data_transforms = transforms.Compose(
                [
                    transforms.RandomCrop(crop_size),
                    transforms.RandomApply(
                        [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ]
            )
        else:
            self.data_transforms = transforms.Compose(
                [
                    transforms.CenterCrop(crop_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ]
            )

    def __call__(self, img):
        return self.data_transforms(img)


class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709."""

    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
