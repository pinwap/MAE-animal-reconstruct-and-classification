IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
IMG_SIZE      = 224
PATCH_SIZE    = 16
NUM_PATCHES_SIDE = IMG_SIZE // PATCH_SIZE
NUM_PATCHES      = NUM_PATCHES_SIDE ** 2

ANIMALS10_EN = [
    "dog", "horse", "elephant", "butterfly", "chicken",
    "cat", "cow", "sheep", "spider", "squirrel",
]
IDX_TO_CLASS = {i: en for i, en in enumerate(ANIMALS10_EN)}
