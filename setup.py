from version import __version__
import sys
from setuptools import setup, find_packages

sys.path[0:0] = ["group_unet"]

setup(
    name="group_unet",
    packages=find_packages(),
    version=__version__,
    description="Group Equivariant Unet",
    install_requires=[
        "einops",
        "numpy",
        "tqdm",
        "torch",
        "torchvision",
        "torchmetrics",
        "albumentations",
        "opencv-python",
    ],
)
