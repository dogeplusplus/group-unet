import sys
from setuptools import setup, find_packages

sys.path[0:0] = ["group_unet"]
from version import __version__

setup(
  name = "group_unet",
  packages = find_packages(),
  version = __version__,
  description = "Group Equivariant Unet",
  install_requires=[
      "einops",
      "numpy",
      "tqdm",
      "torch",
      "torchvision",
      "pillow",
  ],
)