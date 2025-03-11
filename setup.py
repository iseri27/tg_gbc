# Copyright (c) Corona.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

from setuptools import find_packages, setup

setup(
    name="tggbc",
    version="1.0",
    author="Lizhen Xu",
    url="https://github.com/iseri27/tg_gbc",
    description="Trim Gradually Guided By Classification scores.",
    install_requires=["torch"],
    packages=find_packages(exclude=("examples", "build")),
)
