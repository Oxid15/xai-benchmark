import os
import sys

import setuptools

sys.path.append(os.path.join(os.path.dirname(__file__), "xaib"))
from version import __author__, __author_email__, __version__

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="xai-benchmark",
    version=__version__,
    author=__author__,
    author_email=__author_email__,
    license="MIT",
    description="Benchmark for Explainable AI methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/oxid15/xai-benchmark",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={"xaib": "./xaib"},
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "cascade-ml>=0.13,<0.14",
        "scikit-learn>=1.4.1,<2",
        "plotly>=5.20,<6",
        "kaleido==0.2.1",
        "numpy<2",
    ],
)
