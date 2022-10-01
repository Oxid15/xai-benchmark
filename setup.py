import os
import sys
import setuptools
sys.path.append(os.path.dirname(__file__))

import xaib


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="xai-benchmark",
    version=xaib.__version__,
    author=xaib.__author__,
    author_email=xaib.__author_email__,
    license='MIT',
    description="Benchmark for Explainable AI methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/oxid15/xai-benchmark",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
    package_dir={"xaib": "./xaib"},
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    install_requires=[
        'cascade-ml>=0.7.2',
    ]
)