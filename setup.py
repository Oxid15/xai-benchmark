import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="xai-benchmark",
    version="0.3.0",
    author="Ilia Moiseev",
    author_email="ilia.moiseev.5@yandex.ru",
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
    install_requires=["cascade-ml", "scikit-learn", "plotly", "kaleido"],
)
