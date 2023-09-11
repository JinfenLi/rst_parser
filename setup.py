#!/usr/bin/env python
import os

from setuptools import find_packages, setup

with open("README.md") as f:
    long_description = f.read()

setup(

    name="rst_parser",
    version='0.1.3',
    description="A package for RST parsing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jinfen Li",
    author_email="jli284@syr.edu",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        'Programming Language :: Python :: 3.9'
    ],
    url="https://github.com/JinfenLi/rst_parser",
    license="MIT",
    dependency_links=["https://download.pytorch.org/whl/cu118"],
    install_requires=["allennlp", "lightning>=2",
                      "numpy", "pandas", "rich", "torchmetrics>=1",
                      "transformers", "neptune>=1.0"],
    packages=find_packages(),


)
