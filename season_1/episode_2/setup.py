import os

from setuptools import setup


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="jax_classification",
    version="0.0.1",
    author="Saurav Maheshkar & Soumik Rakshit",
    author_email="soumik.rakshit@wandb.com",
    description=("Image Classification on TPUs using Jax"),
    license="Apache License 2.0",
    keywords="classification jax tensorflow tpu",
    packages=["jax_classification", "tests"],
    long_description=read("README.md"),
)
