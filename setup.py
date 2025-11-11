"""Setup script for psu_capstone package."""


from setuptools import setup, find_packages

setup(
    name="psu_capstone",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
