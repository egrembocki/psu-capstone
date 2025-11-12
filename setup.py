from setuptools import find_packages, setup

setup(
    name="psu_capstone",
    version="0.1.0",
    description="PSU Capstone Project",
    author="Team20 SWENG 480",
    author_email="",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        # Add your runtime dependencies here, e.g.:
        # "numpy>=1.23.0",
    ],
    extras_require={
        "dev": [
            "pytest",
            "flake8",
            "black",
            "isort",
            "pre-commit",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
