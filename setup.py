from setuptools import setup, find_packages

# Read the contents of README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements from requirements.txt if it exists
try:
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
except FileNotFoundError:
    # Define requirements directly if no requirements.txt file
    requirements = [
        "numpy",
        "matplotlib",
        "deap",
        "numba",
        "loguru", 
        "gymnasium",
        "networkx",
        "seaborn",
        "PyYAML",
        "scipy"
    ]

setup(
    name="agrn",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Artificial Gene Regulatory Networks framework for evolutionary computation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/agrn",  # Update with your repo URL
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",  # Update with your license
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "flake8",
        ],
    },
    include_package_data=True,
    package_data={
        "agrn": ["*.yaml", "*.yml"],
    },
)
