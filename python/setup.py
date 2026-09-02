"""
Setup script for OpenAHI Python package
"""

from setuptools import setup, find_packages
import os

# Read version
with open(os.path.join("openahi", "__init__.py"), "r") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"').strip("'")
            break
    else:
        version = "0.1.0"

# Read requirements
with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="openahi",
    version=version,
    description="OpenAHI - Open Artificial Hyper Intelligence",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="ZoDev",
    author_email="",
    url="https://github.com/zoDevLang/OpenAHi",
    project_urls={
        "Source": "https://github.com/zoDevLang/OpenAHi",
        "Documentation": "https://github.com/zoDevLang/OpenAHi/docs",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.0",
            "types-requests>=2.0",
        ],
        "gpu": [
            "torch>=2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "openahi-python=openahi.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=[
        "openahi",
        "ai",
        "artificial-intelligence",
        "machine-learning",
        "transformer",
        "language-model",
        "nlp",
    ],
    package_data={
        "openahi": [
            "py.typed",
            "*.json",
            "*.yaml",
            "*.yml",
        ],
    },
    zip_safe=False,
)
