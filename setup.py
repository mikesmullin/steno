#!/usr/bin/env python3
"""Setup script for Steno CLI tool."""

from setuptools import setup, find_packages
from pathlib import Path

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    requirements = []

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    with open(readme_file, encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "A CLI tool for real-time transcription with speaker identification"

setup(
    name="steno",
    version="0.1.0",
    description="Real-time audio transcription with speaker identification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/dnhkng/GLaDOS",
    packages=find_packages(include=["steno*", "lib*"]),
    py_modules=["steno"],  # Include steno.py as a module
    install_requires=requirements,
    python_requires=">=3.11",
    entry_points={
        "console_scripts": [
            "steno=steno:main",  # Creates 'steno' command that calls main() from steno.py
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
