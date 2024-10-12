# -*- coding: utf-8 -*-
from __future__ import annotations

from setuptools import find_packages, setup

def readme():
    with open("README.md") as f:
        return f.read()


setup(
    name="bnMediaScribe",
    version="0.1.0",
    description="Media Content Generator",
    packages=find_packages("src"),
    package_dir={"": "src"},
    long_description=readme(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Developers",
    ],
    keywords="llm image video ai generation",
    url="https://github.com/ahestevenz/media-scribe",
    author="Ariel Hernandez <ahestevenz@bleiben.ar>",
    author_email="ahestevenz@bleiben.ar",
    license="Proprietary",
    install_requires=[
        "numpy==2.1.1",
        "torch==2.4.1",
        "torchaudio==2.4.1",
        "torchvision==0.19.1",
        "diffusers==0.30.3",
        "transformers==4.45.1",
        "pydantic==2.7.4",
        "pillow==10.4.0",
        "pre_commit==4.0.0",
        "loguru==0.7.2",
        "accelerate==1.0.0",
    ],
    test_suite="nose.collector",
    tests_require=["nose"],
    entry_points={
        "console_scripts": [
            "bn-run-scribe-text=bnMediaScribe.scripts.run_scribe_text:main",
            "bn-run-scribe-image=bnMediaScribe.scripts.run_scribe_image:main",
            "bn-run-scribe-image2image=bnMediaScribe.scripts.run_scribe_image2image:main",
        ],
    },
    include_package_data=True,
    zip_safe=True,
)
