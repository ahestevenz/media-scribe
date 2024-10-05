from __future__ import annotations

from setuptools import find_packages
from setuptools import setup


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
    url="https://github.com/ahestevenz/bushfires-object-detection",
    author="Ariel Hernandez <ahestevenz@bleiben.ar>",
    author_email="ahestevenz@bleiben.ar",
    license="Proprietary",
    install_requires=[
        "numpy==2.1.1",
    ],
    test_suite="nose.collector",
    tests_require=["nose"],
    entry_points={
        "console_scripts": [
            "bn-run-train=bnMediaScribe.scripts.run_scribe:main",
        ],
    },
    include_package_data=True,
    zip_safe=True,
)
