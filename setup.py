import setuptools

import monosolver as mono

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="monosolver",
    version=mono.__version__,
    author="Jonatan Westholm",
    author_email="jonatanwestholm@gmail.com",
    description="Yet another mathematical programming solver",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jonatanwestholm/monosolver",
    #packages=setuptools.find_packages(),
    packages=['monosolver'],
    #package_dir={'sugarrush.examples': 'sugarrush/examples'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
