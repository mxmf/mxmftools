import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mxmftools",
    version="0.0.1",
    author="mxmf",
    author_email="mxmf521@outlook.com",
    description="some scripts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mxmf/mxmf-tools",
    install_requires=[
        "matplotlib",
        "scipy",
        "colorcet",
        "click-bash4.2-completion",
        "h5py",
        "lxml",
        "click >= 8.1.0"
        
    ],
    packages=setuptools.find_packages(),
    package_data={
        "mxmftools": ["matplotlibrc"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "mxmf = mxmftools:main",
        ],
    },
)
