from setuptools import setup, find_packages

setup(
    name="qu_los_sim", 
    version="0.1.0",
    author="Anna Ordog",
    author_email="anna.ordog@ubc.ca",
    description="A package for simulating Stokes Q and U for various Faraday rotation models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/aordog/fd-sims",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "RMtools_1D",
        "RMutils",
    ],
)