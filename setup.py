from setuptools import find_packages, setup

with open('requirements.txt') as fid:
    INSTALL_REQUIRES = [line.strip() for line in fid.readlines() if line]

setup(
    name="sail",
    version="0.2.0",
    description="Python package for streaming data and incremental learning",
    url="https://github.com/IBM/sail",
    author="Seshu Tirupathi, Dhaval Salwala, Shivani Tomar",
    author_email="seshutir@ie.ibm.com, dhaval.vinodbhai.salwala@ibm.com",
    license="MIT",
    python_requires="==3.10.*",
    packages=find_packages(),
    install_requires=[
        "numpy==1.23.*",
        "scipy==1.10.*",
        "pandas==1.5.*",
        "scikit-learn==1.2.*",
        "scikit-multiflow==0.5.3",
        "dill",
        "matplotlib",
        "logzero",
        "setuptools",
        "ipykernel",
    ],
    extras_require={
        "tensorflow": ["tensorflow==2.10.1", "scikeras==0.10.0"],
        "pytorch": ["torch==1.12.0", "skorch==0.12.1"],
        "river": ["river==0.15.*"],
        "ray": ["ray==2.2.*"],
        "dev": ["black", "pylint"],
        "all": [
            "tensorflow==2.10.1",
            "scikeras==0.10.0",
            "torch==1.12.0",
            "skorch==0.12.1",
            "river==0.14.*",
            "ray==2.2.*",
        ],
    },
    tests_require=["pytest", "flake8"],
    zip_safe=False,
)