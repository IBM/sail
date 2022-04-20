from setuptools import find_packages, setup

setup(
    name="sail",
    version="0.1.1",
    description="Python package for streaming data and incremental learning",
    url="https://github.com/IBM/sail",
    author="Seshu Tirupathi, Dhaval Salwala, Shivani Tomar",
    author_email="seshutir@ie.ibm.com, dhaval.vinodbhai.salwala@ibm.com",
    license="MIT",
    python_requires=">=3.6",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.5.2",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.2",
        "scikit-multiflow==0.5.3",
        "matplotlib",
        "setuptools",
        "ipykernel",
    ],
    extras_require={
        "tensorflow": ["tensorflow==2.8.0", "scikeras==0.6.1"],
        "pytorch": ["torch==1.10.2", "skorch==0.11.0"],
        "river": ["river==0.10.1"],
        "ray": ["ray==1.11.0"],
        "all": [
            "tensorflow==2.8.0",
            "scikeras==0.6.1",
            "torch==1.10.2",
            "skorch==0.11.0",
            "river==0.10.1",
            "ray==1.11.0",
        ],
    },
    tests_require=["pytest", "flake8"],
    zip_safe=False,
)
