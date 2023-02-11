from setuptools import find_packages, setup

if __name__ == "__main__":
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
        zip_safe=False,
    )