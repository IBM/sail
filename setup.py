from setuptools import find_packages, setup
import sail

if __name__ == "__main__":
    setup(
        name="sail",
        version=sail.__version__,
        description="Python package for streaming data and incremental learning",
        long_description=open("README.md").read(),
        long_description_content_type="text/markdown",
        url="https://github.com/IBM/sail",
        author="Seshu Tirupathi, Dhaval Salwala, Shivani Tomar",
        author_email="seshutir@ie.ibm.com, dhaval.vinodbhai.salwala@ibm.com",
        license="MIT",
        python_requires="==3.10.*",
        packages=find_packages(),
        zip_safe=False,
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.10",
        ],
    )
