from setuptools import find_packages, setup

setup(name='sail',
      version='0.1.0',
      description='Python package for streaming data and incremental learning',
      url='https://github.com/IBM/sail',
      author='Seshu Tirupathi, Mark Purcell',
      author_email='seshutir@ie.ibm.com',
      license='MIT',
      python_requires='>=3.6',
      install_requires=[
          "numpy>=1.21.0",
          "scipy>=1.5.2",
          "river>=0.7.0",
          "pandas>=1.3.0",
          "ray>=1.4.1",
          "scikit-learn>=0.24.2",
          "setuptools"
      ],
      extras_require={
          "keras": ["keras", "scikeras"],
          "tensorflow": ["tensorflow", "tensorflow_addons"],
          "pytorch": ["torch", "torchvision"],
          "all": [
              "keras",
              "tensorflow",
              "torch"
          ]
      },
      tests_require=[
          'pytest',
          'flake8'
      ],
      packages=find_packages(),
      zip_safe=False)
