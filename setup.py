from setuptools import setup, find_namespace_packages

setup(name='blast',
      packages=find_namespace_packages(include=["blast"]),
      version='0.1',
      description='BLAST User Interface',
      url='https://github.com/jeb2112/blast',
      author='JB',
      install_requires=[
            "scipy",
            "scikit-learn",
            "numpy",
            "scikit-image",
            "SimpleITK",
            "pandas",
            "matplotlib",
            "connected-components-3d",
            "twisted"      ])
 