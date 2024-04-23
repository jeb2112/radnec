from setuptools import setup, find_namespace_packages
import versioneer

setup(name='radnec',
      packages=find_namespace_packages(include=["radnec"]),
      version=versioneer.get_version(),
      cmdclass = versioneer.get_cmdclass(),
      description='RADNEC User Interface',
      url='https://github.com/jeb2112/radnec',
      author='JB',
      install_requires=[
            "scipy",
            "scikit-learn",
            "numpy",
            "scikit-image",
            "SimpleITK",
            "ants",
            "pandas",
            "matplotlib",
            "connected-components-3d",
            "twisted",
            "versioneer",
            "nibabel",
            "screeninfo",
            "pydicom"])
 