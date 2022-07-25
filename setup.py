from distutils.core import setup
from setuptools import find_packages

setup(
    name='roboverse',
    packages=find_packages(),
    include_package_data=True,
    exclude_package_data={
        'mypkg': ['*[0-9].urdf']
    }
)
