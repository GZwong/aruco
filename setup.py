"""
In some IDEs (VScode), running files as standalone scripts may result in project-specific import errors. 

This issue can be solved by installing the project as a package in editable mode - project-specific packages now appear
as if they are a package within the virtual environment. 

To do this one of the commands below, depending on which environment you are in
>>> pip install --editable .
>>> conda develop .

"""

from setuptools import setup, find_packages

setup(name='dronekitpy', version='1.0', packages=find_packages())
