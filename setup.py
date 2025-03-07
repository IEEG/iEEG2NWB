# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
    name='iEEG2NWB',
    description='Python scripts for converting ieeg data in edf and TDT formats to NWB',
    version='0.0.1',
    author='Noah Markowitz, Stephan Bickel',
    author_email='',  # Add your email if you want
    url='https://github.com/noahmarkowitz/iEEG2NWB',
    packages=find_packages(),
    package_dir={'ieeg2nwb': 'ieeg2nwb'},
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'pandas',
        'PyYAML',
        'pynwb',
        'pymatreader',
        'PyQt5',
    ],
    extras_require={
        'dev': [
            'pytest',
            'pytest-cov',
            'flake8',
        ],
    },
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'ieeg2nwb=ieeg2nwb.ieeg2nwb:cmnd_line_parser'
            ]
    }
)
