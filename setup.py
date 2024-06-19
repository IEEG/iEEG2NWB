# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
    name='iEEG2NWB',
    description='Python scripts for converting ieeg data in edf and TDT formats to NWB',
    version='0.0.1',
    author='Noah Markowitz, Stephan Bickel',
    include_package_data=True,
    packages=find_packages(),
    package_dir={'ieeg2nwb': 'ieeg2nwb'},
    entry_points={
        'console_scripts': [
            'ieeg2nwb=ieeg2nwb.ieeg2nwb:cmnd_line_parser'
            ]
    }
)
