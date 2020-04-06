#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script"""
import os
from setuptools import setup, find_packages

setup(author="Dih5",
      author_email='dihedralfive@gmail.com',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Natural Language :: English',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
      ],
      description='Compartmental modelling Python package',
      extras_require={
          "docs": ["nbsphinx", "sphinx-rtd-theme", "IPython"],
          "test": ["pytest"],
      },
      keywords=[],
      name='comod',
      packages=find_packages(include=['comod'], exclude=["demos", "tests", "docs"]),
      install_requires=["numpy", "scipy", "pandas", "python-igraph"],
      url='https://github.com/dih5/comod',
      version='0.1.1',

      )
