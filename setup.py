#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script"""
import os
from setuptools import setup, find_packages

# Get the long description from the README file
with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.md')) as f:
    long_description = f.read()

extras = ["sympy",  # Symbolic manipulation to find fixed points
          "network2tikz",  # Exports graphs to tikz
          "pdf2image",  # Show tikz generated graphs in notebooks
          "python-igraph",  # Alternative plot method
          ]

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
          "extras": extras,
          "docs": ["nbsphinx", "sphinx-rtd-theme", "IPython"] + extras,
          "test": ["pytest"] + extras,
      },
      keywords=[],
      long_description=long_description,
      long_description_content_type='text/markdown',
      name='comod',
      packages=find_packages(include=['comod'], exclude=["demos", "tests", "docs"]),
      install_requires=["numpy", "scipy", "pandas"],
      url='https://github.com/dih5/comod',
      version='0.1.1',

      )
