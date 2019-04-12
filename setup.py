#! /usr/bin/env python
# coding=utf-8

import codecs
import os

from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

# get __version__ from _version.py
with open(os.path.join('koho', '_version.py')) as f:
    exec(f.read())
VERSION = __version__

setup(name='koho',
      version=VERSION,
      description='Decision Forest C++ library with a scikit-learn compatible Python interface',
      license='New BSD License',
      packages=['koho','koho.sklearn'],
      author='AI Werkstatt(TM) www.aiwerkstatt.com',
      author_email='drh@aiwerkstatt.com',
      maintainer='AI Werkstatt(TM) www.aiwerkstatt.com',
      maintainer_email='drh@aiwerkstatt.com',
      url='https://koho.readthedocs.io/en/latest/',
      download_url='https://github.com/aiwerkstatt/koho',
      long_description=read("README.rst"),
      zip_safe=False,  # the package can run out of an .egg file
      install_requires=[
          'numpy',
          'scipy',
          'scikit-learn'],
      extras_require={
	    'tests': [
        	'pytest',
        	'pytest-cov'],
    	'docs': [
        	'sphinx',
        	'sphinx-gallery',
        	'sphinx_rtd_theme',
        	'numpydoc',
        	'matplotlib']},
      classifiers=[
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Intended Audience :: Science/Research',
          'Natural Language :: English',
          'License :: OSI Approved :: BSD License',
          'Development Status :: 1 - Planning',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX',
          'Operating System :: Unix',
          'Operating System :: MacOS']
      )
