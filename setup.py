#! /usr/bin/python

from setuptools import setup, find_packages

setup(
    name="word_embeddings_alignment",
    version='0.0.1',
    description='Calculates local pairwise alignment using word embeddings representation',
    url='https://github.com/wojciech-galan/word_embeddings_alignment',
    author='Wojciech Gałan',
    license='GNU GPL v3.0',
    install_requires=[
        'blosum',
        'numpy'
    ],
    packages=find_packages(),
    classifiers=[
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 2.8'
    ],
    entry_points = {
        'console_scripts':[
            'word_embeddings_alignment = word_embeddings_alignment.__main__:main'
        ]

    },
    include_package_data=True
)