#!/usr/bin/env python3

from setuptools import setup

with open('requirements.txt', 'r') as f:
    install_requires = f.read().splitlines()

metadata = dict(
    name='pydradana',
    packages=['pydradana'],
    package_dir={
        'pydradana': 'pydradana',
    },
    author='Chao Gu',
    author_email='guchao.pku@gmail.com',
    maintainer='Chao Gu',
    maintainer_email='guchao.pku@gmail.com',
    description='Analysis Software for Jefferson Lab DRad Experiment.',
    license='GPL-3.0',
    url='https://github.com/asymmetry/pydradana',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Operating System :: MacOS',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Utilities',
    ],
    platforms='Any',
    python_requires='>=3.4',
    install_requires=install_requires,
)

setup(**metadata)
