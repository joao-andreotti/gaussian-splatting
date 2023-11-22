from setuptools import setup, find_packages

setup(
    name='gaussian_splatting',
    version='0.1.0',
    description='Gaussian Splatting',
    author='',
    author_email='',
    url='https://github.com/joao-andreotti/gaussian-splatting',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=[],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
