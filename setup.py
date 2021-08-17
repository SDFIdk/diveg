import os
from setuptools import (
    setup,
    find_packages,
)

with open(os.path.join('src', 'diveg', 'version.py')) as f:
    exec(f.read())


setup(
    name="diveg",
    version=__version__,
    description="Danish InSAR Velocity and Error Grid (DIVEG)",
    url="https://github.com/Kortforsyningen/diveg",
    author="Joachim Mortensen (SDFE, GRF) <joamo@sdfe.dk>",
    author_email="grf@sdfe.dk",
    license="MIT",

    packages=find_packages(where='src'),
    package_dir={'': 'src'},

    install_requires=[
        # 'rich',
        # 'click',
        # 'numpy',
        # 'shapely',
        # 'geopandas',
        # 'rasterio',
        # 'matplotlib',
        # 'seaborn',
    ],

    # setup_requires=[
    #     'pytest-runner',
    # ],
    # tests_require=[
    #     'pytest',
    #     'pytest-cov',
    # ],
    # test_suite="pytest",

    entry_points={
        'console_scripts': [
            'diveg = diveg.cli:main',
        ],
    },
)
