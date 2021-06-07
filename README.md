# [Danish InSAR Velocity and Error Grid (DIVEG)](https://github.com/Kortforsyningen/diveg)

by [Joachim](joamo@sdfe.dk)

The goal of this package is to provide an easy-to-use command-line tool that takes a geopackage file with InSAR-velocity data as input and produces geotiff-raster files with points gridded and aggregated in the desired grid resolution.

## User installation

Requirements

*   Access to the data that the package is made for.
*   Access to [this repository](https://github.com/Kortforsyningen/diveg) on GitHub.
*   [`conda`](https://docs.conda.io/en/latest/miniconda.html) | [repo](https://repo.anaconda.com/miniconda/)
*   [`git`](https://git-scm.com/)

With the requirements, run the following commands in your environment:

```sh
cd your/git/repos
# As user
git clone https://github.com/Kortforsyningen/diveg.git
# As developer
git clone git@github.com:Kortforsyningen/diveg.git

# Install the environment
sh conda-setup.sh
```

## Usage example

Example: Create a .tif file with the default settings:

```
(diveg) $ ls
# (nothing to begin with)
(diveg) $ diveg your/data/insar.gpkg
# (...)
(diveg) $ ls
# diveg_output
(diveg) $ ls diveg_output
# insar_grid_mean_2400x2400.tif    insar_grid_std_2400x2400.tif
```

More examples to come, when the CLI API is finished.
