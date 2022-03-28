# RoughOceanQGEnergetics

This repository contains codes and processed datasets for a manuscript entitled **"Instability of a surface jet over rough topography"**, by A. Palóczy and J. H. LaCasce, submitted to the Journal of Physical Oceanography. This [Jupyter notebook](https://nbviewer.jupyter.org/github/apaloczy/RoughOceanQGEnergetics/blob/main/index.ipynb) provides an overview of the contents.

The directory `plot_figs/` contains the Python codes used to produce the figures in the manuscript (Figures 1-10). The codes depend on the data files in the `simulations/` and `data_reproduce_figs/` directories.

Files in the `simulations/` directory are generated by running the codes in the `code_simulations/`. These codes simulate a two-layer quasigeostrophic system linearized about a mean zonal jet in a doubly-periodic domain over rough bottom topography, implemented with the [`FourierFlows.jl`](https://github.com/FourierFlows/FourierFlows.jl) package.

## Authors
* [André Palóczy](https://www.mn.uio.no/geo/english/people/aca/metos/andrpalo/index.html) (<a.p.filho@geo.uio.no>)
* [Joseph H. LaCasce](https://www.mn.uio.no/geo/english/people/aca/metos/josepl/) (<j.h.lacasce@geo.uio.no>)
