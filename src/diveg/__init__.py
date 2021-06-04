"""
Danish InSAR Velocity and Error Grid (DIVEG)

A tool for building a grid layer aggregating the points from an InSARS dataset.

The dataset is assumed to have the following properties:

Layer: 2D

Components:

|    Field     |                                                                             Description                                                                              |
|--------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| CODE         | Identification code of measurement points obtained by decomposition, positioned in the centre of the decomposition cell.                                             |
| VEL_V/_E     | Vertical/ East-West displacement rate [mm/year]. Positive values correspond to upward/eastward movements. Negative values correspond to downward/westward movements. |
| V_STDEV_V/_E | Vertical/Horizontal East-West displacement rate standard deviation [mm/year].                                                                                        |
| Dyyyymmdd    | Fields containing the temporal evolution of the Vertical/Horizontal East-West displacements. Displacement values are expressed in [mm].                              |

Source: Table 15: Description of the fields contained in the database of Vertical and East-West component vector, TRE ALTAMIRA report

"""
