# 2021-05-21

  - zeros are produced consistently; need some other workaround
  - Add a max\_mb option so we can change it dynamically
  - Change routine to work on regular grid instead of supergrid
  - Add --super option to operate on supergrid
  - Not using the supergrid will yield imperfect results as the refinement
    routine GMesh expects lat/lon coordinates on the cell corners which is
    why the supergrid is being used.
  - Remove clipping of supergrid; moved to --clip for use with global grids
  - Allow plot argument to specify filename

# 2021-05-20

  - TODO: Add one more coarsen step to get back to regular grid from supergrid
  - conversion complete; can reduce memory requirement to work on 10GB on a VM
  - slow promotion of numpy calculations to xarray
  - netCDF4 module required for netcdf version 4 files
  - Removed 'import imp': depricated and unused
  - Convert netCDF4 references to xarray
