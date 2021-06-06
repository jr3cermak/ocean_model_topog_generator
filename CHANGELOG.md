# 2021-06-06

 - Add ability to apply old land mask to new bathy when generating h2 field.
 - Add ability to plot using an extent instead of a global view.
 - Add a hook to run ice9 algorithm.
 - Plots are now in landscape with horizontal colorbars.  Changed colormaps.

# 2021-05-25

 - Add software requirements to requirements.txt and add reference inside
   setup.py.
 - The python library cartopy has a system library requirement of proj and
   geos.

# 2021-05-24

 - From MOM6-examples/global; adopt former metadata from existing variables
 - tile string has a default length of 255
 - Compute h2 (which is hstd\*hstd) meters^2 and provide depth=-height.
 - It has been discovered that this repo follows a three stage process:
   - Step 1: Compute hstd, height and wet/dry on a supergrid.
   - Step 2: Fill ponds.
   - Step 3: Provide depth and h2 reduced to a regular grid.
 - Now we can refactor and add to the grid generation tools.
   - Reimplement current hacks using --fix-by-overlap-qh-grid-shift
 - Package dependencies:
   - xarray netCDF4 numpy pandas matplotlib IPython cartopy pkdtree scipy

# 2021-05-23

 - If we specify --super, the entire supergrid is passed for computation of H
   and roughness.  If not, we just pass the vertices but we also keep a
   separate variable with the actual grid.
 - The grid is partitioned into 4 blocks (0-3).  Blocks 0-2 are extended
   i+1 to overlay the zero band left behind when computing roughness.
   Because we are using the vertices, we clip i+1 and j+1 after concatenating
   the resolved fields.  This results in a 1/2 grid shift of the resolved H
   and roughness which may not be optimal.
 - Final coding for clean variable strings in netCDF files.
 - Disable FillValue defaults in netCDF files.
 - Allow specification of an unified output directory.
 - TODO: Fix the 1/2 grid shift in the calculation
 - TODO: Eradicate the zero band between blocks
 - TODO: Understand the roughness computation
 - TODO: Create the land mask file
 - Update .gitignore
 - Add hashes to grid output
 - Proper placement of encoding before writing netCDF file

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
