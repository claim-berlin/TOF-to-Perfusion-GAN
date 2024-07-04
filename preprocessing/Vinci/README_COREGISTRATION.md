### VINCI coregistration Readme
# 2022
## Workflow

- Scripts are used for coregistration of perfusion parameter maps to TOF
- first copy script into vinci binary folder then execute it there
- hdb_params.xml is a parameter file for the MMM method used by VINCI in coregistration 
- First DSC-Source is coreg to TOF and then perfusion parameter maps coreg to DSC-Source
- additionally slice number of images and size is standardized over database (peg = 127 slices and hdb = 98 slices with both 256x256 size)