# Version 0.0.9
  * preprocess a group of dicom directories 

  ### Version 0.0.8

  * incorporate regular T2 contrast for T2 hyperintensity in place of T1 post-contrast
  * loading from a dicom directory
  * added resample, register and brain extraction to the preprocessing

  ### Version 0.0.7

  * options for loading an individual case directory, or individual T1,FLAIR files
  * select kmeans cluster that is closest to the origin for background stats

  ### Version 0.0.6

  * larger UI default size and draggable resize
  * added window/level contrast button
  * added 3d crosshair overlay in orthogonal slices
  * added scrollbars for slice scrolling and removed right mouse button
  * added mousewheel and keyboard Up/Down for slice scrolling
  * restored select ROI button

  ### Version 0.0.5

  * fixed slice display convention
  * update for loading datasets of different matrix dimension
  * handle no files found error

  ### Version 0.0.4

  * contour overlay option
  * slice number and window/level overlay
  * parallel ROI selection process for ET and T2 hyperintensity
  * sliders update when switching back and forth between ROI's and BLAST layers
  * switched from sitk to nibabel for nifti coordinates 

  ### Version 0.0.3

  * coronal and sagittal views
  * .whl installation file
  * automatic versioning from github tag
  * better python analogue for matlab imfill
  * 1-based ROI numbering
