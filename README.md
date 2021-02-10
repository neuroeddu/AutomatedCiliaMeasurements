# AutomatedCiliaMeasurements
Pipeline to measure cilia, cells, and which cilia are close to which cells from microscopy images using CellProfiler and Python.

# Components
1) CellProfiler pipeline -- This collects the measurements of cells and cilia, as well as where they are in the image (their centres)

2) Python script centertocenter.py -- This finds out which cilia is closest to which cell given the above pipeline, assuming one cell:one cilia ratio and a threshold length away they can be (in pixels) 

3) Python script visualize.py -- This takes saved images of cells and cilia from the pipeline, and numbers them based on their centres in order to visualize the pipelines.
