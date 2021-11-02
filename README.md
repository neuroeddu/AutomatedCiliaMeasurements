# AutomatedCiliaMeasurements
Pipeline to measure cilia, cell nucleis, and which cilia are close to which cell nucleis from microscopy images using CellProfiler and Python.

# Components
1) CellProfiler pipeline -- This collects the number of cell nuclei, cilia, and centrioles, as well as where they are in the image (their centres) & their sizes/shapes. It also saves the trace of nuclei/cilia/centrioles seen in the CP pipeline.

2) pyproject.toml -- Package manager for scripts  

3) preprocess_image_names.py -- Preprocess image names from CellProfiler pipeline output into standard format for scripts 

4) size_filter.py -- Filters out centrioles/nuclei that are too big

5) centertocenter.py -- Maps nuclei to centrioles in a 1:1 or 2 ratio, and centrioles to cilia in a 1:1 ratio

6) label_cp_im.py -- Visualizer for output of the CellProfiler pipeline

7) visualize.py -- Visualizer for output of the centertocenter.py script  

8) calc.py -- Calculates mean and standard deviation of area, length, and width columns 

9) centrioles_without_cells.py -- Visualizes the centrioles that were not assigned to a cell, and where they are in relation to the cell they were the closest to

10) merge_cilia.py -- Merges cilia into clusters according to DBSCAN algorithm
