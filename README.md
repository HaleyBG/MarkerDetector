# MarkerDetector
MarkerDetector is a novel Python-based toolkit for 
detecting fiducial markers in cryo-electron tomography
(cryo-ET).
## Pre-required software
    python3: https://www.python.org/downloads/
    Numpy: pip/conda install numpy
    Opencv: pip install opencv-python
## Usage
Prepare the dataset in MRC format, and run the entry program "main.py".

    python3 main.py <mrc_file_path> <Index of projection> [--dense 0] [--scale 2] [--threshold_ncc 0.55] [--save_all_figure 0]
    
The program will prompt you to enter the required information.

To test the performance of MarkerDetector more quickly, we provide a test script for fiducial markers detection as "test.py". 
    python3 test.py
