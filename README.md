# Face3D
Prototype system for 3D Face Recognition built by Bas Stottelaar and Jeroen Senden for the course Introduction to Biometrics. Written in Python, applicable to the FRGC 3D data set.

## Features
The following algorithms have been implemented:

### Normalization
* Smoothing
* Interpolation
* Cropping
* Zooming
* Key point extraction
* Rotation

### Feature extraction
Histogram based, as proposed by Zhou et al. See http://www.3dface.org/files/papers/zhou-EG08-histogram-face-rec.pdf for more information.

### Distance Metrics
* Method 0: Euclidean Distance with threshold 0.9
* Method 1: City Block Distance with threshold 0.9
* Method 2: Sample Correlation Coefficient with threshold 0.9

## Installation
The following dependencies are expected to be on your system

* Python 2.7 (version 2.6 should work)
* NumPy 1.6
* SciPy 0.11
* Matplotlib 1.2
* SciKit-learn 0.12

In case of missing dependencies, they can be installed via the Python Packet Manager, via the command `pip install <name>`. 

## Quick Start
A few examples to get started:

* `python Face3D.py --enroll /path/to/abs/files --auto-id` &mdash; Enrolls all files in the folder and determines person identification based on filename.
* `python Face3D.py --authenticate /path/to/file.abs` &mdash; Authenticate given file against the enrolled images. Will output the matches with scores.
* `python Face3D.py --reevaluate --parameters N=48,K=12` &mdash; Reevaluate the data set with the given parameters. Does not save data, but you could visualize something with this data.

## Usage
Face3D is a commandline only application. Start a terminal and navigate to this directory where Face3D is extracted. Start the application with the command `python Face3D.py`.

### General
* `python Face3D.py --help` &mdash; Show help.
* `python Face3D.py --parameters` &mdash; Comma seperated key-value parameters for the algorithms. Defaults (and only parameters supported) are `N=67,K=12`.
* `python Face3D.py --database` &mdash; Specify the Face3D Database to work on. Default is `database.db`. You need to specify this option each time if you would like to use another database for operations below.

### Face management
* `python Face3D.py --enroll <file | directory> --person-id <id>|--auto-id` &mdash; Enroll a single file or a complete directory to the Face3D Database. Multiple threads will be spawned in case of multiple files. You have to specify a person ID. In case of auto ID, it will be derrived from the `*.abs` filename (xxxxxd123.abs). This process can take up to 15 minutes for 350+ faces on a Intel Core i7. If a face has already been enrolled, it will notify the user. Simply delete the database file to start over.
* `python Face3D.py --authenticate <file>` &mdash; Match a given face to a face in the database.
* `python Face3D.py --reevaluate` &mdash; Reevaluate the faces with another set of parameters. Works only for feature extraction and other calculations after feature extraction. This comes in handy when evaluating different parameters.

### Visualization & Statistics
* `python Face3D.py --depth-map <output.pdf> [--with-key-points]` &mdash; Write a 3D depth map of enrolled faces to a PDF file, with or without key points.
* `python Face3D.py --feature-map <output.pdf>` &mdash; Write a feature map of enrolled faces to a PDF file. 
* `python Face3D.py --similarity-matrix <output.html>` &mdash; Write a similarity matrix to a HTML file.
* `python Face3D.py --roc-curve <output.pdf>` &mdash; Write a ROC curve to a HTML file.

## Source code
The main application logic is defined in `Face3D.py`. The rest of the code is stored in the folder `face3d/`.

One important file is `face3d/algorithms.py`. Here are all the algorithms programmed that are used for smoothing, interpolating, finding key points, cropping, feature extracting. Dependencies are `face3d/absfile.py` and `face3d/face.py`. The first reads `*.abs` files into memory and the second one is a wrapper for the data and handles views and compression. 

On of the two files left is `face3d/database.py`, a wrapper for an SQLite3 database file. It reads and writes faces and features. Last but not least is `face3d/utils.py` as a place for common used methods.

## Licence
See the LICENCE file.