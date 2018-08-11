# Performance of various Machine Learning Algorithms on Electrical Impedance Tomography Images

# Overview

This report provides an overview of current practice of Electrical Impedance Tomography (EIT), its imaging and use-cases. Electrical Impedance Tomography is a non-invasive type of medical imaging. These advances are improving our capacity to treat and even prevent cancers. The full implications of the subject remain to be explored. Examples of research techniques used in this project are detailed.  

# Dependencies

1.	Python 3.x 
2.	Numpy 
3.	Scipy
4.	Pandas
5.	Matplotlib
6.	Sci-kit learn
7.  OpenCV Python

# Important Files and Folders

```
eit
│   README.md   
│
└───assets
│      datasets - contains datasets in csv
│      eit_images - generated images
│   
└───classification
│      *.ipynb - Classification ML algorithms
│      results.ipynb - Final results and graphs
│   
└───docs
│      documentation and reports
│
└───main
       eit_analysis.py
       eit_classify.py
       eit_dataset.py 
```

# Usage

1.	```generate_image.py```
    * Generates 1000 images
    * Linspace and Meshgrid are numpy methods 

2.	```read_img.py```
    * Reads an image into code
    * ```matrix``` contains three dimensional array of image
    * ```img``` contains three dimensional array of image - image import
    * ```grayscale``` contains two dimensional array of image
    * ```x``` contains x dimension of image
    * ```y``` contains y dimension of image

3.	```eit.py```
    * Plots a contour graphs
    * Adds list of colors and be saved as an image

4.	```eit_dataset.py```
    * Generates dataset without labels - creates file ```eit.csv```
    * ```intensity_range_strings``` contains ranges of intensities
    * ```classify_dict``` contains dataset in the form of dictionary
    * ```df``` contains final file to be converted to csv

5.	```eit_analysis.py```
    * Assigns targets 1 or 0 and created another dataset - creates file ```eit_data.csv```
    * ```target``` contains target array 0s and 1s

6.	```eit_classify.py```
    * Generate classification plots - generates ```eit_contour_plot.csv```
    * ```autolabel``` - function labels bar graphs

7.	```<*>.ipynb``` - All classification ML algorithms - '<*>' means all files

# Results

| No |  Algorithms	                    | Accuracy (%)  |
| ---|:--------------------------------:| -------------:|
| 1	 |  K Nearest Neighbours	        | 93.6%         |
| 2	 |  Decision Tree Classification	| 98.8%         |
| 3	 |  Kernel Support Vector Machines	| 94%           |
| 4	 |  Logistic Regression	            | 88.4%         |
| 5	 |  Naive Bayes	                    | 92.4%         |
| 6	 |  Random Forest Classification	| 99.2%         |
| 7	 |  Support Vector Machines	        | 88%           |


# Credits

[Vinod Agrawal](https://in.linkedin.com/in/vinod-agrawal-8020488)

CTO, Faststream Technologies

# License

Copyright(c) 2018, [Faststream Technologies](https://www.faststreamtech.com)

=======
Author: 
* [Sudhanva Narayana](https://www.sudhanva.in)
* [Shreyas S](https://www.shreyas.im)
