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

* ```read_img.py``` - Read an image into code
* ```generate_images.py``` - Generate images
* ```eit_dataset.py``` - Generate dataset
* ```eit_analysis.py``` - Assign target values
* ```eit_classify.py``` - Create contour plots and classify images

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

Author: [Sudhanva Narayana](https://www.sudhanva.in)