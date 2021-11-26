Course Name: Digital Image Processing and Analysis
CS517, IIT Ropar

Instructor: Dr. Deepti Bathula

Group Members:
1. Apurv Rathore (2019CSB1077)
2. Vasu Bansal (2019CSB1130)

Project Title: Analysis and Implementation of Seam Carving Algorithm

Date of Submission: 26th November 2021

------------------------------------------------------------------------------------------------------
1. Basic requirements to run the code
Windows/Linux machine with Python3 installed.
After that install the following libraries
You can install it via the requirements.txt file
>> pip install -r requirements.txt

In case it does not work, then one can install these libraries manually as well:
>> pip install numpy
>> pip install opencv-python
>> pip install argparse
>> pip install numba
>> pip install scipy

2. How to run the python code:
Open a terminal and type the command
>> python3 image_carving.py -im <input_image_path> -out <output_image_path> [-dy DY] [-dx DX] [-vis]

For example:
python image_carving.py -im input_file.jpg -out output_file.jpg -dy 100 -dx -100 -vis


3. Understanding the arguments
This program can resize the input image without loosing any of the a main objects of the image
main object size saame
increase or decrease Number of rows or column


dx: Number of vertical seams to add (if positive) or subtract (if negative). The default value is 0.
dy: Number of horizontal seams to add (if positive) or subtract (if negative). The default value is 0.
