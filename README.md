# Social_Distance_Monitoring
This code helps in detecting whether the people are maintaining social distance or not.

Below are some descriptions:

Centroidtracker.py is a file that helps in tracking any object in a frame.

person_detection_image.py is a file that detects person in image file

person_detection_video.py is a file that detects person in video file.

person_tracking.py is a file that detects person and keeps tracking them in the frame. It assigns a unique ID to each detected person.

social_distancing is a file that monitors social distance between the persons. If it is less than a threshold value, we display bounding box in red otherwise green.

Some models files are also there in model_files folder which are used in all the python files.
Sample videos are there in video_files folder.
Samples images are there in image_files folder.

OpenCV is used for all sorts of image and video analysis, like facial recognition and detection, license plate reading, photo editing, advanced robotic vision, optical character recognition, and a whole lot more. 

NumPy is a package in Python used for Scientific Computing. NumPy package is used to perform different operations. The ndarray (NumPy Array) is a multidimensional array used to store values of same datatype. These arrays are indexed just like Sequences, starts with zero.

SciPy is a library that uses NumPy for more mathematical functions. SciPy uses NumPy arrays as the basic data structure, and comes with modules for various commonly used tasks in scientific programming, including linear algebra, integration (calculus), ordinary differential equation solving, and signal processing.

Scipy is a series of convenience functions to make basic image processing functions such as translation, rotation, resizing, skeletonization, displaying Matplotlib images, sorting contours, detecting edges, and much more.

Dlib is a landmark's facial detector with pre-trained models, the dlib is used to estimate the location of 68 coordinates (x, y) that map the facial points on a person's face.

TensorFlow is a Python library for fast numerical computing created and released by Google. It is a foundation library that can be used to create Deep Learning models directly or by using wrapper libraries that simplify the process built on top of TensorFlow.

Somtimes it happens that we are having only one object in frame but the bounding boxes are more than one which is nothing but noise. To avoid that we use non-max supression algorithm. It checks the overlapping bounding boxes and merge it into one single box and output image contains no noise.

Centroid tracker code is used to track the particular object in the frame. The object is assigned a unique id and it is assigned to that particular object till it is in the frame. Once the object disappears, it assigns a new unique id to another object.

In person_tracking.py, maxDisappeard value is basically the no. of frames we want our tracker to keep waiting for ur objects. If our object moves out of the frame, our tracker will keep tracking for the particular frames which we are defining so that if the object comes back it will assign the same object id. This value depends from application to application.

To calculate the social distancing between two person, we are going to claculate the centroid of 1st person and centroid of 2nd person and then calculate the distance between the centroid1 and centroid2, If the distance is less than the treshold value means they are not following social distancing else they are following. 

