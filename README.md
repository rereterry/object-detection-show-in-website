# object-detection-show-in-website
This project will present how to upload detection consequence to website and close website and opencv correctly
## Environment
* cuda (choose version by you os and tensorflow version)
* cudnn (choose version by you os and tensorflow version)
* gunicorn
## Python library
* tensorflow-gpu 1.14
* openCV
* labelImg
## File explain
* app.py: It has tensorflow object detection class ,mqtt funciton and flask framework
* camera.py: Used lock to deal with video frame
## Reference
* Tensorflow object detection environment build
  * EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10
* close website and close opencv
  * I search from the stackoverflow, but I forgot the url
* opencv use multiple thread and lock to read webcam
  * https://nrsyed.com/2018/07/05/multithreading-with-opencv-python-to-improve-video-processing-performance/
