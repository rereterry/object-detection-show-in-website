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
## How to work
* After building eniveronment, personal model, and fix two python file
* install gunicorn `pip install gunicorn`
* create `wsgi.py`
* commend : gunicorn --threads 5 --workers 1 --bind 0.0.0.0:5000 app:app
* used website server can help us to deal with multiple request
## Reference
* Tensorflow object detection environment build
  * EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10
* close website and close opencv
  * I search from the stackoverflow, but I forgot the url
* opencv use multiple thread and lock to read webcam
  * https://nrsyed.com/2018/07/05/multithreading-with-opencv-python-to-improve-video-processing-performance/
