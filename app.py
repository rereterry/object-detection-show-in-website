from flask import Flask, render_template, Response, session
from camera import CameraStream
import cv2
app = Flask(__name__)
import os
import numpy as np
import tensorflow as tf
import sys
import time
from PIL import ImageFont, ImageDraw, Image
import paho.mqtt.client as mqtt
import json as json_dict
# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

class TestView(object):
   
    def __init__(self): 
        
        self.client = mqtt.Client()
        self.client.connect('xxx.xxx.xxx.xxx', 8084)
        # predict 用的資訊
        # This is needed since the notebook is stored in the object_detection folder.
        sys.path.append("..")
        self.cap = None
        # Name of the directory containing the object detection module we're using
        MODEL_NAME = 'inference_graph'

        fontPath = "./TaipeiSansTCBeta-Bold.ttf"
        self.font = ImageFont.truetype(fontPath, 32)

        # Grab path to current working directory
        CWD_PATH = os.getcwd()

        # Path to frozen detection graph .pb file, which contains the model that is used
        # for object detection.
        PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

        # Path to label map file
        PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

        # Number of classes the object detector can identify
        NUM_CLASSES = 20

        ## Load the label map.
        # Label maps map indices to category names, so that when our convolution
        # network predicts `5`, we know that this corresponds to `king`.
        # Here we use internal utility functions, but anything that returns a
        # dictionary mapping integers to appropriate string labels would be fine
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

        for key in self.category_index:
            print ("key: %s , value: %s" % (key, self.category_index[key]))

        # Load the Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=detection_graph)

        # Define input and output tensors (i.e. data) for the object detection classifier

        # Input tensor is the image
        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Output tensors are the detection boxes, scores, and classes
        # Each box represents a part of the image where a particular object was detected
        self.detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represents level of confidence for each of the objects.
        # The score is shown on the result image, together with the class label.
        self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

        # Number of objects detected
        self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    
    class MyEncoder(json_dict.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return super(MyEncoder, self).default(obj)
    
    def mqtt_pub(self, payload):
        topic = 'demo/fish'
        json_out = {}
        json_out['GwID'] = "MiGW-B1-1"
        json_out['TimeStamp'] = round(time.time()*1000)
        json_out['DeviceID'] = "B1F-IPCam"
        json_out['Payload'] = payload
        json_str = json_dict.dumps(json_out, cls=self.MyEncoder)
        self.client.publish(topic , json_str, 0, False)

    
    def predict(self, frame):  
        if frame is None:
            # print("The read queue is None, plz do not predict!")
            return None    
        frame_expanded = np.expand_dims(frame, axis=0)
        #print(frame_expanded.shape)
        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: frame_expanded})

        
        
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes)
            .astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=3,
            skip_labels=True,
            skip_scores=True,
            min_score_thresh=0.60)

        payload = []
        
        for i in range(len(scores[0])):
            if scores[0][i]>0.6:
                # payload design
                temp_list = {}
                # temp_list.setdefault('id', str(i))
                temp_list.setdefault('score', str(scores[0][i]))
                temp_list.setdefault('class', str(classes[0][i]))
                temp_list.setdefault('name', self.category_index[classes[0][i]]['name'])
                temp_list.setdefault('ymin', boxes[0][i][0])
                # print("這是還沒json的座標 ", boxes[j][i][0], " 跟 ", boxes[j][i][1])
                temp_list.setdefault('xmin', boxes[0][i][1])
                temp_list.setdefault('ymax', boxes[0][i][2])
                temp_list.setdefault('xmax', boxes[0][i][3])
                payload.append(temp_list)
        
        self.mqtt_pub(payload)

        convert = cv2.imencode('.jpg', frame)[1].tobytes()
        return convert   
    

    def gen_frame(self):
        """
        Video stream generator
        """
        self.cap = CameraStream().start()

        while self.cap:

            frame = self.cap.read()
            convert = self.predict(frame)
            if convert is None:
                continue

            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + convert + b'\r\n')  # concate frame one by one and show result

    
    def __del__(self):
        try: 
            self.cap.stop()
            self.cap.stream.release()
        except:
            print('probably there\'s no cap yet :(')
        cv2.destroyAllWindows()
        self.client.disconnect()

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(TestView().gen_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)