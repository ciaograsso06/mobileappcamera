import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, Response

app = Flask(__name__)

model_path = "ssd_mobilenet_v1_coco_2017_11_17/saved_model"
detect_fn = tf.saved_model.load(model_path)
infer = detect_fn.signatures['serving_default']

LABELS = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light'
}

def gen_frames():
    cap = cv2.VideoCapture(0)  
    
    while True:
        ret, frame = cap.read()
        
        image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis,...]
        
        detections = infer(input_tensor)

        boxes = detections['detection_boxes'][0].numpy()
        classes = detections['detection_classes'][0].numpy().astype(np.int32)
        scores = detections['detection_scores'][0].numpy()

        threshold = 0.5
        for i in range(len(scores)):
            if scores[i] > threshold:
                box = boxes[i]
                ymin, xmin, ymax, xmax = box
                (left, right, top, bottom) = (xmin * frame.shape[1], xmax * frame.shape[1],
                                              ymin * frame.shape[0], ymax * frame.shape[0])

                class_name = LABELS.get(classes[i], 'Unknown')
                label = f'{class_name}: {scores[i]:.2f}'

                frame = cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
                frame = cv2.putText(frame, label, (int(left), int(top - 10)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        ret, buffer = cv2.imencode('.jpg', frame_bgr)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
