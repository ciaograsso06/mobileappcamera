# Real-Time Object Detection with Flask and TensorFlow

This project is a Flask web application that performs real-time object detection using a pre-trained SSD MobileNet model. The application captures video from your webcam, processes it to detect objects, and displays the results in your web browser.

## Features

- Real-time object detection using TensorFlow.
- Displays detected objects with bounding boxes and confidence scores.
- Simple web interface to view the video feed.

## Requirements

- Python 3.x
- Flask
- OpenCV
- NumPy
- TensorFlow

## Installation

1. **Clone the repository**:

```bash
git clone https://github.com/ciaograsso06/mobileappcamera.git


cd mobileappcamera
```


2. **Create a virtual environment** (optional but recommended):

```bash
python -m venv venv


source venv/bin/activate
```


3. **Install the required packages**:

```bash
pip install Flask opencv-python numpy tensorflow

```


4. **Download the SSD MobileNet model**:

You need to download the pre-trained SSD MobileNet model. You can find it [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tutorials/packaging_for_production.md). Extract the model files to a directory named `ssd_mobilenet_v1_coco_2017_11_17`.

5. **Set up the model path**:

Ensure that your model path in the code points to the correct location of the saved model:

```bash
model_path = "ssd_mobilenet_v1_coco_2017_11_17/saved_model"


```


## Usage

1. **Run the application**:

```bash
python app.py
```


2. **Open your web browser** and navigate to `http://127.0.0.1:5000/` to view the video feed with object detection.

## Code Explanation

- The application uses Flask to create a web server and serve HTML content.
- OpenCV captures video from the webcam.
- TensorFlow loads a pre-trained object detection model and processes frames to detect objects.
- Detected objects are displayed with bounding boxes and labels on the video feed.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any improvements or bugs.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [TensorFlow](https://www.tensorflow.org/) for providing powerful machine learning tools.
- [Flask](https://flask.palletsprojects.com/) for creating lightweight web applications.
