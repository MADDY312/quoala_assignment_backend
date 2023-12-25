from flask import Flask,request, jsonify
from io import BytesIO
import base64
import cv2
import os
import tensorflow as tf
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from flask_cors import CORS
from utils import label_map_util
from skimage.filters import threshold_otsu
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line
from skimage.transform import rotate
from scipy.ndimage import gaussian_filter
from utils import visualization_utils as vis_util

app = Flask(__name__)
CORS(app)
def auto_crop_id_card(image_path):
   # Name of the directory containing the object detection module we're using
    MODEL_NAME = 'model'
    IMAGE_NAME = 'id_image.jpeg'

    # Grab path to current working directory
    CWD_PATH = os.getcwd()

    # Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH,'data','labelmap.pbtxt')

    # Path to image
    PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)

    # Number of classes the object detector can identify
    NUM_CLASSES = 1

    # Load the label map.
    # Label maps map indices to category names, so that when our convolution
    # network predicts `5`, we know that this corresponds to `king`.
    # Here we use internal utility functions, but anything that returns a
    # dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.compat.v1.Session(graph=detection_graph)

    # Define input and output tensors (i.e. data) for the object detection classifier

    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Load image using OpenCV and
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    image = cv2.imread(PATH_TO_IMAGE)
    image_expanded = np.expand_dims(image, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

    # Draw the results of the detection (aka 'visulaize the results')
    image, array_coord = vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=3,
        min_score_thresh=0.60)

    ymin, xmin, ymax, xmax = array_coord

    shape = np.shape(image)
    im_width, im_height = shape[1], shape[0]
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)

    # Using Image to crop and save the extracted copied image
    im = Image.open(image_path)
    im.crop((left, top, right, bottom)).save(image_path, quality=95)
    # cv2.imwrite(image_path, image)
    # cv2.imshow('ID-CARD-DETECTOR : ', image)

    # image_cropped = cv2.imread(output_path)



def autocrop(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Iterate through contours and find rectangles
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Filter rectangles based on aspect ratio
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if 0.8 <= aspect_ratio <= 1.2 and len(approx) == 4:
            cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)

    # Display the result
    print("idhar to aaye")
    cv2.imwrite(image_path, image)

def preprocess_image(image_data):
    # Decode base64 and open image using PIL
    image_data_bytes = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_data_bytes))

    # Increase contrast of the image
    # enhancer = ImageEnhance.Contrast(image)
    # image = enhancer.enhance(2.0)  # Adjust the enhancement factor as needed

    # Convert the image to grayscale
    # image = image.convert('L')

    # Binarize and de-noise the image
    # image = image.point(lambda x: 0 if x < 128 else 255)
    # image = image.filter(ImageFilter.MedianFilter())

    # De-skew the image
    # image = image.rotate(image.getexif().get(274, 0), resample=Image.BICUBIC, expand=True)

    return image

def preprocessing_after_crop(image_path):
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(
    src=image,
    ksize=(5, 5),
    sigmaX=0,
    sigmaY=0)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    _, image =cv2.threshold(image, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # _, image = cv2.threshold(image, thresh=165, maxval=255, type=cv2.THRESH_TRUNC + cv2.THRESH_OTSU)
    # skew_correction(image)
    cv2.imwrite(image_path,image)

def new_preprocess(image_path):
    # Convert to grayscale
    gray = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to improve text clarity
    # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # Denoise using median filtering
    # denoised = cv2.medianBlur(thresh, 3)
    # kernel = np.ones((3, 3), np.uint8)
    # denoised = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    # denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite(image_path,gray)

def skew_correction(gray_image):
    orig = gray_image
    # threshold to get rid of extraneous noise
    thresh = threshold_otsu(gray_image)
    normalize = gray_image > thresh
    blur = gaussian_filter(normalize, 3)
    edges = canny(blur)
    hough_lines = probabilistic_hough_line(edges)
    # hough lines returns a list of points, in the form ((x1, y1), (x2, y2))
    # representing line segments. the first step is to calculate the slopes of
    # these lines from their paired point values
    slopes = [(y2 - y1) / (x2 - x1) if (x2 - x1) else 0 for (x1, y1), (x2, y2) in hough_lines]
    # it just so happens that this slope is also y where y = tan(theta), the angle
    # in a circle by which the line is offset
    rad_angles = [np.arctan(x) for x in slopes]
    # and we change to degrees for the rotation
    deg_angles = [np.degrees(x) for x in rad_angles]
    # which of these degree values is most common?
    histo = np.histogram(deg_angles, bins=100)
    # correcting for 'sideways' alignments
    rotation_number = histo[1][np.argmax(histo[0])]
    if rotation_number > 45:
        rotation_number = -(90 - rotation_number)
    elif rotation_number < -45:
        rotation_number = 90 - abs(rotation_number)
    # rotate the image to deskew it
    (h, w) = gray_image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, rotation_number, 1.0)
    rotated = cv2.warpAffine(orig, matrix, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    rotated_image = Image.fromarray(rotated)
    # print("came here",np.array(rotated))
    # image=Image.fromarray(np.array(rotated))
    rotated_image.save("id_image.jpeg")
    # print(rotation_number)
    # return np.array(rotated), rotation_number

    
@app.route('/get_image', methods=['POST'])
def get_image():
    print("aaye kya")
    json_data = request.get_json()
    image_data = json_data['image']
    image_data=image_data.replace("data:image/jpeg;base64,","")

    # Assuming you have the image data as bytes (replace this with your image data)
    # image_data = b'your_binary_image_data_here'
    # print("aaye kya",image_data)
    print("aaye kyyyyya")
    # Encode the image data as base64
    # encoded_image = base64.b64encode(image_data)
    # with open("imageToSavebb.png", "wb") as fh:
    #  fh.write(base64.decodebytes(image_data))
    # image_data_bytes = base64.b64decode(image_data)

    # print(image_data)

    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update with your Tesseract path

    image = preprocess_image(image_data) 
    # image = Image.open(BytesIO(image_data_bytes))
    image.save("id_image.jpeg")
    image_path="id_image.jpeg"
    # auto_crop_id_card(image_path)
    # preprocessing_after_crop(image_path)
    # new_preprocess(image_path)

    # image = cv2.copyMakeBorder(
	# src=cv2.imread(image_path),
	# top=20,
	# bottom=20,
	# left=20,
	# right=20,
	# borderType=cv2.BORDER_CONSTANT,
	# value=(255, 255, 255))
    # cv2.imwrite(image_path,image)
    # image=Image.open(image_path)
    image=Image.open(image_path)
    str=pytesseract.image_to_string(image)

    print("jndjnwd",str)
    # Now, you can perform operations on the image
    # For example, you can save it to a file
    

    # Specify the content type for the response
    # content_type = 'image/jpeg'

    # Return the image file with the specified content type
    return jsonify({"text": str})

if __name__ == '__main__':
    app.run(port=5000)
