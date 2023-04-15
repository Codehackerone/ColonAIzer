import numpy as np
import cv2
from glob import glob
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import CustomObjectScope
import tempfile
from optimized_model import build_model

def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (256, 256))
    x = x/255.0
    return x

def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (256, 256))
    x = np.expand_dims(x, axis=-1)
    return x

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

def predict_image(path):          
    with CustomObjectScope({'iou': iou}):
        model = tf.keras.models.load_model("./coloncancer_50epochs.h5")
    x=read_image(path)      
    y_pred = model.predict(np.expand_dims(x, axis=0))[0] > 0.5
    y_pred = mask_parse(y_pred) * 255.0
    cv2.imwrite(path, y_pred)


def predict_video(video_path):
    # Load the optimized UNET model
    model = build_model()
    model.load_weights('./coloncancer_50epochs.h5')
    
    # Create a VideoCapture object for reading the input video file
    cap = cv2.VideoCapture(video_path)
    
    # Get the frame dimensions of the input video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create a VideoWriter object for writing the segmented output video
    segmented_video_path = tempfile.NamedTemporaryFile(suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(segmented_video_path, fourcc, 25.0, (frame_width, frame_height))
    
    # Loop through each frame of the input video and segment it using the optimized UNET model
    for i in range(num_frames):
        # Read the next frame from the input video
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess the frame for input to the optimized UNET model
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (256, 256))
        frame = frame / 255.0
        frame = np.expand_dims(frame, axis=0)
        
        # Segment the frame using the optimized UNET model
        pred = model.predict(frame)[0]
        pred = (pred > 0.5).astype(np.uint8) * 255
        
        # Resize the segmented mask to the frame dimensions of the input video
        pred = cv2.resize(pred, (frame_width, frame_height))
        
        # Write the segmented frame to the output video file
        out.write(cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR))
    
    # Release the input and output video resources
    cap.release()
    out.release()
    
    return segmented_video_path
