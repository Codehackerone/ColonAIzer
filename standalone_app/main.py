import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk
from model import predict_image

class App:
    def __init__(self, video_source=0):
        self.video_source = video_source
        
        # Create the main window
        self.window = tk.Tk()
        self.window.title("Colon Tumor Segmentation")
        
        # Open the video stream
        self.video_capture = cv2.VideoCapture(self.video_source)
        
        # Create the canvas for displaying the video feed
        self.canvas = tk.Canvas(self.window, width=self.video_capture.get(3), height=self.video_capture.get(4))
        self.canvas.pack()
        
        # Create the button for starting the segmentation
        self.segment_button = tk.Button(self.window, text="Segment", command=self.segment_video)
        self.segment_button.pack(pady=10)
        
        # Create a variable for storing the segmented image
        self.segmented_image = None
        
        # Start the main loop
        self.window.mainloop()
        
    def segment_video(self):
        # Read the current frame from the video stream
        ret, frame = self.video_capture.read()
        
        if ret:
            # Convert the frame to RGB format
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Predict the segmentation map for the frame
            segmented_image_path = predict_image(frame)
            
            # Load the segmented image
            self.segmented_image = Image.open(segmented_image_path)
            
            # Resize the segmented image to fit the canvas
            self.segmented_image = self.segmented_image.resize((int(self.canvas['width']), int(self.canvas['height'])))
            
            # Convert the segmented image to a format that can be displayed in the canvas
            self.segmented_image = ImageTk.PhotoImage(self.segmented_image)
            
            # Display the segmented image in the canvas
            self.canvas.create_image(0, 0, anchor='nw', image=self.segmented_image)
            
            # Update the window to show the new image
            self.window.update()
        
if __name__ == '__main__':
    App()
