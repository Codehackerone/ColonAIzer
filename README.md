# colorectal-tumour-detection
Colon Tumor Segmentation is a deep learning project that predicts colon tumor masks on colonoscopy images. This project has three main components:

- Experimentation - Semantic Segmentation implemented by U-Nets on CVCclinic    database. 
  https://paperswithcode.com/dataset/cvc-clinicdb
  - a lightweight U-Net model for real-time segmentation
- API - Flask based API for segmentation of images as well as videos.
- GUI - Tkinter app that takes input from camera and shows the segmentation maps in real-time.

# Architecture
The model architecture used in this project is the UNet architecture, which is an encoder-decoder type neural network that has been widely used for image segmentation tasks. The architecture consists of a contracting path, which consists of convolutional and max-pooling layers that extract features from the input image, and an expansive path, which consists of convolutional and up-sampling layers that produce the final segmentation map. Skip connections between the contracting and expansive paths allow the network to preserve spatial information from the input image.

The model has been optimized for real-time colon tumor segmentation, using a combination of hyperparameter tuning and optimization techniques.

```
Input (256, 256, 3)

Conv2D (64 filters, 3x3) -> BatchNorm -> ReLU
Conv2D (64 filters, 3x3) -> BatchNorm -> ReLU
MaxPooling2D (2x2)

Conv2D (128 filters, 3x3) -> BatchNorm -> ReLU
Conv2D (128 filters, 3x3) -> BatchNorm -> ReLU
MaxPooling2D (2x2)

Conv2D (512 filters, 3x3) -> BatchNorm -> ReLU
Conv2D (512 filters, 3x3) -> BatchNorm -> ReLU
MaxPooling2D (2x2)

Conv2D (1024 filters, 3x3) -> BatchNorm -> ReLU
Conv2D (1024 filters, 3x3) -> BatchNorm -> ReLU

UpSampling2D (2x2)
Concatenate
Conv2D (512 filters, 3x3) -> BatchNorm -> ReLU
Conv2D (512 filters, 3x3) -> BatchNorm -> ReLU

UpSampling2D (2x2)
Concatenate
Conv2D (128 filters, 3x3) -> BatchNorm -> ReLU
Conv2D (128 filters, 3x3) -> BatchNorm -> ReLU

UpSampling2D (2x2)
Concatenate
Conv2D (64 filters, 3x3) -> BatchNorm -> ReLU
Conv2D (64 filters, 3x3) -> BatchNorm -> ReLU

Conv2D (1 filter, 1x1)
Activation (sigmoid)
```

# API Docs
This project also has a Flask-based API that allows users to submit an image for colon tumor segmentation. The API endpoint /predict accepts a POST request with a file URL, downloads the image from the URL, segments it using the model, and returns the segmented image.

Request
```
  HTTP Method: POST
  Endpoint: /predict
  Request Body:
  file_url: URL of the image to be segmented
  Response
  Status Code: 200 OK
  Response Body:
  segmented_image_path: URL of the segmented image
```

# GUI App
This project includes a GUI app that allows users to see the segmentation maps in real-time from their device's camera. The app is built using the tkinter library and the OpenCV library.


# Conclusion
Colon Tumor Segmentation is a project that utilizes deep learning to perform colon tumor segmentation on colonoscopy images. The project includes a machine learning model that has been optimized for real-time segmentation, a Flask-based API for segmenting images, and a GUI app for real-time segmentation from a camera feed. The project can be used for medical diagnosis and research purposes, and the code can be easily extended to support other image segmentation tasks.