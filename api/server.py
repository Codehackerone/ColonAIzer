from flask import Flask, jsonify, request, send_file, render_template
from model import predict_image
from upload import uploadImage
import requests
import shutil
import tempfile
from model import predict_video
from upload import uploadVideo
import os

EXPORT = os.path.join('static', 'images-export')

app = Flask(__name__)

app.config['EXPORT'] = EXPORT
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route('/')
def index():
    response = {
        'message': 'Hello, World!'        
    }
    return  jsonify(response), 200

# @app.route('/predict', methods=['GET'])
# def get_predict():
#     return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
    # Get the image file from the POST request
        file_url = request.form['file_url']
        if not file_url:
            # Return an error message if no image is uploaded
            return jsonify({'error': 'No file URL provided'}), 400

    # Download the image from cloudinary and store inside buffer folder    
        res = requests.get(file_url, stream = True)
        file_name = "./buffer/"+file_url.split('/')[-1]
        if res.status_code == 200:
            with open(file_name,'wb') as f:
                shutil.copyfileobj(res.raw, f)
            print('Image sucessfully Downloaded: ',file_name)
        else:
            print('Image Couldn\'t be retrieved')
            return jsonify({'error': 'Image Couldn\'t be retrieved'}), 400

        segmented_image_path = predict_image(file_name)
        upload_segmented_path = uploadImage(segmented_image_path, file_name.split('/')[-1].split('.')[0])
            #segmented_image_path, file_name)

        return jsonify(
            {
                'segmented_image_path': upload_segmented_path
            }), 200
    except Exception as e:
        print(e)
        return jsonify({'error': 'Error while predicting image'}), 400

@app.route('/predict_video', methods=['POST'])
def predict_video():
    try:
        # Get the video file from the POST request
        file = request.files.get('file')
        if not file:
            # Return an error message if no video file is uploaded
            return jsonify({'error': 'No video file provided'}), 400
        
        # Save the video file to a temporary location
        with tempfile.NamedTemporaryFile(suffix='.mp4') as f:
            file.save(f.name)
            video_path = f.name
        
        # Process the video using the optimized UNET model
        segmented_video_path = predict_video(video_path)
        
        # Upload the segmented video to a cloud storage service
        upload_segmented_path = uploadVideo(segmented_video_path, file.filename.split('.')[0])
        
        return jsonify({
            'segmented_video_path': upload_segmented_path
        }), 200
    except Exception as e:
        print(e)
        return jsonify({'error': 'Error while predicting video'}), 400

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5045, debug=True)