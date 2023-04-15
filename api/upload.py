import os
from dotenv import load_dotenv
load_dotenv()

# Import the Cloudinary libraries
import cloudinary
import cloudinary.uploader
import cloudinary.api

# Import to format the JSON responses
import json


def upload_image(file_path):
    cloudinary.config(
        cloud_name=os.environ.get('CLOUDINARY_CLOUD_NAME'),
        api_key=os.environ.get('CLOUDINARY_API_KEY'),
        api_secret=os.environ.get('CLOUDINARY_API_SECRET')
    )
    
    try:
        # Upload the image file to Cloudinary
        result = cloudinary.uploader.upload(
            file_path,
            folder="segmented_images"
        )

        # Return the URL of the processed image
        return result['url']

    except Exception as e:
        print(f"Error while uploading image to Cloudinary: {str(e)}")
        return None


def upload_video(file_path):
    cloudinary.config(
        cloud_name=os.environ.get('CLOUDINARY_CLOUD_NAME'),
        api_key=os.environ.get('CLOUDINARY_API_KEY'),
        api_secret=os.environ.get('CLOUDINARY_API_SECRET')
    )
    
    try:
        # Upload the video file to Cloudinary
        result = cloudinary.uploader.upload(
            file_path,
            resource_type="video",
            folder="segmented_videos"
        )

        # Return the URL of the processed video
        return result['url']

    except Exception as e:
        print(f"Error while uploading video to Cloudinary: {str(e)}")
        return None
