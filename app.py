from flask import Flask, request, jsonify
from minio import Minio
from minio.error import S3Error
import os
from io import BytesIO


from flask_cors import CORS
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


from preprocessing import extract_fbanks,convert_to_wav
from predictions import get_embeddings, get_cosine_distance

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

DATA_DIR = 'data_files/'
THRESHOLD = 0.45

minio_client = Minio(
    'localhost:9000',  
    access_key='bBYFms3yAqPSp5AQvgoy',  
    secret_key='dVHpga9oCU8sBV2Y7cdf9CpFVCq0Ywb7C1jqd8OD',  
     secure=False
  
)

bucket_name = 'voice-verif'

# Ensure the bucket exists
if not minio_client.bucket_exists(bucket_name):
    print("Creating the bucket...")
    minio_client.make_bucket(bucket_name)
#----------------------------------------------------------------------------------------------------------------#
@app.route('/')
def home():
    return "Welcome to the Flask backend!"
#----------------------------------------------------------------------------------------------------------------#
@app.route('/register-audio', methods=['POST'])
def register_audio():
    public_address = request.args.get('id')
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    wav_filename = f"{public_address}.wav"
    embeddings_filename = f"{public_address}_embeddings.npy"

    try:
    
        wav_io = convert_to_wav(file)
        minio_client.put_object(
            bucket_name,
            wav_filename,
            wav_io,
            length=wav_io.getbuffer().nbytes,
            content_type='audio/wav'
        )
        wav_io.seek(0)  
        fbanks = extract_fbanks(wav_io)  
        embeddings = get_embeddings(fbanks) 
        mean_embeddings = np.mean(embeddings, axis=0) 
        embeddings_io = BytesIO()
        np.save(embeddings_io, mean_embeddings)
        embeddings_io.seek(0)

        minio_client.put_object(
            bucket_name,
            embeddings_filename,
            embeddings_io,
            length=embeddings_io.getbuffer().nbytes,
            content_type='application/octet-stream'
        )

        return jsonify({"message": "Audio and embeddings uploaded and converted successfully"}), 201

    except S3Error as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
#----------------------------------------------------------------------------------------------------------------#
@app.route('/verify-audio/<string:public_address>', methods=['POST'])
def verify_audio(public_address):
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        wav_io = convert_to_wav(file)

        fbanks = extract_fbanks(wav_io)
        embeddings = get_embeddings(fbanks)
        embeddings_filename = f"{public_address}_embeddings.npy"
        embeddings_io = BytesIO()
        
        try:
            response = minio_client.get_object(bucket_name, embeddings_filename)
            data = response.read()
            if not data:
                raise ValueError("Received empty data from MinIO")
            embeddings_io.write(data)
            embeddings_io.seek(0)
            stored_embeddings = np.load(embeddings_io)
        except S3Error as e:
            print(f"Error accessing stored embeddings from MinIO: {e}")
            return jsonify({"error": f"Error accessing stored embeddings from MinIO: {e}"}), 500
        except Exception as e:
            print(f"Unexpected error accessing stored embeddings: {e}")
            return jsonify({"error": f"Unexpected error accessing stored embeddings: {e}"}), 500

        distances = get_cosine_distance(embeddings, stored_embeddings)
        print('Mean distances:', np.mean(distances), flush=True)
        positives = distances < THRESHOLD
        positives_mean = np.mean(positives)
        print('Positives mean:', positives_mean, flush=True)
        
        if positives_mean >= 0.65:
            print("Success")
            return jsonify({"message": "SUCCESS"}), 200
        else:
            print("Failure")
            return jsonify({"message": "FAILURE"}), 401

    except Exception as e:
        print(f"General error: {e}")
        return jsonify({"error": f"General error: {e}"}), 500

#----------------------------------------------------------------------------------------------------------------#
if __name__ == '__main__':
   
    app.run(debug=True, use_reloader=False)
