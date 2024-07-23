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
    print("entered register audio fct")
    public_address = request.args.get('id')
    print("public @",public_address)
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    wav_filename = f"{public_address}.wav"
    embeddings_filename = f"{public_address}_embeddings.npy"

    try:
        print("converting to wav")
        wav_io = convert_to_wav(file)
        print("done converting")

        # Save .wav file to MinIO
        minio_client.put_object(
            bucket_name,
            wav_filename,
            wav_io,
            length=wav_io.getbuffer().nbytes,
            content_type='audio/wav'
        )
        print("done saving wav in minio")
        
        wav_io.seek(0)  
        print("extracting embeddings")
        fbanks = extract_fbanks(wav_io)  
        embeddings = get_embeddings(fbanks) 
        mean_embeddings = np.mean(embeddings, axis=0) 
        print("EMBEDDINGS SIZE IN REGISTER BEGORE STORING",mean_embeddings.shape)
        embeddings_io = BytesIO()
        print("saving embeddings")
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
        print("Converting access-recorded audio to WAV")
        wav_io = convert_to_wav(file)
        print("Done converting")

        fbanks = extract_fbanks(wav_io)
        embeddings = get_embeddings(fbanks)
        print("Done getting the new embeddings")
        
        embeddings_filename = f"{public_address}_embeddings.npy"
        print("Embeddings filename:", embeddings_filename)
        embeddings_io = BytesIO()
        print("Let's get the stored embeddings")
        
        try:
            response = minio_client.get_object(bucket_name, embeddings_filename)
            data = response.read()
            if not data:
                raise ValueError("Received empty data from MinIO")
            embeddings_io.write(data)
            embeddings_io.seek(0)
            print("Let's np.load the embeddings")
            stored_embeddings = np.load(embeddings_io)
            print("Done getting stored embeddings")
        except S3Error as e:
            print(f"Error accessing stored embeddings from MinIO: {e}")
            return jsonify({"error": f"Error accessing stored embeddings from MinIO: {e}"}), 500
        except Exception as e:
            print(f"Unexpected error accessing stored embeddings: {e}")
            return jsonify({"error": f"Unexpected error accessing stored embeddings: {e}"}), 500
        
        #stored_embeddings = stored_embeddings.reshape((1, -1))
        print("OLD EMBEDDINGS SHAPE:",stored_embeddings.shape)
        print("NEW EMBEDDINGS SHAPE:",embeddings.shape)

        print("Calculate cosine distance")
        distances = get_cosine_distance(embeddings, stored_embeddings)
        print('Mean distances:', np.mean(distances), flush=True)

        print("Determining if match is valid")
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
