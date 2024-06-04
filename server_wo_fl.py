import http.server
import socketserver
import cgi
import os
from resnet_50 import ResNet50Custom
# Specify the port and upload folder
PORT = 8001

UPLOAD_FOLDER = 'uploads'
import librosa
import io
import soundfile as sf
import numpy as np

# Ensure the upload folder exists
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)
class ModelInitializer():
    def __init__(self) -> None:
        self.model = ResNet50Custom(10, (32,1290,1))
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.load_weights('genre_classification_model_073.h5')
        self.labels = np.load("label_classes.npy")
    def get_genre(self, song, sr):
        mfccs = librosa.feature.mfcc(y=song, sr=sr, n_mfcc=32)
        mfccs = librosa.util.fix_length(mfccs, size=1290, axis=1)
        mfccs = np.expand_dims(mfccs, axis=-1)
        mfccs = np.expand_dims(mfccs, axis=0)
        return self.labels[np.argmax(self.model.predict(mfccs))]
model = ModelInitializer()
class SimpleHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        # Check if content type is multipart/form-data
        content_type, pdict = cgi.parse_header(self.headers['Content-Type'])
        if content_type == 'multipart/form-data':
            # Parse the form data
            pdict['boundary'] = bytes(pdict['boundary'], "utf-8")
            form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={'REQUEST_METHOD': 'POST'}, keep_blank_values=True)

            # Extract the file from the form
            if len(form.list):
                file_item = form.list[0]
                sound = file_item.file.read()
                y, sr = librosa.load(io.BytesIO(sound), sr=22050)

                #y, sr = librosa.resample(data, orig_sr=samplerate, target_sr=22050)
                frame_crop = librosa.time_to_samples([float(form.list[1].value), float(form.list[2].value)], sr=22050)
                
                y = y[frame_crop[0]:frame_crop[1]]

                label = model.get_genre(y, sr)
                
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(label)
                # if file_item.filename:
                #     # Set the file path
                #     file_path = os.path.join(UPLOAD_FOLDER, file_item.filename)

                #     # Save the file
                #     with open(file_path, 'wb') as output_file:
                #         output_file.write(file_item.file.read())

                #     # Send response back to the client
                #     self.send_response(200)
                #     self.send_header('Content-type', 'text/html')
                #     self.send_header('Access-Control-Allow-Origin', '*')
                #     self.end_headers()
                #     self.wfile.write(b'File uploaded successfully')
                #     return
                # else:
                #     print("File item has no filename")
            else:
                print("No file part in form")
        else:
            print(f"Invalid content type: {content_type}")

            # Handle errors
            self.send_response(400)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(b'No file uploaded or invalid form')

Handler = SimpleHTTPRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Serving on port {PORT}")
    httpd.serve_forever()
