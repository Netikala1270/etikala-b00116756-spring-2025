from flask import Flask, render_template, Response
import os
from flask import Flask, render_template, request
from FER_Camera import VideoCamera,process_uploaded_image

app = Flask(__name__)

@app.route("/", methods = ['GET','POST'])
def index():
    return render_template("index.html")

def generate(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n') 

@app.route("/video_feed")
def video_feed():
    return Response(generate(VideoCamera()),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/video")
def video():
    return render_template("video.html")

@app.route("/image")
def image():
    return render_template("image.html")

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route('/upload', methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    print("Upload directory:", target)

    if not os.path.isdir(target):
        os.mkdir(target)

    file = request.files['file']
    filename = file.filename
    dest = os.path.join(target, filename)
    print("Saving file to:", dest)

    file.save(dest)  # Save the uploaded file
    print(filename)
    #  Predict the Pox type
    predicted_class = process_uploaded_image(filename)

    return render_template('complete.html', image_name=predicted_class)


if __name__=='__main__':
    app.run(debug=True)