#from keras.preprocessing import img_to_array
from flask import Flask, render_template, request, redirect, url_for, session, flash,Response
#import cv2
import numpy as np
#from keras.models import load_model
#from mtcnn.mtcnn import MTCNN
from flask_pymongo import PyMongo
from passlib.hash import pbkdf2_sha256

app = Flask(__name__)
app.config['SECRET_KEY'] = '123456'
app.config['MONGO_URI'] = 'mongodb://localhost:27017/your_database_name'

mongo = PyMongo(app)
"""

emotion_classifier = load_model("data_mini_XCEPTION.106-0.65.hdf5", compile=False)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
Emotions = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

detector = MTCNN()

genderProto = "gender_deploy.prototxt" #structure of the model
genderModel = "gender_net.caffemodel" #gender prediction model

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
genderList = ['Male', 'Female']

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
genderNet = cv2.dnn.readNet(genderModel, genderProto)

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        roi = gray[y:y + h, x:x + w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        preds = emotion_classifier.predict(roi)[0]
        label = Emotions[preds.argmax()]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"{label}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0))
    return frame

def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        else:
            frame_with_faces = detect_faces(frame)
            ret, buffer = cv2.imencode('.jpg', frame_with_faces)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def detect_gen(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(image=gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        roi = gray[y:y + h, x:x + w]
        blob = cv2.dnn.blobFromImage(roi, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f"{gender}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
    return frame

def gender_frames():
    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        else:
            frame_with_faces = detect_gen(frame)
            ret, buffer = cv2.imencode('.jpg', frame_with_faces)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
"""
@app.route('/')
def index():
    return render_template('index.html')




@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        users = mongo.db.users
        existing_user = users.find_one({'username': request.form['username']})

        if existing_user and pbkdf2_sha256.verify(request.form['password'], existing_user['password']):
            session['username'] = request.form['username']
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password. Please try again.', 'danger')

    return render_template('signin.html')


@app.route('/Signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        users = mongo.db.users
        existing_user = users.find_one({'username': request.form['username']})

        if existing_user is None:
            hashed_password = pbkdf2_sha256.hash(request.form['password'])
            users.insert_one({'username': request.form['username'], 'password': hashed_password})
            flash('Account created successfully!', 'success')
            return redirect(url_for('signin'))
        else:
            flash('Username already exists. Please choose a different one.', 'danger')
    return render_template("Signup")        

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/dash')
def dash():
    return render_template('dash.html')

@app.route('/eman')
def eman():
    return render_template('eman.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')
"""

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
    return Response(gender_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/signout')
def signout():
    session.pop('username', None)
    flash('You have been signed out.', 'info')
    return redirect(url_for('signin'))
"""

if __name__ == '__main__':
    app.run(debug=True)