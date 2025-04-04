from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import cv2
import os
import numpy as np

json_file = open("model.json", "r")
loaded_json_model = json_file.read()
json_file.close()

model = model_from_json(loaded_json_model)
model.load_weights("model_weights.h5")

face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  
emotions = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')


def image_predicts(image):    
    
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
    
    for (x, y, w, h) in faces_detected:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), thickness=3)
        roi_gray = gray_img[y:y + w, x:x + h]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img = roi_gray.reshape((1, 48, 48, 1)) / 255.0
        
        max_index = np.argmax(model.predict(img), axis=-1)[0]
        predicted_emotion = emotions[max_index]
        print(predicted_emotion)
        cv2.putText(image, predicted_emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return image

def process_uploaded_image(image_path):
    image = cv2.imread(os.path.join("images", image_path))
    print(image)
    if image is None:
        raise ValueError("Invalid image path or unsupported format.")
    
    processed_image = image_predicts(image)
    output_path = 'static/images/processed_image.jpg'
    cv2.imwrite(output_path, processed_image)
    #cv2.imwrite('processed_image.jpg', processed_image)
    return 'processed_image.jpg'



def image_predict(image):
    return emotions[np.argmax(model.predict(image))]

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret,test_img=self.video.read()# captures frame and returns boolean value and captured image  
          
        
        gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)  

        try:
            faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

            for (x,y,w,h) in faces_detected:  
                cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=3)  
                roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image  
                roi_gray=cv2.resize(roi_gray,(48,48))  
                img = roi_gray.reshape((1,48,48,1))
                img = img /255.0

                max_index = np.argmax(model.predict(img.reshape((1,48,48,1))), axis=-1)[0]

                  
                predicted_emotion = emotions[max_index]  

                cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        except:
            pass

        resized_img = cv2.resize(test_img, (1000, 600))

        _, jpeg = cv2.imencode('.jpg', resized_img)

        return jpeg.tobytes()