import cv2
from keras.models import model_from_json
import numpy as np
import webbrowser  
# from keras_preprocessing.image import load_img
json_file = open("facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

model.load_weights("facialemotionmodel.h5")
haar_file=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade=cv2.CascadeClassifier(haar_file)

final_face = ''

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0

webcam=cv2.VideoCapture(0)
labels = {0 : 'angry', 1 : 'disgust', 2 : 'fear', 3 : 'happy', 4 : 'neutral', 5 : 'sad', 6 : 'surprise'}
while True:
    i,im=webcam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(im,1.3,5)
    try: 
        for (p,q,r,s) in faces:
            image = gray[q:q+s,p:p+r]
            cv2.rectangle(im,(p,q),(p+r,q+s),(255,0,0),2)
            image = cv2.resize(image,(48,48))
            img = extract_features(image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            final_face = prediction_label
            # print("Predicted Output:", prediction_label)
            # cv2.putText(im,prediction_label)
            cv2.putText(im, '% s' %(prediction_label), (p-10, q-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,2, (0,0,255))

            
        if(final_face == 'happy'):
            url= 'https://www.javatpoint.com/python-tutorial'
            webbrowser.open_new_tab(url)  
            break
        elif(final_face == 'sad'):
            url= 'https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwj97e6i84eCAxWlT2wGHdvOAtoQtwJ6BAgOEAI&url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DsZfnwayum2M&usg=AOvVaw1pPMLEMqRobikwY67FZpjV&opi=89978449'
            webbrowser.open_new_tab(url)
            break
        elif(final_face == 'angry'):
            url= 'https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjSv56184eCAxVBRWcHHV1GB-8QwqsBegQICxAB&url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3D_mEC54eTuGw&usg=AOvVaw2GRcpJpIxnNnpr_2L03ROG&opi=89978449'
            webbrowser.open_new_tab(url)
            break
        elif(final_face == 'surprise'):
            url= 'https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwj6l47F84eCAxW8V2wGHchUATAQwqsBegQICRAB&url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DbMlJKBXU1v4&usg=AOvVaw0FYmcHrlFYuAE5pt53DG14&opi=89978449'
            webbrowser.open_new_tab(url)
            break
                

        cv2.imshow("Output",im)
        cv2.waitKey(27)
    except cv2.error:
        pass