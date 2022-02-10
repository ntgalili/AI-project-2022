import cv2
from mtcnn.mtcnn import MTCNN
from twilio.rest import Client
import time


# --------------
# calling and massege:

# You must register on the website and complete the details: https://www.twilio.com/try-twilio
from_whatsapp = 'whatsapp:_______'    # Twilio Sandbox for WhatsApp (whatsapp:+123456789)
to_whatsapp = 'whatsapp:_______'     # your number you signed with
myClient = Client('___ACCOUNT SID___', '___AUTH TOKEN___')

# send a message to my phone, of text or text and a picture with a link
def send_message(msg, img = 0):
    if(img == 0):
        myClient.messages.create(from_=from_whatsapp, to=to_whatsapp, body=msg)
        return
    myClient.messages.create(from_=from_whatsapp, to=to_whatsapp,body=msg,media_url=img)


# -------------
# setting of camera:

camera = cv2.VideoCapture(0)  # '0' is leptop camera, '1' is a webcam

if not camera.isOpened(): # If Camera Device is not opened, exit the program
    print("Video device or file couldn't be opened")
    exit()
camera.set(3, 640) # set width
camera.set(4, 480) # set hight

cv2.namedWindow("video")

# counter for the names of the pictures, in case we want to save the them on a computer folder
img_counter = 0


#--------------
# age part:
# (Based on https://www.thepythoncode.com/article/predict-age-using-opencv)

# Initializing the paths of the models' weights and architecture, and loading the models:

ageProto="age_deploy.prototxt"  # The model architecture - containing all the neural network layer’s definitions
# download from: https://github.com/ntgalili/AI-project-2022
ageModel="age_net.caffemodel"  # The pre-trained model weights for age detection
# download from: https://drive.google.com/open?id=1kWv0AjxGSN0g31OeJa02eBGM0R_jcjIl

ageNet=cv2.dnn.readNet(ageModel,ageProto) # Load age prediction model

# The 8 age classes of this CNN probability layer
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
# Gets an image of a face as input, and returns an age:
def age_detected(face):
    # Input image to preprocess before passing it through our dnn for classification.
    blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
    
    # Predict Age:
    ageNet.setInput(blob)
    agePreds = ageNet.forward()
    return ageList[agePreds[0].argmax()]

#-------------
# face part:
# (Based on https://towardsdatascience.com/face-detection-models-which-to-use-and-why-d263e82c302c)

detector = MTCNN() # Load the prediction model

# Gets an image as input, and returns the image with marked squares, and a list of identified ages:
def face_and_age_detected(image):
    faces = detector.detect_faces(image) # faces detection:
    
    list_of_ages = []
    
    # For each of the faces identified we'll predict it age, and print a square on the image
    for result in faces:
        x, y, w, h = result['box']
        x1, y1 = x + w, y + h

        # Crop the image for a particular face and send it to age detector
        padding = 20
        face = image[max(0, y - padding):
                     min(y1 + padding, image.shape[0] - 1), max(0, x - padding)
                                                            :min(x1 + padding, image.shape[1] - 1)]
        age = age_detected(face)
        list_of_ages += [age]
        
        # draw the rectangle around the face, and write the age
        cv2.rectangle(image, (x, y), (x1, y1), (0, 0, 255), 2)
        cv2.putText(image, age, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    
    return image, list_of_ages


# ----------------
# settings of the main loop:

# In order to determine whether a warning should be sent or not, we will use the following constants:
cases_count = 0     # Counts the state values of the images in each round
check_count = 3     # The number of photos in each round
ok_val = check_count+1 # Normal condition value - no people / adults only / adult presence with the children

def regVal():
    x = check_count//2
    return (check_count - x) + (ok_val * x)
reg_val = regVal()  # A cut line = A line that determines the classification (normal condition or forgetting a child)

# Timer for capture a new photo every 'sec' seconds
timer = 0
sec=2


# (The loop based on https://realpython.com/face-detection-in-python-using-a-webcam/)
while True:
    # # Capture frame-by-frame
    ret, frame = camera.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("video", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed - closing the program
        print("Escape hit, closing...")
        break

    # If waiting time is over 'sec' seconds:
    elif (time.time() - timer >= sec):
        timer = time.time() # update timer

        """ save the pictures in Folder pictures:
        img_name = "opencv_frame_{}.png".format(img_counter)
        directory = r"C:\Users\office\PycharmProjects\openCVpython\images" # the relevant path
        #Set a location for saving the captured images
        os.chdir(directory)
        cv2.imwrite(img_name, frame)
        #print("{} written!".format(img_name))
        img = cv2.read(directory + "\\" + img_name) """
        
        result_image, list_of_age = face_and_age_detected(frame)   # get image with mark faces and the list of the ages
        img_name = "opencv_frame_{}.png".format(img_counter%check_count)

        cv2.imshow(img_name, result_image)
        img_counter += 1
        
        # Checking the condition of the current frame according to the identified ages:
        if ('(0-2)' in list_of_age or
                '(4-6)' in list_of_age or
                '(8-12)' in list_of_age):
            # print("there is a child")
            if ('(15-20)' in list_of_age or
                    '(25-32)' in list_of_age or
                    '(38-43)' in list_of_age or
                    '(48-53)' in list_of_age or
                    '(60-100)' in list_of_age):
                # print("there is an adult with the child")
                cases_count += ok_val
            else:
                # print("children only")
                cases_count += 1
        else:
            # print("no children")
            cases_count += ok_val


        # Checking each 'check_count' photos what the condition of the vehicle is:
        if img_counter % check_count == 0:
            if cases_count <= reg_val:
                #print("call")
                send_message("call")
            else:
                #print("all ok")
                send_message("all ok")
            cases_count = 0
            

camera.release()
cv2.destroyAllWindows()
cv2.waitKey(0)
