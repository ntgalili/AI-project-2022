import cv2
from mtcnn.mtcnn import MTCNN
from twilio.rest import Client
import time


# --------------
# calling and massege:

# You must register on the website and complete the details:
# https://www.twilio.com/try-twilio
from_whatsapp = 'whatsapp:_______'    # Twilio Sandbox for WhatsApp (whatsapp:+123456789)
to_whatsapp = 'whatsapp:_______'     # your number you signed with

# send a message to my phone, of text or text and a picture with a link
def send_message(msg, img = 0):
    myClient = Client('___ACCOUNT SID___', '___AUTH TOKEN___')
    if(img == 0):
        myClient.messages.create(from_=from_whatsapp, to=to_whatsapp, body=msg)
        return
    myClient.messages.create(from_=from_whatsapp, to=to_whatsapp,body=msg,media_url=img)


# -------------
# setting:

camera = cv2.VideoCapture(0)  # '0' means leptop camera
# If Camera Device is not opened, exit the program
if not camera.isOpened():
    print("Video device or file couldn't be opened")
    exit()
camera.set(3, 640) # set width
camera.set(4, 480) # set hight

cv2.namedWindow("video")
# counter for the names of the pictures
img_counter = 0


#--------------
# age part:

ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

ageNet=cv2.dnn.readNet(ageModel,ageProto)

def age_detected(face):
    blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
    ageNet.setInput(blob)
    agePreds = ageNet.forward()
    return ageList[agePreds[0].argmax()]

#-------------
# face part:

detector = MTCNN()

def face_and_age_detected(image):
    #image = img.copy()
    faces = detector.detect_faces(image)  # result
    list_of_ages = []
    for result in faces:
        x, y, w, h = result['box']
        x1, y1 = x + w, y + h

        # age detection:
        padding = 20
        face = image[max(0, y - padding):
                     min(y1 + padding, image.shape[0] - 1), max(0, x - padding)
                                                            :min(x1 + padding, image.shape[1] - 1)]
        age = age_detected(face)
        list_of_ages += [age]
        cv2.rectangle(image, (x, y), (x1, y1), (0, 0, 255), 2)
        cv2.putText(image, age, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    return image, list_of_ages


# ----------------
cases_count = 0
check_count = 3
ok_val = check_count+1


def regVal():
    if check_count % 2 == 0:
        x = check_count/2
    else:
        x = (check_count//2) +1
    return x + ok_val * (check_count-x)

reg_val = regVal()


#list_age = [[],[],[]]

#Timer for capture every sec seconds
timer = 0
#sec to capture
sec=2


while True:
    ret, frame = camera.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("video", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break


        # If waiting time is over sec
    elif (time.time() - timer >= sec):         # k%256 == 32:# SPACE pressed #
        timer = time.time() #update timer


        # # save the pictures in Folder pictures
        img_name = "opencv_frame_{}.png".format(img_counter%check_count)
        # directory = r"C:\Users\office\PycharmProjects\openCVpython\images"
        # #Set a location for saving the captured images
        # os.chdir(directory)
        # cv2.imwrite(img_name, frame)
        # #print("{} written!".format(img_name))


        #img = cv2.read(directory + "\\" + img_name)
        result_image, list_of_age = face_and_age_detected(frame)
        # get image with mark faces and the list of the ages

        # cv2.imshow(img_name, result_image)
        cv2.imshow(img_name, result_image)
        if ('(0-2)' in list_of_age or
                '(4-6)' in list_of_age or
                '(8-12)' in list_of_age):
            # print("there is a child")
            if ('(15-20)' in list_of_age or
                    '(25-32)' in list_of_age or
                    '(38-43)' in list_of_age or
                    '(48-53)' in list_of_age or
                    '(60-100)' in list_of_age):
                cases_count += ok_val
                # print("there is an adult")
            else:
                # create_call()
                cases_count += 1

        else:
            # print("there is not children under 6")
            cases_count += ok_val

        img_counter += 1

        # Check each check_count photos what the condition of the vehicle is
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
