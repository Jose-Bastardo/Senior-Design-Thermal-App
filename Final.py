#import mariadb
import os  # Used to destroy image near the end
from os import path
import pickle  # Used to save face encoding
import face_recognition  # Used to access various facial recognition libaries and functions
import sys  # Used for sys.exit()
from datetime import datetime  # Used to print date on log.txt
import random  # Used to generate random reservation ID
import cv2  # Import OpenCV library for camera displays
import dlib
from math import hypot
import time

datFolderLoc = 'Face_Recognition/extract/'

# logFile = open("log.txt", "a")
# sys.stdout = logFile
print("\n\n========================================================================================")
print("----------------------------------------------------------------------------------------")
print("Running webcam_comparision.py")
# Print out date and time of code execution
todayDate = datetime.now()
print("Date and time of code execution:", todayDate.strftime("%m/%d/%Y %I:%M:%S %p"))
print("----------------------------------------------------------------------------------------")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # camera port 0

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


font = cv2.FONT_HERSHEY_SIMPLEX


def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    # hor_line = cv2.line(frame, left_point, right_point,(0,255,0), 1)

    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
    # ver_line = cv2.line(frame, center_top, center_bottom,(0,255,0), 1)

    # length of the line
    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    ratio = hor_line_length / ver_line_length, ver_line_length
    return ratio

for n in range(5):
    blink = 1
    TOTAL = 0
    thres = 5.1
    startTime = 0
    timeStart = False

    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # for gray images(lightweight)
        faces = detector(gray)
        for face in faces:
            # x, y = face.left(), face.top()
            # x1, y1 = face.right(), face.bottom()
            # cv2.rectangle(frame, (x,y), (x1,y1), (0,255,0), 3 )# green box, thickness of box
            landmarks = predictor(gray, face)
            left_eye_ratio, _ = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
            right_eye_ratio, myVerti = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
            blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
            personal_threshold = 0.67 * myVerti  # 0.67 is just the best constant I found with experimentation

            if (left_eye_ratio > personal_threshold or right_eye_ratio > personal_threshold) and blink == 1:
                TOTAL += 1
                time.sleep(0.2)  # average persons blinking time
            if (left_eye_ratio > personal_threshold or right_eye_ratio > personal_threshold):
                blink = 0
            else:
                blink = 1

            # cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)

        key = cv2.waitKey(5)
        if TOTAL >= 2:
            if timeStart == False:
                startTime = time.time()
                timeStart = True

            timeElapsed = time.time() - startTime
            if timeElapsed > 1.5:
                camImg = "Test" + str(n) + ".jpg"
                cv2.imwrite("Face_Recognition/uploadedImages/" + camImg, frame)
                break

        # debug: press space to exit webcam
        if key % 256 == 32:
            # camImg = "Test.jpg"
            # cv2.imwrite(camImg, frame)
            cap.release()
            cv2.destroyAllWindows()
            print("Debug space pressed. Terminating program.")
            print("========================================================================================")
            sys.exit()

cap.release()
cv2.destroyAllWindows()

uploadedImageLoc = 'Face_Recognition/uploadedImages/'
datFolderLoc = 'Face_Recognition/dat/'


# Create random number based on specified length of n
def randomNum(n):
    min = pow(10, n - 1)
    max = pow(10, n) - 1
    return random.randint(min, max)


# Create or open log.txt to be written
logFile = open("log.txt", "a")
sys.stdout = logFile
print("\n\n========================================================================================")
print("----------------------------------------------------------------------------------------")
print("Running uploaded images' patterns extraction")
# Print out date and time of code execution
todayDate = datetime.now()
print("Date and time of code execution:", todayDate.strftime("%m/%d/%Y %I:%M:%S %p"))
print("----------------------------------------------------------------------------------------")

# Check if upload folder directory is valid
if (path.exists(uploadedImageLoc) == False):
    print('Folder path does not exists. Terminating program....')
    sys.exit()

else:
    # For loop to read images on uploadedImages folder
    for imageFile in os.listdir(uploadedImageLoc):
        print("Loading image:", imageFile)

        # checks if the file does exists
        if os.path.isfile(uploadedImageLoc + imageFile):
            # print("File located! Executing facial pattern extraction")

            # start of encoding
            known_image = face_recognition.load_image_file(uploadedImageLoc + imageFile)
            try:
                known_face_encoding = face_recognition.face_encodings(known_image)[0]
            except IndexError:
                print(
                    "I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
                sys.stdout.close()
                sys.exit()

            known_faces = [
                known_face_encoding
            ]

            # Save encoding as .dat file
            fileLength = len(imageFile) - 4
            datExtension = ".dat"
            if (fileLength <= 0):
                print("ERROR! File naming failed! Terminating program....")
                sys.stdout.close()
                sys.exit()

            else:
                # Generate a random 20 number code for reservation ID
                datFileName = str(randomNum(20))
                with open(datFolderLoc + datFileName + datExtension, 'wb') as f:
                    pickle.dump(known_faces, f)
                    print("Saved pattern data as " + datFileName + " under dat folder.")
                os.remove(uploadedImageLoc + imageFile)
                # Send number to the hotel database
                # OR, do this after booking

        else:
            print("Image doesn't exist! Terminating program....")
            print("========================================================================================")
            sys.exit()
    print("SUCCESS: Patterns extraction completed. Terminiating program.")
    print("========================================================================================")
    sys.exit()