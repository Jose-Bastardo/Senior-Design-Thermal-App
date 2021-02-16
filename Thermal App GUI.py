from kivy.graphics import *
import mariadb
import numpy as np
from kivy.clock import Clock
from kivy.graphics.context_instructions import Color
from kivy.graphics.texture import Texture
from kivy.uix.button import Button
from kivy.uix.camera import Camera
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
import dbfunctions
from kivy.app import App
from kivy.lang import Builder
from kivy.core.window import Window
import time
import random
from datetime import datetime
import sys
import pickle
import os
from os import path
import cv2
import dlib
from math import hypot
import face_recognition

Window.fullscreen = True

Builder.load_string('''


''')


class KivyCamera(Image):
    def __init__(self, capture, fps, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.capture = capture
        Clock.schedule_interval(self.update, 1.0 / fps)


    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            # convert it to texture
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tobytes()
            image_texture = Texture.create(
                size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # display image from the texture
            self.texture = image_texture


class layout(FloatLayout):
    def __init__(self, **kwargs):
        # make sure we aren't overriding any important functionality
        super(layout, self).__init__(**kwargs)
        self.capture = cv2.VideoCapture(0)
        self.my_camera = KivyCamera(capture=self.capture, fps=30)
        self.add_widget(self.my_camera)

        capturebutton = Button(text="Capture",
                               size_hint=(.7, .1),
                               # on_press=self.capturebtn(),
                               pos_hint={'center_x': .5, 'y': .1},
                               )
        comparebutton = Button(text="Compare",
                               size_hint=(.7, .1),
                               # on_press=self.comparebtn(),
                               pos_hint={'center_x': .5, 'y': 0},
                               )
        self.successtb = Label(text="[color=ffffff]Success[/color]",
                               pos_hint={"center_x": .5},
                               markup=True,
                               )
        self.failtb = Label(text="[color=ffffff]Fail[/color]",
                            pos_hint={"center_x": .5},
                            markup=True,
                            )

        self.add_widget(capturebutton)
        self.add_widget(comparebutton)

        capturebutton.bind(on_press=lambda x: self.capturebtn(None))
        comparebutton.bind(on_press=lambda x: self.comparebtn(None))

    def on_stop(self):
        # without this, app will not exit even if the window is closed
        self.capture.release()

    def capturebtn(self, *args):
        '''
        Function to capture the images and give them the names
        according to their captured time and date.
        '''

        logFile = open("log.txt", "a")
        sys.stdout = logFile
        print("\n\n========================================================================================")
        print("----------------------------------------------------------------------------------------")

        timestr = time.strftime("%Y%m%d_%H%M%S")
        ret, frame = self.capture.read()
        cv2.imwrite("Face_Recognition/images/IMG_{}.png".format(timestr), frame)
        print("Captured")

        uploadedImageLoc = 'Face_Recognition/images/'
        datFolderLoc = 'Face_Recognition/dat/'

        # Create random number based on specified length of n
        def randomNum(n):
            min = pow(10, n - 1)
            max = pow(10, n) - 1
            return random.randint(min, max)

        # Create or open log.txt to be written
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
                        dbfunctions.insertfacedb(27, datFolderLoc + datFileName + datExtension)
                        os.remove(datFolderLoc + datFileName + datExtension)

                else:
                    print("Image doesn't exist! Terminating program....")
                    print(
                        "========================================================================================")
                    sys.exit()
            print("SUCCESS: Patterns extraction completed. Terminiating program.")
            print("========================================================================================")

    def comparebtn(self, *args):

        def randomNum(n):
            min = pow(10, n - 1)
            max = pow(10, n) - 1
            return random.randint(min, max)

        datFolderLoc = 'Face_Recognition/extract/'

        # logFile = open("log.txt", "a")
        # sys.stdout = logFile
        print("\n\n=======================================================================================")
        print("----------------------------------------------------------------------------------------")
        print("Running webcam_comparision.py")
        # Print out date and time of code execution
        todayDate = datetime.now()
        print("Date and time of code execution:", todayDate.strftime("%m/%d/%Y %I:%M:%S %p"))
        print("----------------------------------------------------------------------------------------")

        cap = self.capture

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

            #key = cv2.waitKey(5)
            if TOTAL >= 2:
                if (timeStart == False):
                    startTime = time.time()
                    timeStart = True;

                timeElapsed = time.time() - startTime
                if (timeElapsed > 1.5):
                    camImg = "Test.jpg"
                    cv2.imwrite(camImg, frame)
                    break

        # Search for patterns from webcam image
        unknown_image = face_recognition.load_image_file(camImg)
        try:
            unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
        except IndexError:
            print(
                "I wasn't able to locate any faces in at least one of the images. Check the image files. Terminating program....")
            os.remove(camImg)
            print("========================================================================================")
            sys.exit()

        print("Face found. Beginning comparision check....")
        # For loop to compare patterns from webcam with .dat files
        # If comparison returns true, break from for loop
        personFound = False
        data = dbfunctions.returnallfaces()
        for d in data:
            randnum = str(randomNum(20))
            dataFile = datFolderLoc + randnum + ".dat"
            s = open(dataFile, 'wb')
            s.write(d[1])
            s.close()
            with open(dataFile, 'rb') as f:
                known_faces = pickle.load(f)

            results = face_recognition.compare_faces(known_faces, unknown_face_encoding)
            if (results[0] == True):
                personFound = True
                break
            else:
                os.remove(dataFile)

        # Prints results
        if (personFound == True):
            self.add_widget(self.successtb)
            print("SUCCESS: ", dataFile, " has a face that matches the person in", camImg)
            # Destroy webcam image and respective .dat file
            print("Deleting ", dataFile, " and ", camImg, "from system before terminating program.")
            print("========================================================================================")
            os.remove(camImg)
            os.remove(dataFile)

        else:
            self.add_widget(self.failtb)
            print("FAILURE:", "All available .dat files don't have any face that matches with the person found in",
                  camImg)
            print("Deleting", camImg, "from system before terminating program.")
            # Deletes webcam image
            os.remove(camImg)
            print("========================================================================================")
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class CamApp(App):
    @property
    def build(self):
        return layout


if __name__ == '__main__':
    CamApp().run()
