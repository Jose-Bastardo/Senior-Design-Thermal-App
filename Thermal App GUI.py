import multiprocessing
import threading
from typing import TextIO

import cv2
import dlib
import face_recognition
import winsound
from kivy.graphics.context_instructions import Color
from kivy.graphics.vertex_instructions import Rectangle
from validate_email import validate_email
import kivy

kivy.require('1.9.0')
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.properties import partial
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
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
from math import hypot
import smtplib, ssl
from multiprocessing import Process
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label

Window.fullscreen = False

port = 465  # For SSL
admin_email = None
smtp_server = "smtp.gmail.com"
sender_email = "notarealemailplsignore@gmail.com"
global receiver_email  # Enter receiver address
password = "apesapesapes"
global firstname
global lastname
global newuserid
facethread = None
comparisonthread = None
userid = None
faces = None
global unknown_face_encoding

cascPath = "Face_Recognition/haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

def verifyfirstinstall():
    dir = "config.txt"
    if (path.isfile(dir)):
        getadminemail()
        return 1
    else:
        return 0

# Get saved admin email from text file
def getadminemail():
    global admin_email
    dir = "config.txt"
    with open(dir) as fp:
        line = fp.readline()
        while line:
            if (line.find("admin_email") >= 0):
                x = line.split("= ")
                admin_email = x[1]
                fp.close()
                break
            else:
                line = fp.readline()


# Start Face Encoding Thread
def start_face_encoding(image):
    global unknown_face_encoding
    unknown_face_encoding = face_recognition.face_encodings(image)[0]


# Start Facial Recognition Thread
def start_facial_recognition(_, frame, ):
    if frame is None:
        return

    global facethread, faceCascade

    if (facethread == None):
        facethread = threading.Thread(target=facialrecognition,
                                      args=(faceCascade, _, frame,))
        facethread.start()
    elif facethread.is_alive():
        print(facethread.is_alive())
        return
    else:
        print(facethread.is_alive())
        facethread = threading.Thread(target=facialrecognition, args=(faceCascade, _, frame,))
        facethread.start()


# Start Facial Comparison Thread
def start_facial_comparison(image):
    global comparisonthread

    if (comparisonthread == None):
        comparisonthread = threading.Thread(target=facecomparison,
                                            args=(image,))
        comparisonthread.start()
    elif comparisonthread.is_alive():
        print(comparisonthread.is_alive())
        return
    else:
        print(comparisonthread.is_alive())
        comparisonthread = threading.Thread(target=facecomparison, args=(image,))
        comparisonthread.start()


# Compares detected faces to facial data in database
def facecomparison(image):
    global faces
    global unknown_face_encoding

    def randomNum(n):
        min = pow(10, n - 1)
        max = pow(10, n) - 1
        return random.randint(min, max)

    datdir = "Face_Recognition/dat/"
    imagesdir = "Face_Recognition/scan_compare/"

    """
    for (x, y, w, h) in faces:
        img = image[y:y+h, x:x+w]
        randnum = str(randomNum(20))
        imgdir = imagesdir + randnum
        cv2.imwrite(imgdir, img)
        unknown_image = face_recognition.load_image_file(imgdir)
        try:
            unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
        except IndexError:
            print(
                "I wasn't able to locate any faces in at least one of the images. Check the image files. Terminating program....")
            os.remove(imgdir)
            print("========================================================================================")
            continue
    """

    if faces is not None:
        randnum = str(randomNum(20))
        imgdir = imagesdir + randnum + ".jpg"
        cv2.imwrite(imgdir, image)
        unknown_image = face_recognition.load_image_file(imgdir)

        try:
            """
            p = Process(target=start_face_encoding, args=(unknown_image,))
            p.start()
            p.join()
            """
            unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
        except IndexError:
            print(
                "I wasn't able to locate any faces in at least one of the images. Check the image files. Terminating program....")
            os.remove(imgdir)
            print("========================================================================================")
            return

        os.remove(imgdir)

        data = dbfunctions.returnallfaces()
        for d in data:
            randnum = str(randomNum(20))
            dataFile = datdir + randnum + ".dat"
            s = open(dataFile, 'wb')
            s.write(d[1])
            s.close()
            with open(dataFile, 'rb') as f:
                known_faces = (pickle.load(f))

            results = face_recognition.compare_faces(known_faces, unknown_face_encoding)
            os.remove(dataFile)
            if (results[0] == True):
                global firstname, lastname, receiver_email, userid
                userid = d[0]
                firstname, lastname, receiver_email = dbfunctions.returnuser(userid)
                break


# Recognizes faces from captured frame
def facialrecognition(faceCascade, _, frame):
    global faces

    # logFile = open("log.txt", "a")
    # sys.stdout = logFile
    print("\n\n=======================================================================================")
    print("----------------------------------------------------------------------------------------")
    print("Running webcam_comparision.py")
    # Print out date and time of code execution
    todayDate = datetime.now()
    print("Date and time of code execution:", todayDate.strftime("%m/%d/%Y %I:%M:%S %p"))
    print("----------------------------------------------------------------------------------------")

    # Read the image
    image = frame
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    newfaces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(20, 20),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(newfaces) is not 0:
        print("Found {0} faces!".format(len(newfaces)))
        faces = newfaces

        print("Face found. Beginning comparision check....")
        start_facial_comparison(image)
        # threading.Thread(target=facecomparison, args=(image,)).start()
        # facecomparison(image)
        # For loop to compare patterns from webcam with .dat files
        # If comparison returns true, break from for loop

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    else:
        print(
            "I wasn't able to locate any faces in at least one of the images. Check the image files. Terminating program....")
        print("========================================================================================")
        faces = None


# Widget that displays camera
class KivyCamera(Image):
    def __init__(self, capture, fps, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.capture = capture
        self.faces = None
        self.temp = None
        self.tlimit = 1.5 * 60
        self.flimit = 20
        self.ftime = self.flimit
        self.timer = self.tlimit
        self.squarecolor = (0, 0, 0)
        # Clock.schedule_interval(partial(self.start_facial_recognition, faceCascade), 1.0 / fps)
        Clock.schedule_interval(partial(self.update, ), 1.0 / fps)

    # Updates frame on camer widget
    def update(self, *args):

        ret, frame = self.capture.read()

        if self.ftime is self.flimit:
            self.ftime = 0
            start_facial_recognition(ret, frame)
        else:
            self.ftime += 1

        global faces, userid

        temp = self.temp
        timer = self.timer
        tlimit = self.tlimit

        if faces is not None:
            for (x, y, w, h) in faces:
                if timer == tlimit:
                    if temp == None:
                        self.squarecolor = (0, 0, 0)
                    elif temp > 98.6:
                        self.timer = 0
                        self.squarecolor = (0, 0, 255)
                        if userid is not None:
                            dbfunctions.newscanhist(userid, temp, False)
                            self.start_user_mail_thread()
                            self.start_admin_mail_thread()
                        self.temp = None
                        userid = None
                        # winsound.Beep(500, 1500)
                    elif temp <= 98.6:
                        self.timer = 0
                        self.squarecolor = (0, 255, 0)
                        if userid is not None:
                            dbfunctions.newscanhist(userid, temp, True)
                        self.temp = None
                        userid = None
                cv2.rectangle(frame, (x, y), (x + w, y + h), self.squarecolor, 2)

        if self.timer != self.tlimit:
            self.timer += 1

        if Window.height - frame.shape[0] > Window.width - frame.shape[1]:
            scale_percent = Window.width / frame.shape[1]
        else:
            scale_percent = Window.height / frame.shape[0]

        width = int(frame.shape[1] * scale_percent)
        height = int(frame.shape[0] * scale_percent)
        dim = (width, height)
        resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        if ret:
            # convert it to texture
            buf1 = cv2.flip(resized, 0)
            buf = buf1.tobytes()
            image_texture = Texture.create(
                # size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
                size=(resized.shape[1], resized.shape[0]), colorfmt='bgr')

            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

            # display image from the texture
            self.texture = image_texture

    # Starts thread to send email to user
    def start_user_mail_thread(self):
        threading.Thread(target=self.usersendemail, args=()).start()

    # Starts thread to send email to admin
    def start_admin_mail_thread(self):
        threading.Thread(target=self.adminsendemail, args=()).start()

    # Send email to users
    def usersendemail(self):
        global smtp_server, sender_email, admin_email, password
        # Create a secure SSL context
        context = ssl.create_default_context()

        usermessage = """\
Subject: Corserva High Temperature Detected
        
Hello """ + firstname + """ """ + lastname + """,
        
A high temperature has been detected from the Corserva Kiosk. Please speak to nearby attendant from a manual screening."""

        with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
            server.login(sender_email, password)
            # TODO: Send email here
            server.sendmail(sender_email, receiver_email, usermessage)

    # Send email to admin/attendant
    def adminsendemail(self):
        global smtp_server, sender_email, admin_email, password
        # Create a secure SSL context
        context = ssl.create_default_context()
        adminmessage = """\
Subject: High Temperature Detected

High Temperature has been detected from user """ + firstname + """ """ + lastname + """."""
        with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
            server.login(sender_email, password)
            # TODO: Send email here
            server.sendmail(sender_email, admin_email, adminmessage)

    def temppass(self):
        self.temp = 96

    def tempnone(self):
        self.temp = None

    def tempfail(self):
        self.temp = 100


# Main Layout that displays the main page
class layout(FloatLayout):
    def __init__(self, **kwargs):
        # make sure we aren't overriding any important functionality
        super(layout, self).__init__(**kwargs)
        self.capture = cv2.VideoCapture(0)
        self.my_camera = KivyCamera(capture=self.capture, fps=30, size=Window.size)
        self.add_widget(self.my_camera)

        self.capturebutton = Button(text="Capture",
                                    size_hint=(.5, .1),
                                    pos_hint={'center_x': .5, 'y': .1},
                                    )

        self.settingsbutton = Button(background_normal='assets\settingsbuttonimage.png',
                                     # background_down='assets\settingsbuttonpressed.png',
                                     size_hint=(.2, .2),
                                     pos_hint={"x": .8, "y": 0.8}
                                     )
        self.lowtemp = Button(text="Low Temp",
                              size_hint=(.2, .1),
                              pos_hint={'center_x': .2, 'y': 0},
                              )
        self.notemp = Button(text="No Temp",
                             size_hint=(.2, .1),
                             pos_hint={'center_x': .5, 'y': 0},
                             )
        self.hightemp = Button(text="High Temp",
                               size_hint=(.2, .1),
                               pos_hint={'center_x': .8, 'y': 0},
                               )

        self.lowtemp.bind(on_press=lambda x: self.my_camera.temppass())
        self.notemp.bind(on_press=lambda x: self.my_camera.tempnone())
        self.hightemp.bind(on_press=lambda x: self.my_camera.tempfail())
        self.capturebutton.bind(on_press=lambda x: self.capturebtn())
        self.settingsbutton.bind(on_press=lambda x: self.gotosettings())

        self.add_widget(self.lowtemp)
        self.add_widget(self.notemp)
        self.add_widget(self.hightemp)
        self.add_widget(self.capturebutton)
        self.add_widget(self.settingsbutton)

    # Transitions to user registration page
    def gotosettings(self):
        app.screen_manager.transition.direction = 'left'
        app.screen_manager.current = 'settingspage'

    def on_stop(self):
        # without this, app will not exit even if the window is closed
        self.capture.release()

    # Starts thread to capture facial data
    def start_capture_thread(self, *args):
        threading.Thread(target=self.capturebtn, args=()).start()

    # Function to capture facial data from camera feed
    def capturebtn(self):
        '''
        Function to capture the images and give them the names
        according to their captured time and date.
        '''

        # logFile = open("log.txt", "a")
        # sys.stdout = logFile
        print("\n\n========================================================================================")
        print("----------------------------------------------------------------------------------------")

        timestr = time.strftime("%Y%m%d_%H%M%S")
        ret, frame = self.capture.read()
        cv2.imwrite("Face_Recognition/extract/IMG_{}.png".format(timestr), frame)
        print("Captured")

        uploadedImageLoc = 'Face_Recognition/extract/'
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
                        return

                    known_faces = [
                        known_face_encoding
                    ]

                    # Save encoding as .dat file
                    fileLength = len(imageFile) - 4
                    datExtension = ".dat"
                    if (fileLength <= 0):
                        print("ERROR! File naming failed! Terminating program....")
                        sys.stdout.close()

                    else:
                        # Generate a random 20 number code for reservation ID
                        datFileName = str(randomNum(20))
                        with open(datFolderLoc + datFileName + datExtension, 'wb') as f:
                            pickle.dump(known_faces, f)
                            print("Saved pattern data as " + datFileName + " under dat folder.")

                        global newuserid

                        dbfunctions.insertfacedb(newuserid, datFolderLoc + datFileName + datExtension)

                        os.remove(datFolderLoc + datFileName + datExtension)

                    os.remove(uploadedImageLoc + imageFile)

                else:
                    print("Image doesn't exist! Terminating program....")
                    print(
                        "========================================================================================")
            print("SUCCESS: Patterns extraction completed. Terminiating program.")
            print("========================================================================================")


class Settings_Page(FloatLayout):
    def __init__(self, **kwargs):
        # make sure we aren't overriding any important functionality
        super(Settings_Page, self).__init__(**kwargs)

        with self.canvas:
            Color(.145, .1529, .302, 1, mode='rgba')
            Rectangle(pos=self.pos, size=Window.size)

        self.registeruserbutton = Button(text="Register User",
                                         size_hint=(.5, .1),
                                         pos_hint={'center_x': .5, 'y': .6},
                                         background_color=(.4, .65, 1, 1)
                                         )

        self.adminemailbutton = Button(text="Change Admin Email",
                                       size_hint=(.5, .1),
                                       pos_hint={'center_x': .5, 'y': .3},
                                       background_color=(.4, .65, 1, 1),
                                       )

        self.registeruserbutton.bind(on_press=lambda x: self.registeruserscreen())
        self.adminemailbutton.bind(on_press=lambda x: self.adminemailscreen())

        self.add_widget(self.registeruserbutton)
        self.add_widget(self.adminemailbutton)

    def registeruserscreen(self):
        app.screen_manager.transition.direction = 'left'
        app.screen_manager.current = 'registeruserpage'

    def adminemailscreen(self):
        app.screen_manager.transition.direction = 'left'
        app.screen_manager.current = 'adminemailpage'


class Register_User_Page(FloatLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        with self.canvas:
            Color(.145, .1529, .302, 1, mode='rgba')
            Rectangle(pos=self.pos, size=Window.size)

        self.invalidemail = Label(text="",
                                  markup='true',
                                  pos_hint={'center_x': .5, 'y': .35},
                                  )

        self.submit = Button(text="Submit",
                             size_hint=(.5, .1),
                             pos_hint={'center_x': .5, 'y': .1},
                             background_color=(.4, .65, 1, 1),
                             )

        self.first = TextInput(hint_text='Please enter your first name',
                               multiline=False,
                               pos_hint={'center_x': .5, 'y': .5},
                               size_hint=(.5, .05),
                               )
        self.last = TextInput(hint_text='Please enter your last name',
                              multiline=False,
                              pos_hint={'center_x': .5, 'y': .3},
                              size_hint=(.5, .05),
                              )
        self.email = TextInput(hint_text='Please enter your email address',
                               multiline=False,
                               pos_hint={'center_x': .5, 'y': .7},
                               size_hint=(.5, .05),
                               )

        self.add_widget(self.invalidemail)
        self.add_widget(self.first)
        self.add_widget(self.last)
        self.add_widget(self.email)
        self.add_widget(self.submit)
        self.submit.bind(on_press=lambda x: self.submitregthread())

    def submitregthread(self):
        thread = threading.Thread(target=self.submitregistration)
        thread.start()

    def submitregistration(self):
        first = self.first.text
        last = self.last.text
        email = self.email.text

        is_valid = validate_email(email_address=email, check_format=True, check_blacklist=True,
                                  check_dns=True, dns_timeout=10, check_smtp=True, smtp_timeout=10,
                                  smtp_helo_host='my.host.name', smtp_from_address='my@from.addr.ess',
                                  smtp_debug=False)
        if is_valid:
            self.invalidemail.text = ""
            dbfunctions.printuser(dbfunctions.newuser(first, last, email))
            app.screen_manager.transition.direction = 'right'
            app.screen_manager.current = 'mainpage'
        else:
            self.invalidemail.text = "[color=ff3333]Please Enter a Valid Email Address[/color]"

        self.first.text = ""
        self.last.text = ""
        self.email.text = ""


class Admin_Email_Page(FloatLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        with self.canvas:
            Color(.145, .1529, .302, 1, mode='rgba')
            Rectangle(pos=self.pos, size=Window.size)

        self.submit = Button(text="Submit",
                             size_hint=(.5, .1),
                             pos_hint={'center_x': .5, 'y': .1},
                             background_color=(.4, .65, 1, 1),
                             )

        self.adminemail = TextInput(hint_text='Please enter in a new email for use as admin email',
                                    multiline=False,
                                    pos_hint={'center_x': .5, 'y': .5},
                                    size_hint=(.5, .05),
                                    )

        self.invalidemail = Label(text="",
                                  markup='true',
                                  pos_hint={'center_x': .5, 'y': .35},
                                  )

        self.add_widget(self.invalidemail)
        self.add_widget(self.adminemail)
        self.add_widget(self.submit)

        self.submit.bind(on_press=lambda x: self.submitadminemail())

    def submitadminemail(self):
        thread = threading.Thread(target=self.changeadminemail)
        thread.start()

    def changeadminemail(self):
        global admin_email
        email = self.adminemail.text

        if email == admin_email:
            print("email exists")
            self.invalidemail.text = "[color=ff3333]Email is Already in Use[/color]"
            self.adminemail.text = ""
            return

        is_valid = validate_email(email_address=email, check_format=True, check_blacklist=True,
                                  check_dns=True, dns_timeout=10, check_smtp=True, smtp_timeout=10,
                                  smtp_helo_host='my.host.name', smtp_from_address='my@from.addr.ess',
                                  smtp_debug=False)

        if is_valid:

            self.invalidemail.text = ""
            admin_email = email
            file = open("config.txt", 'w')
            file.write("admin_email = " + email)
            file.close()
            app.screen_manager.transition.direction = 'right'
            app.screen_manager.current = 'mainpage'

        else:
            self.invalidemail.text = "[color=ff3333]Please Enter a Valid Email Address[/color]"

        self.adminemail.text = ""


class ThermalApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.screen_manager = None

    def build(self):
        dbfunctions.deletedb()

        self.screen_manager = ScreenManager()

        if verifyfirstinstall():
            self.MainPage = layout()
            screen = Screen(name='mainpage')
            screen.add_widget(self.MainPage)
            self.screen_manager.add_widget(screen)

            self.adminemailpage = Admin_Email_Page()
            screen = Screen(name='adminemailpage')
            screen.add_widget(self.adminemailpage)
            self.screen_manager.add_widget(screen)

        else:
            self.adminemailpage = Admin_Email_Page()
            screen = Screen(name='adminemailpage')
            screen.add_widget(self.adminemailpage)
            self.screen_manager.add_widget(screen)

            self.MainPage = layout()
            screen = Screen(name='mainpage')
            screen.add_widget(self.MainPage)
            self.screen_manager.add_widget(screen)

        self.settingspage = Settings_Page()
        screen = Screen(name='settingspage')
        screen.add_widget(self.settingspage)
        self.screen_manager.add_widget(screen)

        self.userregistrationpage = Register_User_Page()
        screen = Screen(name='registeruserpage')
        screen.add_widget(self.userregistrationpage)
        self.screen_manager.add_widget(screen)

        return self.screen_manager


if __name__ == '__main__':
    app = ThermalApp()
    app.run()
