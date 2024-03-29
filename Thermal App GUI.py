import threading
from multiprocessing.context import Process
import cv2
import dlib
import winsound
import time
import random
from datetime import datetime
import sys
import pickle
import os
from os import path
import smtplib, ssl

from kivy.graphics.context_instructions import Color
from kivy.graphics.vertex_instructions import Rectangle
from kivy.uix.label import Label
from validate_email import validate_email

port = 465  # For SSL
admin_email = None
smtp_server = "smtp.gmail.com"
global receiver_email  # Enter receiver address
password = None
global firstname
global lastname
global newuserid
facethread = None
camthread = None
comparisonthread = None
userid = None
faces = None
global unknown_face_encoding
camtexture = None
perffacecomp = False
captureregistration = False
cthread = None
stop_cthread = False
from kivy.core.window import Window

cascPath = "Face_Recognition/haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

Window.fullscreen = True

def verifyfirstinstall():
    dir = "config.txt"
    if (path.isfile(dir)):
        getadminemail()
        return 1
    else:
        return 0


# Updates frame on camer widget
def update(ret, frame):
    global faces, userid

    temp = app.MainPage.my_camera.temp
    timer = app.MainPage.my_camera.timer
    tlimit = app.MainPage.my_camera.tlimit

    if faces is not None:
        for (x, y, w, h) in faces:
            if timer == tlimit:
                if temp == None:
                    squarecolor = (0, 0, 0)
                elif temp > 98.6:
                    app.MainPage.my_camera.timer = 0
                    app.MainPage.my_camera.squarecolor = (0, 0, 255)
                    if userid is not None:
                        dbfunctions.newscanhist(userid, temp, False)
                        app.MainPage.my_camera.start_user_mail_thread()
                        app.MainPage.my_camera.start_admin_mail_thread()
                    app.MainPage.my_camera.temp = None
                    userid = None
                    # winsound.Beep(500, 1500)
                elif temp <= 98.6:
                    app.MainPage.my_camera.timer = 0
                    app.MainPage.my_camera.squarecolor = (0, 255, 0)
                    if userid is not None:
                        dbfunctions.newscanhist(userid, temp, True)
                    app.MainPage.my_camera.temp = None
                    userid = None
            cv2.rectangle(frame, (x, y), (x + w, y + h), app.MainPage.my_camera.squarecolor, 2)

    if app.MainPage.my_camera.timer != app.MainPage.my_camera.tlimit:
        app.MainPage.my_camera.timer += 1

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

        # display image from the texture
        Clock.schedule_once(partial(app.MainPage.my_camera.updateTexture, image_texture, buf))


# Get saved admin email from text file
def getadminemail():
    global admin_email, password
    dir = "config.txt"
    with open(dir) as fp:
        line = fp.readline()
        while line:
            if (line.find("admin_email") >= 0):
                x = line.split("= ")
                admin_email = x[1]
            if (line.find("password") >= 0):
                x = line.split("= ")
                password = x[1]
                fp.close()
                break
            else:
                line = fp.readline()


def start_cam_update(_, frame):
    if frame is None:
        return

    global camthread

    if (camthread == None):
        camthread = threading.Thread(target=update,
                                     args=(_, frame,))
        camthread.start()
    elif camthread.is_alive():
        print(camthread.is_alive())
        return
    else:
        camthread = threading.Thread(target=update, args=(_, frame,))
        camthread.start()


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


def facecomparisonpool(image):
    p = Process(facecomparison(image))
    p.start()
    p.join()


# Compares detected faces to facial data in database
def facecomparison(image):
    import face_recognition

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
        unknown_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # the facial embeddings for face in input
        # unknown_image = face_recognition.face_encodings(rgb)

        try:
            unknown_face_encoding = face_recognition.face_encodings(unknown_image)
        except IndexError:
            print(
                "I wasn't able to locate any faces in at least one of the images. Check the image files. Terminating program....")
            print("========================================================================================")
            return

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

        if perffacecomp:
            print("Face found. Beginning comparision check....")
            start_facial_comparison(image)
            # threading.Thread(target=facecomparison, args=(image,)).start()
            # facecomparison(image)
            # For loop to compare patterns from webcam with .dat files
            # If comparison returns true, break from for loop

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    else:
        faces = None


if __name__ == '__main__':
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
    from kivy.uix.screenmanager import ScreenManager, Screen
    from kivy.uix.textinput import TextInput

    Window.fullscreen = True


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
            self.frame = None
            Clock.schedule_interval(partial(self.update, ), 1.0 / fps)

        # Updates frame on camer widget
        def update(self, *args):

            ret, self.frame = self.capture.read()

            if self.ftime is self.flimit:
                self.ftime = 0
                start_facial_recognition(ret, self.frame)
            else:
                self.ftime += 1

            global faces, userid

            temp = self.temp
            timer = self.timer
            tlimit = self.tlimit

            if faces is not None:
                for (x, y, w, h) in faces:
                    if not captureregistration:
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
                    cv2.rectangle(self.frame, (x, y), (x + w, y + h), self.squarecolor, 2)

            if self.timer != self.tlimit:
                self.timer += 1

            if Window.height - self.frame.shape[0] > Window.width - self.frame.shape[1]:
                scale_percent = Window.width / self.frame.shape[1]
            else:
                scale_percent = Window.height / self.frame.shape[0]

            width = int(self.frame.shape[1] * scale_percent)
            height = int(self.frame.shape[0] * scale_percent)
            dim = (width, height)
            resized = cv2.resize(self.frame, dim, interpolation=cv2.INTER_AREA)

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
            global smtp_server, admin_email, password
            # Create a secure SSL context
            context = ssl.create_default_context()

            usermessage = """\
Subject: Corserva High Temperature Detected

Hello """ + firstname + """ """ + lastname + """,

A high temperature has been detected from the Corserva Kiosk. Please speak to nearby attendant from a manual screening."""

            with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
                server.login(admin_email, password)
                # TODO: Send email here
                server.sendmail(admin_email, receiver_email, usermessage)

        # Send email to admin/attendant
        def adminsendemail(self):
            global smtp_server, admin_email, password
            # Create a secure SSL context
            context = ssl.create_default_context()
            adminmessage = """\
Subject: High Temperature Detected

High Temperature has been detected from user """ + firstname + """ """ + lastname + """."""
            with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
                server.login(admin_email, password)
                # TODO: Send email here
                server.sendmail(admin_email, admin_email, adminmessage)

        def start_capture_thread(self, first, last, email):
            global cthread
            cthread = threading.Thread(target=self.capturefunc, args=(first, last, email))
            cthread.start()

        def capturefunc(self, first, last, email):
            global stop_cthread
            print("Started Capture Thread")
            app.MainPage.textlabel.text = "Please Face Camera"
            # logFile = open("log.txt", "a")
            # sys.stdout = logFile
            i = 5
            time.sleep(2)
            for x in range(5):
                if stop_cthread:
                    app.screen_manager.transition.direction = 'right'
                    app.screen_manager.current = 'registeruserpage'
                    app.MainPage.textlabel.text = ""
                    app.MainPage.add_widget(app.MainPage.lowtemp)
                    app.MainPage.add_widget(app.MainPage.notemp)
                    app.MainPage.add_widget(app.MainPage.hightemp)
                    app.MainPage.remove_widget(app.MainPage.backbutton)
                    app.MainPage.add_widget(app.MainPage.settingsbutton)
                    return

                app.MainPage.textlabel.text = "[color=ffffff]" + str(i) + "[/color]"
                i -= 1
                time.sleep(1)

            if stop_cthread:
                app.screen_manager.transition.direction = 'right'
                app.screen_manager.current = 'registeruserpage'
                app.MainPage.textlabel.text = ""
                app.MainPage.add_widget(app.MainPage.lowtemp)
                app.MainPage.add_widget(app.MainPage.notemp)
                app.MainPage.add_widget(app.MainPage.hightemp)
                app.MainPage.remove_widget(app.MainPage.backbutton)
                app.MainPage.add_widget(app.MainPage.settingsbutton)
                return

            app.MainPage.textlabel.text = "[color=ffffff]Starting Capture[/color]"

            if faces is None:
                app.MainPage.textlabel.text = "[color=ffffff]Face not Found[/color]"
                time.sleep(2)
                self.start_capture_thread(first, last, email)
                return
            else:
                ret, frame = self.capture.read()

                # Create random number based on specified length of n
                def randomNum(n):
                    min = pow(10, n - 1)
                    max = pow(10, n) - 1
                    return random.randint(min, max)

                datFolderLoc = "./Face_Recognition/dat"
                datExtension = ".dat"
                datFileName = str(randomNum(5))

                import face_recognition
                known_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    known_face_encoding = face_recognition.face_encodings(known_image)[0]
                except IndexError:
                    print(
                        "I wasn't able to locate any faces in at least one of the images.")
                    return

                # Save encoding as .dat file

                with open(datFolderLoc + datFileName + datExtension, 'wb') as f:
                    pickle.dump(known_face_encoding, f)
                    print("Saved pattern data as " + datFileName + " under dat folder.")

                global newuserid

                dbfunctions.insertfacedb(dbfunctions.newuser(first, last, email),
                                         datFolderLoc + datFileName + datExtension)

                os.remove(datFolderLoc + datFileName + datExtension)

                print("SUCCESS: Patterns extraction completed. Terminiating program.")
                print("========================================================================================")

                app.screen_manager.transition.direction = 'left'
                app.screen_manager.current = 'successpage'
                time.sleep(5)
                app.MainPage.textlabel.text = ""
                app.MainPage.add_widget(app.MainPage.lowtemp)
                app.MainPage.add_widget(app.MainPage.notemp)
                app.MainPage.add_widget(app.MainPage.hightemp)
                app.MainPage.remove_widget(app.MainPage.backbutton)
                app.MainPage.add_widget(app.MainPage.settingsbutton)
                app.screen_manager.transition.direction = 'left'
                app.screen_manager.current = 'mainpage'

                global perffacecomp
                perffacecomp = True

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
            self.textlabel = Label(text="",
                                   markup=True,
                                   pos_hint={'center_x': .5, 'y': .4},
                                   font_size="20sp",
                                   )

            self.backbutton = Button(text="Back",
                                     size_hint=(.2, .1),
                                     pos_hint={'center_x': .8, 'y': .8},
                                     background_color=(.4, .65, 1, 1),
                                     )

            self.lowtemp.bind(on_press=lambda x: self.my_camera.temppass())
            self.notemp.bind(on_press=lambda x: self.my_camera.tempnone())
            self.hightemp.bind(on_press=lambda x: self.my_camera.tempfail())
            self.settingsbutton.bind(on_press=lambda x: self.gotosettings())
            self.backbutton.bind(on_press=lambda x: self.gobackthread())

            self.add_widget(self.textlabel)
            self.add_widget(self.lowtemp)
            self.add_widget(self.notemp)
            self.add_widget(self.hightemp)
            self.add_widget(self.settingsbutton)

        def gobackthread(self):
            threading.Thread(target=self.goback, ).start()

        def goback(self):
            global stop_cthread, cthread
            stop_cthread = True
            cthread.join()

        # Transitions to user registration page
        def gotosettings(self):
            global perffacecomp
            perffacecomp = False
            app.screen_manager.transition.direction = 'left'
            app.screen_manager.current = 'settingspage'

        def on_stop(self):
            # without this, app will not exit even if the window is closed
            self.capture.release()


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

            self.backbutton = Button(text="Back",
                                     size_hint=(.2, .1),
                                     pos_hint={'center_x': .8, 'y': .8},
                                     background_color=(.4, .65, 1, 1),
                                     )

            self.registeruserbutton.bind(on_press=lambda x: self.registeruserscreen())
            self.adminemailbutton.bind(on_press=lambda x: self.adminemailscreen())
            self.backbutton.bind(on_press=lambda x: self.mainscreen())

            self.add_widget(self.registeruserbutton)
            self.add_widget(self.adminemailbutton)
            self.add_widget(self.backbutton)

        def registeruserscreen(self):
            app.screen_manager.transition.direction = 'left'
            app.screen_manager.current = 'registeruserpage'

        def adminemailscreen(self):
            app.adminemailpage.backbutton.disabled = False
            app.screen_manager.transition.direction = 'left'
            app.screen_manager.current = 'adminemailpage'

        def mainscreen(self):
            global perffacecomp
            perffacecomp = True
            app.screen_manager.transition.direction = 'right'
            app.screen_manager.current = 'mainpage'


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
            self.backbutton = Button(text="Back",
                                     size_hint=(.2, .1),
                                     pos_hint={'center_x': .8, 'y': .8},
                                     background_color=(.4, .65, 1, 1),
                                     )

            self.add_widget(self.invalidemail)
            self.add_widget(self.first)
            self.add_widget(self.last)
            self.add_widget(self.email)
            self.add_widget(self.submit)
            self.add_widget(self.backbutton)
            self.backbutton.bind(on_press=lambda x: self.gotosettings())
            self.submit.bind(on_press=lambda x: self.submitregthread())

        def submitregthread(self):
            thread = threading.Thread(target=self.submitregistration)
            thread.start()

        def submitregistration(self):
            first = self.first.text
            last = self.last.text
            email = self.email.text
            empty = ""

            is_valid = validate_email(email_address=email, check_format=True, check_blacklist=True,
                                      check_dns=True, dns_timeout=10, check_smtp=True, smtp_timeout=10,
                                      smtp_helo_host='my.host.name', smtp_from_address='my@from.addr.ess',
                                      smtp_debug=False)
            if first is empty or last is empty or email is empty:
                self.invalidemail.text = "[color=ff3333]Please Enter All Fields[/color]"
            elif is_valid:
                self.invalidemail.text = ""
                app.MainPage.my_camera.squarecolor = (0, 0, 0)
                app.MainPage.remove_widget(app.MainPage.settingsbutton)
                app.MainPage.remove_widget(app.MainPage.lowtemp)
                app.MainPage.remove_widget(app.MainPage.notemp)
                app.MainPage.remove_widget(app.MainPage.hightemp)
                app.MainPage.add_widget(app.MainPage.backbutton)
                global stop_cthread
                stop_cthread = False
                app.screen_manager.transition.direction = 'left'
                app.screen_manager.current = 'mainpage'
                app.MainPage.my_camera.start_capture_thread(first, last, email)
            else:
                self.invalidemail.text = "[color=ff3333]Please Enter a Valid Email Address[/color]"

            self.first.text = ""
            self.last.text = ""
            self.email.text = ""

        def gotosettings(self):
            app.screen_manager.transition.direction = 'right'
            app.screen_manager.current = 'settingspage'
            self.first.text = ""
            self.last.text = ""
            self.email.text = ""


    class Reg_Success_Page(FloatLayout):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            with self.canvas:
                Color(.145, .1529, .302, 1, mode='rgba')
                Rectangle(pos=self.pos, size=Window.size)

            self.sucesstext = Label(text="[color=ffffff]Registration Successful[/color]",
                                    markup='true',
                                    pos_hint={'center_x': .5, 'y': 0},
                                    font_size="30sp",
                                    )

            self.add_widget(self.sucesstext)


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
                                        pos_hint={'center_x': .5, 'y': .6},
                                        size_hint=(.5, .05),
                                        )
            self.adminemailpass = TextInput(hint_text='Please enter password for email account',
                                            multiline=False,
                                            pos_hint={'center_x': .5, 'y': .4},
                                            size_hint=(.5, .05),
                                            password=True
                                            )

            self.invalidemail = Label(text="",
                                      markup='true',
                                      pos_hint={'center_x': .5, 'y': .35},
                                      )
            self.backbutton = Button(text="Back",
                                     size_hint=(.2, .1),
                                     pos_hint={'center_x': .8, 'y': .8},
                                     background_color=(.4, .65, 1, 1),
                                     )

            self.add_widget(self.invalidemail)
            self.add_widget(self.adminemailpass)
            self.add_widget(self.adminemail)
            self.add_widget(self.submit)
            self.add_widget(self.backbutton)
            self.backbutton.bind(on_press=lambda x: self.gotosettings())
            self.submit.bind(on_press=lambda x: self.submitadminemail())

        def submitadminemail(self):
            thread = threading.Thread(target=self.changeadminemail)
            thread.start()

        def gotosettings(self):
            app.screen_manager.transition.direction = 'right'
            app.screen_manager.current = 'settingspage'
            self.adminemail.text = ""
            self.adminemailpass.text = ""

        def changeadminemail(self):
            global admin_email, password
            email = self.adminemail.text
            passw = self.adminemailpass.text
            self.adminemail.text = ""
            self.adminemailpass.text = ""

            if email == admin_email:
                print("email exists")
                self.invalidemail.text = "[color=ff3333]Email is Already in Use[/color]"
                self.adminemail.text = ""
                self.adminemailpass.text = ""
                return

            is_valid = validate_email(email_address=email, check_format=True, check_blacklist=True,
                                      check_dns=True, dns_timeout=10, check_smtp=True, smtp_timeout=10,
                                      smtp_helo_host='my.host.name', smtp_from_address='my@from.addr.ess',
                                      smtp_debug=False)

            if is_valid:

                global smtp_server
                # Create a secure SSL context
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
                    try:
                        server.login(email, passw)
                    except:

                        self.invalidemail.text = "[color=ff3333]Invalid Email and Password Combination[/color]"
                        return
                    finally:
                        server.quit()

                self.invalidemail.text = ""
                admin_email = email
                password = passw
                file = open("config.txt", 'w')
                file.write("admin_email = " + email + "\n")
                file.write("password = " + passw + "\n")
                file.close()
                app.screen_manager.transition.direction = 'right'
                app.screen_manager.current = 'mainpage'
                global perffacecomp
                perffacecomp = True

            else:
                self.invalidemail.text = "[color=ff3333]Please Enter a Valid Email Address[/color]"

            self.adminemail.text = ""
            self.adminemailpass.text = ""


    class ThermalApp(App):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.screen_manager = ScreenManager()
            global perffacecomp

            dbfunctions.deletedb()

            if verifyfirstinstall():
                self.MainPage = layout()
                screen = Screen(name='mainpage')
                screen.add_widget(self.MainPage)
                self.screen_manager.add_widget(screen)

                self.adminemailpage = Admin_Email_Page()
                screen = Screen(name='adminemailpage')
                screen.add_widget(self.adminemailpage)
                self.screen_manager.add_widget(screen)

                perffacecomp = True

            else:
                self.adminemailpage = Admin_Email_Page()
                screen = Screen(name='adminemailpage')
                screen.add_widget(self.adminemailpage)
                self.screen_manager.add_widget(screen)

                self.MainPage = layout()
                screen = Screen(name='mainpage')
                screen.add_widget(self.MainPage)
                self.screen_manager.add_widget(screen)

                self.adminemailpage.backbutton.disabled = True

                perffacecomp = False

            self.settingspage = Settings_Page()
            screen = Screen(name='settingspage')
            screen.add_widget(self.settingspage)
            self.screen_manager.add_widget(screen)

            self.userregistrationpage = Register_User_Page()
            screen = Screen(name='registeruserpage')
            screen.add_widget(self.userregistrationpage)
            self.screen_manager.add_widget(screen)

            self.successpage = Reg_Success_Page()
            screen = Screen(name='successpage')
            screen.add_widget(self.successpage)
            self.screen_manager.add_widget(screen)

        def build(self):
            return self.screen_manager


    app = ThermalApp()
    app.run()
