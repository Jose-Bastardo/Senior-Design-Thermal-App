import kivy
import mariadb
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
import time
from kivy.config import Config
from kivy.core.window import Window
Window.fullscreen = True

Builder.load_string('''
<CameraClick>:
    orientation: 'vertical'
    Camera:
        id: camera
        resolution: (400, 400)
        play: True
    Button:
        text: 'Capture'
        size_hint_y: None
        height: '48dp'
        on_press: root.capture()
''')


class CameraClick(BoxLayout):
    def capture(self):
        '''
        Function to capture the images and give them the names
        according to their captured time and date.
        '''
        camera = self.ids['camera']
        timestr = time.strftime("%Y%m%d_%H%M%S")
        camera.export_to_png("Face_Recognition/images/IMG_{}.png".format(timestr))
        print("Captured")


class TestCamera(App):

    def build(self):
        return CameraClick()


TestCamera().run()