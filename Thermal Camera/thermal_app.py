from thermal_camera import ThermalCamera
from matplotlib import pyplot as plt

cam = ThermalCamera
#cam.calibrate()
cam.capture_image(plt, True)
