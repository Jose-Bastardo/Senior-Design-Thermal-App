import numpy
from matplotlib import pyplot as plt
from matplotlib import cm
import cv2

from import_clr import *
clr.AddReference("ManagedIR16Filters")
from Lepton import CCI
from IR16Filters import IR16Capture, NewIR16FrameEvent, NewBytesFrameEvent

from System.Drawing import ImageConverter
from System import Array, Byte
from capture_image import *

class ThermalCamera:
    # Known constant emitter temp for calibration
    global emitter_temp
    emitter_temp = 98.7

    # Thermal drift correction
    global correction 
    correction = 0
    
    def __init__():        
        found_device = None
        for device in CCI.GetDevices():
            if device.Name.startswith("PureThermal"):               
                found_device = device
            break

        if not found_device:
            print("Couldn't find Lepton device")
        else:
            lep = found_device.Open()

            # Set manual FFC
            lep.sys.FfcShutterMode = 0
            
            # Open the shutter
            lep.sys.SetShutterPosition(CCI.Sys.ShutterPosition.OPEN);
        
            # Raw14 is the default video output format
            # 16 bits per pixel of which the two most significant bits are zero, except in TLinear mode
            lep.vid.SetVideoOutputFormat(CCI.Vid.VideoOutputFormat.RAW14);

            # Disable telemetry
            lep.sys.TelemetryEnableState = 0
        
            # Enable radiometry
            # Radiometry must be enabled to access the related calibration and software features
            # such as TLinear and Spotmeter
            lep.rad.SetEnableState(CCI.Rad.Enable.ENABLE);
            
            lep.oem.SetVideoOutputEnable(CCI.Oem.VideoOutputEnable.ENABLE);
        
            # Enable camera shutdown to protect heating beyond operational temperature range
            lep.oem.GetThermalShutdownEnableChecked().oemThermalShutdownEnable = CCI.Oem.State.ENABLE;
        
    def get_camera_up_time():
        return lep.sys.GetCameraUpTime()

    def set_manual_ffc():
        lep.sys.FfcShutterMode = 0

    def set_auto_ffc():
        lep.sys.FfcShutterMode = 1

    def run_ffc():
        lep.sys.RunFFCNormalization()
    
    def enable_telemetry():        
        lep.sys.TelemetryEnableState = 1
        
    def disable_telemetry():
        lep.sys.TelemetryEnableState = 0
        
    def centikelvin_to_celsius(t):
        return (t - 27315) / 100
    
    def to_fahrenheit(ck):      
        c = ThermalCamera.centikelvin_to_celsius(ck)
        return c * 9 / 5 + 32

    # Capture an image
    # Display the image on <plt> if <display> is True
    def capture_image(plt, display):      
        if display is True:
            fig = plt.imshow(get_image_array(), cmap=cm.gray)
            
            plt.gca().set_axis_off()
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.savefig("capture.jpeg", dpi=144)
            img = cv2.imread('capture.jpeg', cv2.IMREAD_UNCHANGED)
            img_resized = cv2.resize(img,(160,120), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite("capture_resized.jpeg", img_resized)
            plt.show(block=False)
                  
    # Return max temperature of all pixels
    # Call capture_image() before this function
    def get_max_temp():
        try:
            return ThermalCamera.to_fahrenheit(numpyArr.max()) + correction
        finally:
            print("No image data to get_max_temp()")
        
    # Return average temperature of all pixels
    # Call capture_image() before this function
    def get_avg_temp():
        try:
            return ThermalCamera.to_fahrenheit(numpyArr.mean()) + correction
        finally:
            print("No image data to get_avg_temp()")

    def calibrate():
        ThermalCamera.capture_image(plt, True)
        capture = numpyArr

        img_rgb = cv2.imread('capture_resized.jpeg') 
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

        template = cv2.imread('template.jpeg',0)
        w, h = template.shape[::-1] 
        res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED) 
        threshold = 0.7
        loc = numpy.where(res >= threshold)
         
        for pt in zip(*loc[::-1]): 
            cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    
            cv2.imwrite('found.jpeg', img_rgb)

            roi = []
            for row in range(pt[0], pt[0] + w):
                for col in range(pt[1], pt[1] + h):
                    roi = numpy.append(roi, numpyArr[row][col])

            roi_max_temp = ThermalCamera.to_fahrenheit(roi.max())
            global correction
            correction = emitter_temp - roi_max_temp
            break
