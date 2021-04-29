from import_clr import *
clr.AddReference("ManagedIR16Filters")
from Lepton import CCI
from IR16Filters import IR16Capture, NewIR16FrameEvent, NewBytesFrameEvent
import numpy

# Frame callback function
# Will be called everytime a new frame comes in from the camera
numpyArr = None
def getFrameRaw(arr, width, height):
    global numpyArr
    numpyArr = numpy.fromiter(arr, dtype="uint16").reshape(height, width)
    
# Build an IR16 capture device
capture = IR16Capture()
capture.SetupGraphWithBytesCallback(NewBytesFrameEvent(getFrameRaw))
capture.RunGraph()

while numpyArr is None:
    time.sleep(.1)

def get_image_array():
    return numpyArr
    
    
