#============================================================================================
#Facial recognition code copied and modified from Github
#Link: https://github.com/ageitgey/face_recognition/blob/master/examples/recognize_faces_in_pictures.py
#Author: Adam Geitgey (ageitgey)
#Date copied: February 6, 2020
#============================================================================================

#Get patterns from stored files in UploadedImage folder
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os #Used to destroy image near the end
from os import path
import pickle #Used to save face encoding 
import face_recognition #Used to access various facial recognition libaries and functions
import sys #Used for sys.exit()
from datetime import datetime #Used to print date on log.txt
import random #Used to generate random reservation ID
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#Part 1: System runs faical recognition on uploaded images within its folder
#and save patterns into .dat files
uploadedImageLoc = 'Face_Recognition/uploadedImages/'
datFolderLoc = 'Face_Recognition/dat/'

#Create random number based on specified length of n
def randomNum(n):
    min = pow(10, n-1)
    max = pow(10, n) - 1
    return random.randint(min, max)

#Create or open log.txt to be written
logFile = open("log.txt", "a")
sys.stdout = logFile
print("\n\n========================================================================================")
print("----------------------------------------------------------------------------------------")
print("Running uploaded images' patterns extraction")
#Print out date and time of code execution
todayDate = datetime.now()
print ("Date and time of code execution:", todayDate.strftime("%m/%d/%Y %I:%M:%S %p"))
print("----------------------------------------------------------------------------------------")

#Check if upload folder directory is valid
if(path.exists(uploadedImageLoc) == False):
    print('Folder path does not exists. Terminating program....')
    sys.exit()

else:
    #For loop to read images on uploadedImages folder
    for imageFile in os.listdir(uploadedImageLoc): 
        print("Loading image:", imageFile)
        
        #checks if the file does exists
        if os.path.isfile(uploadedImageLoc + imageFile): 
            #print("File located! Executing facial pattern extraction")

            #start of encoding
            known_image = face_recognition.load_image_file(uploadedImageLoc + imageFile)
            try: 
                known_face_encoding = face_recognition.face_encodings(known_image)[0]
            except IndexError:
                print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
                sys.stdout.close()
                sys.exit()
        
            known_faces = [
                known_face_encoding
                ]
    
            #Save encoding as .dat file
            fileLength = len(imageFile) - 4
            datExtension = ".dat"
            if(fileLength <= 0):
                print("ERROR! File naming failed! Terminating program....")
                sys.stdout.close()
                sys.exit()
                
            else:
                #Generate a random 20 number code for reservation ID
                datFileName = str(randomNum(20))
                with open(datFolderLoc + datFileName + datExtension, 'wb') as f:
                    pickle.dump(known_faces, f)
                    print("Saved pattern data as " + datFileName + " under dat folder.")
                os.remove(uploadedImageLoc + imageFile)
                #Send number to the hotel database
                #OR, do this after booking

        else:
            print("Image doesn't exist! Terminating program....")
            print("========================================================================================")
            sys.exit()
    print("SUCCESS: Patterns extraction completed. Terminiating program.")
    print("========================================================================================")
    sys.exit()
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++