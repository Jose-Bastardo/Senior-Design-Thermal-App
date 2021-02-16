import difflib
import mariadb
import dbfunctions

userid = dbfunctions.newuser("luke", "smith", "testemail122@gmail.com", "password")
dbfunctions.printuser(userid)

dbfunctions.insertfacedb(31, "Face_Recognition/dat/47245376683007053545.dat")
facedata = dbfunctions.returnface(31)
datbits = open("Face_Recognition/dat/47245376683007053545.dat", "rb")

print(facedata)
print(datbits.read())

d = difflib.Differ()
e = d.compare(facedata, datbits.read())

if e:
    print("success")
else:
    print("fail")
