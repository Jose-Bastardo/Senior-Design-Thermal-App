import difflib
import mariadb
import dbfunctions

dbfunctions.deletedb()

userid = dbfunctions.newuser("luke", "smith", "testemail122@gmail.com", "password")
dbfunctions.printuser(userid)

dbfunctions.insertfacedb(userid, "Face_Recognition/dat/47245376683007053545.dat")
facedata = dbfunctions.returnface(userid)
datbits = open("Face_Recognition/dat/47245376683007053545.dat", "rb")

print(facedata)
print(datbits.read())

d = difflib.Differ()
e = d.compare(facedata, datbits.read())

if e:
    print("success")
else:
    print("fail")
