import mariadb
import dbfunctions

userid = dbfunctions.newuser("john", "smith", "testemail122@gmail.com", "password")
dbfunctions.printuser(userid)

data = dbfunctions.returnallfaces()

print(data)