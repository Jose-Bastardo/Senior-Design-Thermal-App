import mariadb

#file = open("Face_Recognition_Test/dat/23851874046344168675.dat", "rb")
#binary = file.read()
connection = mariadb.connect(user="admin", password="2901567j", database="thermalapp", host="thermal-app.ckyrcuyndxij.us-east-2.rds.amazonaws.com")

cursor = connection.cursor()

cursor.execute("INSERT INTO user(firstname, lastname, email, password) VALUES (?,?,?,?)",
               ("John", "Smith", "johnsmith@hotmail.com", "passwordtest"))
#file.close()

# retrieve data from user table
cursor.execute("SELECT userid, firstname, lastname, email, password FROM user")

# different ways of printing dataprint data
row = cursor.fetchone()
print(row)
print(*row, sep='\t')
print(row[1])

#file = open("Face_Recognition_Test/extract/test.dat", "wb")
#file.write(row[5])
#file.close()

# free resources
cursor.close()
connection.close()