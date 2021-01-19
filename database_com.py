import mariadb

file = open("Face_Recognition_Test/dat/23851874046344168675.dat", "rb")
binary = file.read()
connection = mariadb.connect(user="admin", password="2901567j", database="test", host="thermal-app.ckyrcuyndxij.us-east-2.rds.amazonaws.com")

cursor = connection.cursor()

cursor.execute("DROP TABLE IF EXISTS user")
cursor.execute("CREATE TABLE user(id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,"
               "first_name VARCHAR(100), last_name VARCHAR(100), email VARCHAR(100), password VARCHAR(100), dat BLOB)")

cursor.execute("INSERT INTO user(first_name, last_name, email, password, dat) VALUES (?,?,?,?,?)",
               ("John", "Smith", "johnsmith@hotmail.com", "passwordtest", binary))
file.close()

# retrieve data
cursor.execute("SELECT id, first_name, last_name, email, password, dat FROM user")

# print content
row= cursor.fetchone()
print(row)
print(*row, sep='\t')
print(row[1])

file = open("Face_Recognition_Test/extract/test.dat", "wb")
file.write(row[5])
file.close()

# free resources
cursor.close()
connection.close()