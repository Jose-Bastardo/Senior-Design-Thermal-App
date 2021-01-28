import mariadb
import datetime


# Insert new user into database
def newuser(firstname, lastname, email, password):
    # Establish connection to database
    connection = mariadb.connect(user="admin", password="2901567j", database="thermalapp",
                                 host="thermal-app.ckyrcuyndxij.us-east-2.rds.amazonaws.com")
    cursor = connection.cursor()
    # Insert new user information
    cursor.execute("INSERT INTO user(firstname, lastname, email, password) VALUES (?,?,?,?)",
                   (firstname, lastname, email, password))
    connection.commit()

    cursor.close()
    connection.close()


# Insert facial recognition file into database
def insertfacedb(userid, file):
    # Establish connection to database
    connection = mariadb.connect(user="admin", password="2901567j", database="thermalapp",
                                 host="thermal-app.ckyrcuyndxij.us-east-2.rds.amazonaws.com")
    cursor = connection.cursor()

    dat = open(file, "rb")
    # Insert new user information
    cursor.execute("INSERT INTO dat(userid, datfile) VALUES (?,?)",
                       (userid, dat.read()))
    connection.commit()

    cursor.close()
    connection.close()

def returnface(datid):
    # Establish connection to database
    connection = mariadb.connect(user="admin", password="2901567j", database="thermalapp",
                                 host="thermal-app.ckyrcuyndxij.us-east-2.rds.amazonaws.com")
    cursor = connection.cursor()

    cursor.execute("Select datfile from dat "
                   "where datid = " + datid)

    dat = cursor.fetchone()

    cursor.close()
    connection.close()

    return dat[0]

# Print user information
def printuser(userid):
    # Establish connection to database
    connection = mariadb.connect(user="admin", password="2901567j", database="thermalapp",
                                 host="thermal-app.ckyrcuyndxij.us-east-2.rds.amazonaws.com")
    cursor = connection.cursor()

    cursor.execute("SELECT userid, firstname, lastname, email, password FROM user "
                   "where userid = " + userid)

    # print data
    row = cursor.fetchone()
    print(row)
    print(*row, sep='\t')
    print(row[1])

    cursor.close()
    connection.close()


def newscanhist(userid, temp, passed):
    x = datetime.datetime.now()

    # Establish connection to database
    connection = mariadb.connect(user="admin", password="2901567j", database="thermalapp",
                                 host="thermal-app.ckyrcuyndxij.us-east-2.rds.amazonaws.com")
    cursor = connection.cursor()

    cursor.execute("INSERT INTO history(userid, temp, date, time, passed) VALUES(?,?,STR_TO_DATE(?, '%m/%d/%y'),?,?)",
                    (userid, temp, x.strftime("%x"), x.strftime("%X"), passed))
    connection.commit()

    cursor.close()
    connection.close()
