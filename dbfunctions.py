import mariadb
import datetime
from databaseinfo import database

#delete all data from database for testing
def deletedb():
    connection = mariadb.connect(user=database.username, password=database.password, database=database.name,
                                 host=database.host)
    cursor = connection.cursor()

    cursor.execute("Delete from history")
    cursor.execute("Delete from dat")
    cursor.execute("Delete from user")
    connection.commit()
    connection.close()

# Insert new user into database
def newuser(firstname, lastname, email):
    # Establish connection to database
    connection = mariadb.connect(user=database.username, password=database.password, database=database.name,
                                 host=database.host)
    cursor = connection.cursor()
    # Insert new user information

    cursor.execute("Select userid from user "
                   " where firstname = '" + firstname +
                   "' and lastname = '" + lastname +
                   "' and email = '" + email + "'")

    userid = cursor.fetchone()

    if userid is not None:
        print("user already exists\n")
        return userid[0]

    cursor.execute("INSERT INTO user(firstname, lastname, email) VALUES (?,?,?)",
                   (firstname, lastname, email))
    connection.commit()

    cursor.execute("Select userid from user "
                   " where firstname = '" + firstname +
                   "' and lastname = '" + lastname +
                   "' and email = '" + email + "'")

    userid = cursor.fetchone()

    cursor.close()
    connection.close()

    return userid[0]


# Insert facial recognition file into database
def insertfacedb(userid, file):
    # Establish connection to database
    connection = mariadb.connect(user=database.username, password=database.password, database=database.name,
                                 host=database.host)
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
    connection = mariadb.connect(user=database.username, password=database.password, database=database.name,
                                 host=database.host)
    cursor = connection.cursor()

    cursor.execute("Select datfile from dat "
                   "where userid = '" + str(datid) + "'")

    dat = cursor.fetchone()
    if dat == None:
        return

    cursor.close()
    connection.close()

    return dat[0]


def returnallfaces():
    # Establish connection to database
    connection = mariadb.connect(user=database.username, password=database.password, database=database.name,
                                 host=database.host)
    cursor = connection.cursor()

    cursor.execute("Select userid, datfile from dat")

    data = cursor.fetchall()

    if data is None:
        print("No dat files")
        return None

    cursor.close()
    connection.close()

    return data


# Print user information
def printuser(userid):
    # Establish connection to database
    connection = mariadb.connect(user=database.username, password=database.password, database=database.name,
                                 host=database.host)
    cursor = connection.cursor()

    cursor.execute("SELECT userid, firstname, lastname, email FROM user "
                   "where userid = " + str(userid))

    # print data
    row = cursor.fetchone()
    if row is None:
        return
    print(row)
    print(*row, sep='\t')
    print(row[1])

    cursor.close()
    connection.close()

def returnuser(userid):
    # Establish connection to database
    connection = mariadb.connect(user=database.username, password=database.password, database=database.name,
                                 host=database.host)
    cursor = connection.cursor()

    cursor.execute("SELECT firstname, lastname, email FROM user "
                   "where userid = " + str(userid))

    # print data
    data = cursor.fetchone()

    cursor.close()
    connection.close()

    return data

def searchuser(userid):
    # Establish connection to database
    connection = mariadb.connect(user=database.username, password=database.password, database=database.name,
                                 host=database.host)
    cursor = connection.cursor()

    cursor.execute("SELECT userid, firstname, lastname, email FROM user "
                   "where userid = " + str(userid))

    # print data
    userdata = cursor.fetchone()

    cursor.close()
    connection.close()

    return userdata


def newscanhist(userid, temp, passed):
    x = datetime.datetime.now()

    # Establish connection to database
    connection = mariadb.connect(user=database.username, password=database.password, database=database.name,
                                 host=database.host)
    cursor = connection.cursor()

    cursor.execute("INSERT INTO history(userid, temp, date, time, passed) VALUES(?,?,STR_TO_DATE(?, '%m/%d/%y'),?,?)",
                   (userid, temp, x.strftime("%x"), x.strftime("%X"), passed))
    connection.commit()

    cursor.close()
    connection.close()
