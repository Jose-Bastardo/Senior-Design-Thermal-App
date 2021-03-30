user: admin
Endpoint: thermal-app.ckyrcuyndxij.us-east-2.rds.amazonaws.com
password: 2901567j

CREATE TABLE user(userid INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
               firstname VARCHAR(100), 
			   lastname VARCHAR(100),
			   email VARCHAR(100)
			   );
			   
CREATE TABLE dat(datid INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
userid int unsigned,
datfile blob,
foreign key (userid) references user(userid)
);

create table history(histid int unsigned auto_increment,
userid int unsigned,
temp smallint,
date date,
time time,
passed boolean,
primary key(histid),
foreign key(userid) references user(userid)
);