poorpassword = "password"
okaypassword = "Password"
goodpassword = "Password1"
strongpassword = "Password1!"

password = strongpassword

import re
pwminlength = 6
pwscore = 0
if len(password) < pwminlength:
    print("Password is too short!")
if re.search("[\d]",password):
    pwscore = pwscore + 1
if re.search("[A-Z]",password):
    pwscore = pwscore + 1
if re.search("[^a-zA-Z\d]",password):
    pwscore = pwscore + 1
print(pwscore)