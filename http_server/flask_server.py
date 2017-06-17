"""
MIT licence https://opensource.org/licenses/MIT

Copyright (c) 2017 Jan Zikes zikesjan@gmail.com
"""

from flask import Flask

app = Flask(__name__)

@app.route("/detest_faces")
def detect_faces():
    return "Hello World!"