from flask import Flask, render_template, abort, request
import json

"""
This is a fun little idea to try to create a quiz-system to store all the trivia that I know and keep training me over
time. There are two parts of this system:

Part A) Responsible for adding the questions and their respective answers
Part B) Creates a quiz, scores me and tracks my progress over time

The goal is to keep my at my toes when it comes to my general knowledge. Some features to add in time:
- An individual counter to see which questions I'm getting more right or wrong over time, to try to train me harder on 
those

"""

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/<key>', methods=['GET', 'POST'])
def add_question(key):

    data_json = json.dumps(request.args)

    return render_template(f'{key}')

if __name__ == '__main__':
    app.run()
