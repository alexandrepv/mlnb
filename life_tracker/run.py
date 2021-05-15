"""
The life tracker app is a simple framework I created to log every day of my life from now on in a simple way.
Every entry will be stored as a dictionary in a long list of entries,

Tips:
- You must put your html templates in a "templates" folder in order for Flask to find them
"""

from flask import Flask, render_template, abort
app = Flask(__name__)

@app.route('/')
def home():

    return render_template('base.html')

if __name__ == '__main__':
    app.run()
