from flask import Flask
from flask import render_template
from flask import request

app = Flask(__name__)

@app.route("/", methods=['GET'])
def main_page():
    return render_template('mainpage.html')

@app.route('/predict', methods=['POST'])
def predict():
    return "Got "+str(request.get_json())