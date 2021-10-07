from flask import Flask
from flask import render_template
from flask import request
import prediction

model, keys = prediction.setup()
app = Flask(__name__)

@app.route("/", methods=['GET'])
def main_page():
    return render_template('mainpage.html')

@app.route('/predict', methods=['POST'])
def predict():
    req = request.get_json()
    example = [float(req["cards"]), float(req["onplay"])] + str(req["hand"]).replace(", ", " ").split(",")
    return prediction.predictExample(example, model, keys)