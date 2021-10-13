from flask import Flask, render_template, request
from prediction import setup, predictExample
from markupsafe import escape

keys, cardnames, mulliganWinRates = setup("keys.txt", "cardnames.txt", "mulliganWinRates.txt")
app = Flask(__name__)

@app.route("/", methods=['GET'])
def main_page():
    return render_template('mainpage.html', cardnames=cardnames)

@app.route('/predict', methods=['POST'])
def predict():
    req = request.get_json()
    example = [float(req["cards"]), float(req["onplay"])] + str(req["hand"]).replace(", ", " ").split(",")
    result = predictExample(example, keys, mulliganWinRates)
    return escape(result)


@app.route("/about", methods=['GET'])
def about_page():
    return render_template('about.html')