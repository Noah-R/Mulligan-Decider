from flask import Flask, render_template, request
from prediction import setup, predictExample

model, keys, cardnames, mulliganWinRates = setup("model_30_sep_2021_2", "keys.txt", "cardnames.txt", "mulliganWinRates.txt")
app = Flask(__name__)

@app.route("/", methods=['GET'])
def main_page():
    return render_template('mainpage.html', cardnames=cardnames)

@app.route('/predict', methods=['POST'])
def predict():
    req = request.get_json()
    example = [float(req["cards"]), float(req["onplay"])] + str(req["hand"]).replace(", ", " ").split(",")
    return predictExample(example, model, keys, mulliganWinRates)