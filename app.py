from flask import Flask, render_template, request
from prediction import setup, predictExample
from markupsafe import escape

#these three lines only work on unix
#import resource
#softlimit, hardlimit = resource.getrlimit(resource.RLIMIT_AS)
#resource.setrlimit(resource.RLIMIT_AS, (536870912, hardlimit))

#cardnames = open("cardnames.txt", "r").read()#empty verson
model, keys, cardnames, mulliganWinRates = setup("model_11_oct_2021_1", "keys.txt", "cardnames.txt", "mulliganWinRates.txt")
app = Flask(__name__)

@app.route("/", methods=['GET'])
def main_page():
    return render_template('mainpage.html', cardnames=cardnames)

@app.route('/predict', methods=['POST'])
def predict():
    req = request.get_json()
    example = [float(req["cards"]), float(req["onplay"])] + str(req["hand"]).replace(", ", " ").split(",")
    result = predictExample(example, model, keys, mulliganWinRates)
    print(result)
    return escape(result)


@app.route("/about", methods=['GET'])
def about_page():
    return render_template('about.html')