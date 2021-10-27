# Mulligan-Decider

This is a machine learning model for Magic: the Gathering, which decides whether a player should keep or mulligan an opening hand in a limited game. The model predicts the probability of winning if the player keeps the hand, using a neural network which trains on 17lands game data. It compares this probability to the average win rate after mulliganing, and advises the user to pick the option with the higher chance of winning.

The model also advises, in the case of mulligans where more cards are submitted than can be kept, which card should be kept and which should be put on the bottom of the library.

This repository also includes a logistic regression model, which makes significantly worse predictions. It was implemented mostly as a quick and dirty way to get TensorFlow configured and working, but it also serves to establish a linear baseline for predictive accuracy to compare more complex models against.

Data and license are available at https://www.17lands.com/public_datasets

Some functions also use MTGJSON set files, which are available at https://mtgjson.com/downloads/all-sets/

All model training/testing code is in the base directory. The repository also includes a two-part web app to serve predictions to the user, located in /webapp and /serving. It works, but I have been unable to deploy it, since TensorFlow Serving depends on gRPC, which depends on HTTP/2, which Heroku flatly does not support. To run the web app on localhost:

    webapp:
            Uncomment localhost url in `prediciton.py`
            `pip install -r requirements.txt`
            `flask run`

    serving:
            `docker pull tensorflow/serving`
            `docker run -p 8501:8501 --name tfs --mount type=bind,source=C:\Users\noahr\Desktop\Mulligan-Decider\serving\currentModel,target=/models/model -t tensorflow/serving`(Replace my absolute path to currentModel with yours)

## Room for expansion
    Create a neural network to predict mulligan win rates
        Train on examples of games where the player mulliganed, using only play/draw, hand size, and seven random cards from the deck_* columns
        Simpler alternative, train on proportion of each basic land in the deck
        More complex alternative, train on the entire deck
