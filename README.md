# Mulligan-Decider

This is a machine learning model for Magic: the Gathering, which decides whether a player should keep or mulligan an opening hand in a limited game. The model predicts the probability of winning if the player keeps the hand using a neural network which trains on 17lands game data. It predicts the probability of winning if the player mulligans the hand by taking the simple win rate of all hands in the dataset with fewer cards. The player is advised to pick the option with the higher chance of winning.

Of note, the dataset only shows the cards in the final opening hand, after resolving all mulligans and putting cards on bottom of library. This leads to some bias. For instance, there are very few one-land hands in the data set, because players generally mulligan one-land hands. The only one-land hands in the dataset are hands that the player chose to keep, which means they likely have a disproportionately high number of early plays/cheap draw spells/mana fixing. Therefore, the model likely underestimates the extent to which only having one land hurts your chances of winning.

This also leads to a complexity of interpretation with regard to hands of less than seven cards. On a mulligan to six, for instance, the model predicts the probability of winning based on the six cards ultimately kept, but the player sees seven cards when deciding whether to mulligan to five, before putting one on bottom of library. This does not affect training or testing, but at prediction time, the prediction function predicts the probability of winning for each six-card combination of the seven cards, and the user is advised to choose the combination with the highest chance of winning.

This repository also includes a logistic regression model, which makes significantly worse predictions. It was implemented mostly as a quick and dirty way to get TensorFlow configured and working, but it also serves to establish a linear baseline for predictive accuracy to compare more complex models against.

Data and license are available at https://www.17lands.com/public_datasets
Some functions also use MTGJSON set files, which are available at https://mtgjson.com/downloads/all-sets/

## Near term roadmap

    Build out web app
        Decide the best format for presenting predictions to the user
        User-proof back end
        Write a 'How this works' section
        Improve front end

    Set up a website to host this on

    Improve machine learning stack
        Try a three-layer configuration
        Configure a model that can use all the data
        Improve algorithm efficiency of prediction function