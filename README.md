# Mulligan-Decider

##To do next

	Make separate files for training data/test data, reconfigure code to use

	Write out neural_training

## Current Logistic Regression Model

	Play/draw

	Number of cards in hand

	Whether the hand is a two, three, four, or five lander

	Number of one, two, three, and four mana spells

## Addable Features

	Number of each card in opening hand
	
	Number of creatures/noncreatures
	
	Number of spells of each color
	
	Number of double/triple/quad symbol spells of each color
	
	Number of sources of each color
	
	Has turn one through turn four play, with mana to cast

	Is missing colors for spells in hand

	Taplands/colors
	
	Interaction terms between cards
	
	Opponent mulligans

	User win rate

## Models
	
	Chance of winning if keep
	
		Logistic regression with all features
	
	Chance of winning if mulligan
	
		Logistic regression without cards in opening hand features, possibly trained on only examples with an additional mulligan or more
	
	Neural Network

## Investigate
	
	How does it denote draws?

Data and license are available at https://www.17lands.com/public_datasets