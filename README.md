# Mulligan-Decider

Current Model
	Player win rate
	Player rank
	Opponent rank
	Play/draw
	Player mulligans
	Number of each card in opening hand
	Number of each card in deck
	Number of each card in sideboard
	Rank differential between players

To Crunch(Separately for in hand and in deck)
	Names of columns to import, rather than importing all and dropping
	Number of lands
	Number of one through eight drop creatures and noncreatures
	Number of spells of each color
	Number of double/triple/quad symbol spells of each color
	Number of sources of each color
	Has turn one through turn four play, with mana to cast

Models
	Chance of winning if keep
		Logistic regression with all features
	Chance of winning if mulligan
		Logistic regression without cards in opening hand features, possibly trained on only examples with an additional mulligan or more

Room for Expansion
	Taplands/colors
	Interaction terms between cards
	Opponent mulligans

Investigate
	How does it denote draws?

Data and license are available at https://www.17lands.com/public_datasets