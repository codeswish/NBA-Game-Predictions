# NBA-Game-Predictions
A NBA game prediction neural network.

Introduction

I'm a masters student at Carnegie Mellon University who recently took a class in deep learning. To solidify the concepts I learned, I turned to my passion of NBA analytics to build a Recurrent Neural Network that could predict the outcomes of NBA games.

I was surprised by the results! [It appears 538 has an accuracy of approximately 76%](https://blog.albertkuo.me/post/2022-01-21-how-good-is-fivethirtyeight-s-nba-prediction-model/) and my network currently performs just as well. I hope to improve on the model and methodology in the future.

Before I dive into it, I want to share my [Colab notebook](https://colab.research.google.com/drive/1VAG5EXLXq9To7wtZVOT3VoX3X2mvILya?usp=sharing) containing all my code. I used a combination of Pytorch, Numpy and Pandas as the basis of my codebase.

Data Aggregation and Preprocessing

I owe a huge thanks to [this dataset publicly available on Kaggle](https://www.kaggle.com/datasets/nathanlauga/nba-games). It contains NBA game and ranking data from 2014 onwards, which was the core of my dataset.

Logic

When thinking about how an NBA team performs in a given game, there are a couple factors that I believe contribute the most:

*   the team's current ranking
    
*   how the team has performed recently
    
*   the opponent's current ranking
    
*   how the opponent has performed recently
    
*   how the two teams fared against one another in the past
    

Maintaining historical data is performed using Long-Short Term Memory (LSTM) units in deep learning. For more information on LSTM's refer to [this link](https://machinelearningmastery.com/gentle-introduction-long-short-term-memory-networks-experts/).

Hence, to predict the a game on a given day N between teams X and Y, I collected data about:

*   X's last 10 games
    
*   Y's last 10 games
    
*   X's rank on day N
    
*   Y's rank on day N
    
*   Last 3 matchups between X and Y
    

Game statistics included the following features:

`[WIN, IS_HOME, PTS, FG_PCT, FT_PCT, FG3_PCT, AST, REB]`

Ranking statistics included the following features:

`[G, W_PCT, HOME_W_PCT, AWAY_W_PCT]`

(Where `G` corresponds to game #)

Matchup statistics included the following features:

`[WIN, IS_HOME, PTS_home, FG_PCT_home, FT_PCT_home, FG3_PCT_home, AST_home, REB_home, PTS_away, FG_PCT_away, FT_PCT_away, FG3_PCT_away, AST_away, REB_away]`

I also normalized these statistics between 0 and 1 to allow my model converge more quickly (if some values like points, assists, or rebounds are very different, the model could take longer).

Let's prepare a single training instance for the game between the Warriors and Celtics on December 10, 2022.

  

[![r/NBAanalytics - An NBA Game Prediction Neural Network with ~77% accuracy.](https://preview.redd.it/d1kdmswv2qca1.png?width=1116&format=png&auto=webp&v=enabled&s=d3e3a623baf044a2887307d5bdd283d90e6db406)](https://preview.redd.it/d1kdmswv2qca1.png?width=1116&format=png&auto=webp&v=enabled&s=d3e3a623baf044a2887307d5bdd283d90e6db406)

  

[![r/NBAanalytics - An NBA Game Prediction Neural Network with ~77% accuracy.](https://preview.redd.it/o4ycttpw2qca1.png?width=1096&format=png&auto=webp&v=enabled&s=be7afa8e3695ff686b145970d0e3769888d4c56f)](https://preview.redd.it/o4ycttpw2qca1.png?width=1096&format=png&auto=webp&v=enabled&s=be7afa8e3695ff686b145970d0e3769888d4c56f)

  

[![r/NBAanalytics - An NBA Game Prediction Neural Network with ~77% accuracy.](https://preview.redd.it/2oxwidhx2qca1.png?width=2602&format=png&auto=webp&v=enabled&s=20cad3ed59ffba954cbb9fdca0589e698206be55)](https://preview.redd.it/2oxwidhx2qca1.png?width=2602&format=png&auto=webp&v=enabled&s=20cad3ed59ffba954cbb9fdca0589e698206be55)

  

[![r/NBAanalytics - An NBA Game Prediction Neural Network with ~77% accuracy.](https://preview.redd.it/gaafgh7y2qca1.png?width=778&format=png&auto=webp&v=enabled&s=fb2ccfd0c61a5fa0e579674b68cf163b7fb4557f)](https://preview.redd.it/gaafgh7y2qca1.png?width=778&format=png&auto=webp&v=enabled&s=fb2ccfd0c61a5fa0e579674b68cf163b7fb4557f)

  

[![r/NBAanalytics - An NBA Game Prediction Neural Network with ~77% accuracy.](https://preview.redd.it/ybxem8xy2qca1.png?width=786&format=png&auto=webp&v=enabled&s=cb109bb78615f931ef354f1ebfbbd2c55bb3f13b)](https://preview.redd.it/ybxem8xy2qca1.png?width=786&format=png&auto=webp&v=enabled&s=cb109bb78615f931ef354f1ebfbbd2c55bb3f13b)

Finally, we know the outcome or label to be \`1\` (AKA the home team GSW won).

My training data consisted of 22873 examples and validation data consisted of 2542 examples

  

Model Design and Hyperparameters

My model consisted of 3 LSTM's to maintain the history of each team AND to maintain the history of the matchups between the 2 teams.

I combined the outputs of these LSTMs with a simple linear layer.

Next, we also will make a sequential linear layer to handle the rankings. These linear layers had a cylindrical architecture. Within the linear layers I used the GELU activation function as I found it outperformed ReLU and TanH. I am yet to experiment with dropout within these layers to improve validation accuracy, but found my model was never overfitting.

Finally, we will combine everything through one fully connected layer.

The output of the `fc` layer will go into a sigmoid function since our output is binary.

Hyperparameters

Loss function - since this was a binary classification problem I used `BCELoss`

Optimizer - I used `Adam`, but am yet to experiment much with alternatives (e.g. `AdamW` or `SGD`)

Schedulers - I'm yet to, but plan on adding schedulers to modify my static learning rate of 0.01

Regularization - I'm yet to encounter overfitting issues, so have not added any regularization techniques.

  

Results

I trained my model for 100 epochs and achieved a training accuracy of 76% and validation accuracy of 78%. I plan on improving my model in the future and am open to suggestions on how to do so.

Conclusions

Deep learning methods have not been applied to NBA game prediction very much per my research, but they are a very valid way of predicting game outcomes. Particularly, using a RNN to combine the features of previous home games, previous away games, and the matchups between the two can be very useful in order to beyond simple rankings.

  

Thanks for reading my post and please feel free to comment with any questions or suggestions!

And again here is a link to my [Colab notebook](https://colab.research.google.com/drive/1VAG5EXLXq9To7wtZVOT3VoX3X2mvILya?usp=sharing) containing all my code.


Source: [An NBA Game Prediction Neural Network with ~77% accuracy. by u/vagartha](https://www.reddit.com/r/NBAanalytics/comments/10ewy7l/an_nba_game_prediction_neural_network_with_77/)
