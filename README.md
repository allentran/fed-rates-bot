# fed-rates-bot
# Translate Federal Reserve Documents

Various experiments based around translating Fed documents to word embeddings, and translating sequences of word embeddings into paths of the federal funds rate.  I strip out dates and cardinal values so that this isn't a completely trivial exercise.

The models revolve around various forms of stacked LSTMs and piping the output to either a mixture density layer or the standard squared loss error.

The best result that I have is shown below (on test data).

![Results](https://github.com/allentran/fed-rates-bot/blob/master/prelim_results.png)
