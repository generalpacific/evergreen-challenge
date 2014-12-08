EvergreenChallenge
==================

https://www.kaggle.com/c/stumbleupon

### Running the code

`main.py` is the main python code that runs the evaluation. Use following arguments to run the code:
   * Raw text file
   * Train data file
   * Test data file
   * Classifier to use:
    * `0`: Logistic Regression
    * `1`: Naive Bayes
    * `2`: Random Forest
   * Type of Preprocessing
    * `0`: Use TFIDF on boiler plate text
    * `1`: Use non-boiler plate text attributes
    * `2`: Use non-boiler plate text attributes AND extract one top LDA topic per boiler plate attribute
    * `3`: Use LDA topic vectos for boiler  plate code
   * Debug (optional): to print debug statments 
