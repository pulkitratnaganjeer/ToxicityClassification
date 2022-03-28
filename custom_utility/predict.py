import pandas as pd
from custom_utility import preprocess_text
from custom_utility import tokenize

def predictToxicity(tokenizerFile, seqLength, model, testFile='', text=None):
    '''
    Function to predict the toxicity scores of the given text or the comment texts in the given Test Dataset.

    Parameter:
    ---------
    tokenizerFile: str
        File path of the tokenizer object.
    seqLength: int
        Maximum sequence length of the comment texts.
    model: tensorflow.keras.Model
        Model to used for prediction of the toxicity score.
    testFile: str
        File path of the Test Dataset.
    text: str
        Comment Text for which the toxicity score has to be calculated.
    '''

    if (text == None and testFile != ''):

        # Read the Test Dataset given in the Kaggle Problem
        test = pd.read_csv(testFile)

    else:

        test = pd.DataFrame({'id': [1], 'comment_text': [text]})

    # Preprocess the comment text and store the processed text in a new feature 'preprocessed_text'
    lstComments = test['comment_text'] # List of all comment texts
    lstProcessedComments = list() # List to store the preprocess comment texts

    # Preprocess each comment and store in the list 'lstProcessedComments'
    for comment in lstComments:

        lstProcessedComments.append(preprocess_text.preprocess(comment))
        
    # Create a new Feature 'preprocessed_text' for the new preprocessed comments.
    test['preprocessed_text'] = lstProcessedComments

    # Tokenize the preprocess comments texts (Post Padding)
    gloveCommentTest = tokenize.gloveEmbedText(texts=test['preprocessed_text'], maxLen = seqLength, tokenizerObjFile=tokenizerFile)

    # Predict the probabilities of the class label of the Test Dataset and store it in a new feature 'yPredProb' of the Test Dataset.
    test['prediction'] = model.predict(gloveCommentTest)[0].flatten()

    # Return the dataframe required in the format of submission file
    return test[['id', 'prediction']]