import pandas as pd
from custom_utility import preprocess_text
from custom_utility import tokenize
from tensorflow.keras import preprocessing

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

def function1(text, tokenizerObj, model, textFeature='comment_text', maxSeqLen=210, paddingType='post'):
    '''
    Function to implement the data pipeline for transforming the dataset into the required format as required by the Model
    and predict whether the given text is toxic or not, along with the toxicity score.
    
    Parameters:
    ----------
    text: str or Series or DataFrame containing the comment text(s)
        Comment Text(s) to be checked for toxicity
    tokenizerObj: Tokenizer
        Tokenizer object to be used for tokenizing the text(s).
    model: keras.engine.functional.Functional
        Pre-trained Model for doing the predictions.
    textFeature: str
        Name of the feature containing comment texts in case a DataFrame is passed as input.
    maxSeqLen: int
        Maximum sequence length.
    paddingType: str
        Type of padding to be done: post or pre.
    '''
    
    # region - Data Pre-processing -----------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------
    
    # Check the input text and convert to a Series if it is not, for further processing.
    if isinstance(text, str):
        
        rawText = pd.Series(text)
    
    elif isinstance(text, pd.core.frame.DataFrame):
        
        # Extract the comment text feature.
        rawText = text[textFeature]
        
    else:
        
        rawText = text
        
    # Pre-processing the comment text(s) and store it in a list
    preprocessedText = rawText.apply(preprocess_text.preprocess)
    
    # endregion - Data Pre-processing --------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------
    
    
    
    # region - Tokenization ------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------

    # Do integer encoding of the input text(s).
    intEncodedTexts = tokenizerObj.texts_to_sequences(preprocessedText)
    
    # Pad the integer encoded comments texts (Post Padding) and return.
    paddedText = preprocessing.sequence.pad_sequences(intEncodedTexts, maxlen=maxSeqLen, padding=paddingType)

    # endregion - Tokenization ---------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------
    
    
    
    # region - Prediction --------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------
    
    yPredProb = model.predict(paddedText)[0].flatten()
    yPredToxic = ['Yes' if prob >= 0.5 else 'No' for prob in yPredProb]
    
    # endregion - Prediction -----------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------


    
    # Create a DataFrame containing original text, Toxic Class and Toxicity Score (Probability)
    result = pd.DataFrame({'Comment Text': rawText, 'Toxic': yPredToxic, 'Probability': yPredProb})
    
    return result