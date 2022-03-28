import pickle
from tensorflow.keras import preprocessing

def gloveEmbedText(texts, maxLen, tokenizerObjFile, padding='post'):
    '''
    Function to convert the texts into GloVe Embeddings by padding with the same length of the input maximum sequence length.

    Parameters:
    -----------
    texts: Series or Text
        Series or Text containing the comment text(s).
    maxLen: int
        Maximum sequence length to be used as the padding length.
    tokenizerObjFile: str
        File path containing the tokenizer object.
    padding: str
        Kind of padding to do. Pre-padding or post padding.
    '''

    # Load the tokenizer object
    with open(tokenizerObjFile, 'rb') as f:
        
        tokenizer = pickle.load(f)

    # Do integer encoding of the input text(s).
    intEncodedTexts = tokenizer.texts_to_sequences(texts)
    
    # Pad the integer encoded comments texts (Post Padding) and return.
    return preprocessing.sequence.pad_sequences(intEncodedTexts, maxlen=maxLen, padding=padding)