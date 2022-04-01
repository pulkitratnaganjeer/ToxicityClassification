import flask
import pickle
from tensorflow.keras import models
from custom_utility import predict as predictModule

app = flask.Flask(__name__)

# File location having the tokenizer object trained on the training data.
tokenizerObjFile = 'Resources/tokenizer.pkl'

# File location having the pre-trained model.
modelFile = 'BestModels/modelBiLSTM.h5'

# Reference: https://github.com/nidhibansal1902/Jigsaw-Unintended-Bias-in-Toxicity-Classification/blob/master/Jigsaw-LSTM%20with%20Glove%20Embedding%20New.ipynb
def customLoss(yActual, yPred):
    '''
    Function to calculate loss for the toxic class label.
    
    Parameters:
    -----------
    yActual: array-like
        Actual Class Labels.
    yPred: array-like
        Predicted Class Labels.
    '''
    
    return binary_crossentropy(backend.reshape(yActual[:, 0], (-1, 1)), yPred) * yActual[:, 1]

def loadPretrainedObj():
    '''
    Function to load the pre-trained objects like tokenizer and model.
    '''
    # region: Load Tokenizer

    # Load the tokenizer object
    with open(tokenizerObjFile, 'rb') as f:
        
        tokenizer = pickle.load(f)

    # endregion: Load Tokenizer

    # region: Load Pre-trained Model

    # Load the tokenizer object
    model = models.load_model(modelFile, custom_objects={'customLoss': customLoss})

    # endregion: Load Pre-trained Model

# Load the tokenizer object
with open(tokenizerObjFile, 'rb') as f:
    
    tokenizer = pickle.load(f)

# endregion: Load Tokenizer

# region: Load Pre-trained Model

# Load the tokenizer object
model = models.load_model(modelFile, custom_objects={'customLoss': customLoss})

# Call the function 'loadPretrainedObj()' to load the tokenizer and pre-trained model.
loadPretrainedObj()

@app.route('/')
@app.route('/index', methods=['GET'])
def index():

    return flask.render_template('index.html')

@app.route('/home', methods=['GET'])
def routeToHomepage():

    return flask.render_template('index.html')

@app.errorhandler(404)
def fallback(e):

    return flask.render_template('error.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():

    if flask.request.method == 'POST':

        # Fetch the input text.
        comment = flask.request.form['comment']

        # Call the function 'function1()' from the 'predict' module of the package 'custom_utility' to predict the toxicity.
        toxicityResult = predictModule.function1(comment, tokenizerObj=tokenizer, model=model)

        toxicityResult = {
            'isToxic': toxicityResult.iloc[0]['Toxic'],
            'toxicityScore': toxicityResult.iloc[0]['Probability']
        }

    return flask.render_template('index.html', toxicity=toxicityResult)


    



    

if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=8080, debug=True) # TODO: Remove debug later before deployment.
    app.run(host='0.0.0.0', port=8080)