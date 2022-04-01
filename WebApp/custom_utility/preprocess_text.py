import re
import unidecode
from bs4 import BeautifulSoup
from custom_utility.emoticon_dictionary import emoticonsDict
from custom_utility.contraction_dictionary import contractionMap

def removeHTMLTags(text):
    '''
    Function to remove the HTML Tags from a given text.
    
    Parameter:
    ---------
    text: str
        Text from which the HTML tags has to be removed.
    '''
    
    # Reference: 'Remove html tags using BeautifulSoup' - https://www.geeksforgeeks.org/remove-all-style-scripts-and-html-tags-using-beautifulsoup/
    
    # Create a BeautifulSoup object to parse the given html text content
    soup = BeautifulSoup(text, 'html.parser')
    
    # Remove the <style> and <script> tags from the html content because they contains the styling sheet and javascript
    # file references and won't give any meaningful context.
    for data in soup(['style', 'script']):
        
        # Remove tag
        data.decompose()
        
    # Return the html tag free content
    return ' '.join(soup.stripped_strings)


def removeAccentedChars(text):
    '''
    Function to remove the accented characters from a given text.
    
    Parameter:
    ---------
    text: str
        Text from which the accented character has to be removed.
    '''
    
    # Reference: "remove accented characters python" - https://www.geeksforgeeks.org/how-to-remove-string-accents-using-python-3/
    
    # Remove accents
    return unidecode.unidecode(text)


def lowercase(text):
    '''
    Function to convert a given text to its lowercase.
    
    Parameter:
    ---------
    text: str
        Text that has to be converted to lowercase.
    '''
    
    return text.lower()


def removeIPLinkNum(text, ipAddress=True, hyperlink=False, numbers=True):
    '''
    Function to remove IP Address and Number from the given text.
    
    Parameter:
    ---------
    text: str
        Text from which IP Address and number(s) have to be removed.
    '''
    
    # Replace IP Address with empty string.
    # Reference: 'Remove IP Address Python' - https://www.geeksforgeeks.org/extract-ip-address-from-file-using-python/#:~:text=The%20regular%20expression%20for%20valid,%5C.)%7B
    if ipAddress == True:
        
        text = re.sub(r'((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)', '', text)
    
    # Remove hyperlinks
    # Reference: 'Regex for hperlinks Python' - https://www.geeksforgeeks.org/python-check-url-string/
    if hyperlink == True:
        
        text = re.sub(r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))", "", text)
    
    # Remove numbers.
    if numbers == True:
        
        text = re.sub(r'[0-9]', '', text)
    
    # Remove the extra space if any.
    text = re.sub(r'[ ][ ]+', ' ', text)
    
    return text


# Replace Emoticons with correponding words
def replaceEmoticons(text):
    
    for emoticon in emoticonsDict:
        
        word = "_".join(emoticonsDict[emoticon].split())
        
        text = text.replace(emoticon, ' ' + word + ' ')
        
        # Remove the extra space if any.
        text = re.sub(r'[ ][ ]+', ' ', text)
    
    return text


def removeSpecialChars(text, removeAll=False):
    '''
    Function to remove the special characters from the given text.
    
    Parameter:
    ---------
    text: str
        Text from which the special characters have to be removed.
    removeAll: boolean
        Flag to check whether to remove all special characters or all except ' . ? !
    '''
    
    if removeAll == True:
        
        text = re.sub(r'[^A-Za-z ]+', '', text) # Remove all special characters.
        
    else:
        
        text = re.sub(r'[^A-Za-z\'.?! ]+', '', text) # Remove all special characters except ' . ? !
        
    # Remove the extra space if any.
    text = re.sub(r'[ ][ ]+', ' ', text)
    
    return text


def processSpecialTokens(text, isBERTUsed=False):
    '''
    Function to add one space around sentence end markers and remove duplicates.
    
    Parameter:
    ---------
    text: str
        Text in which space has to be added around sentence end tokens.
    isBERTUsed: boolean
        Boolean flag to indicate if BERT is used in the modelling, then do not apply this pre-processing.
    '''
    
    if (isBERTUsed == False):
    
        text = re.sub(r'[!]+[ ]*[!]*', ' ! ', text) # Add space around ! with exclmrk.
        text = re.sub(r'[?]+[ ]*[?]*', ' ? ', text) # Replace ? with qstmrk.
        text = re.sub(r'[.]+[ ]*[.]*', ' . ', text) # Replace . with eosmkr.

        # Remove the extra space if any.
        text = re.sub(r'[ ][ ]+', ' ', text)
    
    return text


def decontract(text, isBERTUsed=False):
    '''
    Function to decontract a given text.
    
    Parameter:
    ---------
    text: str
        Text to be decontracted.
    isBERTUsed: boolean
        Boolean flag to indicate if BERT is used in the modelling, then do not apply this pre-processing.
    '''
    
    if isBERTUsed==False:
    
        # Iterate through all the contraction keys and replace the keys with their corresponding values (expanded form)
        for word in contractionMap.keys():

            text = lowercase(text) # Convert to lowercase.
            text = re.sub(word, contractionMap[word], text) # Replace the contracted word with its decontracted form.
        
    return text


def preprocess(text, html=True, accent=True, lower=True, ipLinkNum=True, emoticon=True, specialChar=True, 
               specialToken=True, decontraction=True, isBERTUsed=False, removeAllSpecialChar=False, hyperlink=False):
    '''
    Function to perform all the data-preprocessing on a given text.
    
    Parameters:
    ----------
    text: str
        Text on which the pre-processing has to be performed.
    html: boolean
        Flag to check whether to remove html tags from the text or not.
    accent: boolean
        Flag to check whether to remove the accented characters from the text or not.
    lower: boolean
        Flag to check whether to perform lowercase on the text or not.
    ipLinkNum: boolean
        Flag to check whether to remove the IP Address, Hyperlink(s) and number(s) from the text or not.
    emoticon: boolean
        Flag to check whether to replace the emoticons with their corresponding words in the text or not.
    specialChar: boolean
        Flag to check whether to remove the special characters from the text or not.
    specialToken: boolean
        Flag to check whether to replace the special tokens with their corresponding words in the text or not.
    decontraction: boolean
        Flag to check whether to do decontraction in the given text or not.
    isBERTUsed: boolean
        Boolean flag to indicate if BERT is used in the modelling, then do not apply this pre-processing.
    removeAllSpecialChar: boolean
        Flag to check whether to remove all special characters or all except ' . ? !
    hyperlink: boolean
        Flag to check whether to remove the hyperlink from the text or not.
    '''
    
    if html == True:
        
        # Call the function 'removeHTMLTags()' to remove the html tags from the html content
        text = removeHTMLTags(text)
        
    if accent == True:
        
        # Call the function 'removeAccentedChars()' to remove the accented characters from the text.
        text = removeAccentedChars(text)
        
    if lower == True:
        
        # Call the function 'lowercase()' to convert the text to its lowercase.
        text = lowercase(text)
        
    if ipLinkNum == True: 
        
        # Call the 'removeIPLinkNum()' to remove the IP Address, Hyperlinks and numbers from the text.
        text = removeIPLinkNum(text, hyperlink=hyperlink)
        
    if emoticon == True:
        
        # Call the 'replaceEmoticons()' to replace emoticons by their corresponding words.
        text = replaceEmoticons(text)
        
    if specialChar == True:
        
        # Call the 'removeSpecialChars()' to remove the special characters from a text.
        text = removeSpecialChars(text, removeAllSpecialChar)
        
    if specialToken == True:
        
        # Call the 'processSpecialTokens' to add space around sentence end tokens.
        text = processSpecialTokens(text, isBERTUsed)
        
    if decontraction == True:
        
        # Call the 'decontract()' function to decontract a given text. 
        text = decontract(text, isBERTUsed)
        
    return text