import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def computeAUC(yActual, yPredProb):
    '''
    Function to calcuate the Overall AUC
    
    Parameters:
    ----------
    yActual: array-like
        Ground truth (correct) class labels for 'n' samples on CV/Test Dataset.
    yPredProb: array-like
        Predicted probabilities, as returned by a model's predict_proba method on CV/Test Dataset.
    '''
    
    try:
        
        return roc_auc_score(yActual, yPredProb)

    except:
        
        return np.nan


def computeSubgroupAUC(data, subgroup, actualClassLabel, predClassLabel):
    '''
    Function to compute the AUC of the within-subgroup negative examples and the background positive examples.
    
    Parameters:
    ----------
    data: DataFrame
        Dataset containing the CV or Test dataset to be evaluated.
    subgroup: str
        Name of the Identity Subgroup.
    actualClassLabel: str
        Name of the Class Label. Note that the Class Label should have binary values i.e., 0 and 1 and not toxicity scores.
    predClassLabel: str
        Name of the Class Label containing the Predicted probabilities, as returned by a model's predict_proba method.
    '''
    
    # Create a subset of the subgroup having score greater than the threshold.
    subgroup = data[data[subgroup]]
    
    # Compute and return the AUC Score of the subgroup
    return computeAUC(subgroup[actualClassLabel], subgroup[predClassLabel])


def computeBPSNAUC(data, subgroup, actualClassLabel, predClassLabel):
    '''
    Function to compute the AUC of the within-subgroup negative examples and the background positive examples.
    
    Parameters:
    ----------
    data: DataFrame
        Dataset containing the CV or Test dataset to be evaluated.
    subgroup: str
        Name of the Identity Subgroup.
    actualClassLabel: str
        Name of the Class Label. Note that the Class Label should have binary values i.e., 0 and 1 and not toxicity scores.
    predClassLabel: str
        Name of the Class Label containing the Predicted probabilities, as returned by a model's predict_proba method.
    '''
    
    # Create a subset of Subgroup Negative from the given dataset.
    subgroupNegativeSet = data[data[subgroup] & ~data[actualClassLabel]]
    
    # Create a subset of Background Positive from the given dataset.
    backgroundPositiveSet = data[~data[subgroup] & data[actualClassLabel]]
    
    # Combine both the Subgroup Negative and Background Positive subsets.
    finalSet = subgroupNegativeSet.append(backgroundPositiveSet)
    
    # Compute and return the BPSN AUC Score.
    return computeAUC(finalSet[actualClassLabel], finalSet[predClassLabel])


def computeBNSPAUC(data, subgroup, actualClassLabel, predClassLabel):
    '''
    Function to compute the AUC of the within-subgroup positive examples and the background negative examples.
    
    Parameters:
    ----------
    data: DataFrame
        Dataset containing the CV or Test dataset to be evaluated.
    subgroup: str
        Name of the Identity Subgroup.
    actualClassLabel: str
        Name of the Class Label. Note that the Class Label should have binary values i.e., 0 and 1 and not toxicity scores.
    predClassLabel: str
        Name of the Class Label containing the Predicted probabilities, as returned by a model's predict_proba method.
    '''
    
    # Create a subset of Subgroup Positive from the given dataset.
    subgroupPositiveSet = data[data[subgroup] & data[actualClassLabel]]
    
    # Create a subset of Background Negative from the given dataset.
    backgroundNegativeSet = data[~data[subgroup] & ~data[actualClassLabel]]
    
    # Combine both the Subgroup Positive and Background Negative subsets.
    finalSet = subgroupPositiveSet.append(backgroundNegativeSet)
    
    # Compute and return the BNSP AUC Score.
    return computeAUC(finalSet[actualClassLabel], finalSet[predClassLabel])


def computeBiasMetricsForModel(data, subgroups, predClassLabel, actualClassLabel):
    '''
    Function to compute per-subgroup metrics for all subgroups and one model.
    
    Parameters:
    ----------
    data: DataFrame
        Dataset containing the CV or Test dataset to be evaluated.
    subgroup: list
        List of the names of Identity Subgroups.
    actualClassLabel: str
        Name of the Class Label. Note that the Class Label should have binary values i.e., 0 and 1 and not toxicity scores.
    predClassLabel: str
        Name of the Class Label containing the Predicted probabilities, as returned by a model's predict_proba method.
    '''
    
    metricScores = list() # List to store the performance metric scores for each identity subgroup.
    
    # Define few constants to be used for dictionary keys.
    subgroupName, subgroupSize, subgroupAUC, bpsnAUC, bnspAUC = 'Subgroup', 'Subgroup Size', 'Subgroup AUC', 'BPSN AUC', \
                                                            'BNSP AUC'
    
    # Iterate through each of the identity subgroup and compute different AUC scores defined by the functions above.
    for subgroup in subgroups:
        
        # Define a dictionary to store the performance metric scores for the current subgroup in the iteration.
        metricScore = {
            subgroupName: subgroup,
            subgroupSize: len(data[data[subgroup]])
        }
        
        # Subgroup AUC
        metricScore[subgroupAUC] = computeSubgroupAUC(data=data, subgroup=subgroup, actualClassLabel=actualClassLabel,
                                                        predClassLabel=predClassLabel)
        
        # BPSN AUC
        metricScore[bpsnAUC] = computeBPSNAUC(data=data, subgroup=subgroup, actualClassLabel=actualClassLabel,
                                                        predClassLabel=predClassLabel)
        
        # BNSP AUC
        metricScore[bnspAUC] = computeBNSPAUC(data=data, subgroup=subgroup, actualClassLabel=actualClassLabel,
                                                        predClassLabel=predClassLabel)
        
        # Append the metric scores for the subgroup to the list 'metricScores'
        metricScores.append(metricScore)
        
    # Return the DataFrame containing the final performance metric scores for all identity subgroup.
    return pd.DataFrame(metricScores).sort_values(subgroupAUC)


def computePowerMean(biasMetric, p=-5):
    '''
    Function to compute the generalized mean of Bias AUCs.
    
    Parameters:
    -----------
    biasMetric: Series
        Series containing the bias metric for all the identity subgroups.
    p: float
        Value to be used for the power on bias metrics.
    '''
    
    # Calculate the sum of the pth power of the bias metrics.
    total = sum(np.power(biasMetric, p))
    
    # Return the generalized mean of the Bias AUCs
    return np.power(total / len(biasMetric), 1/p)


def computeOverallAUC(data, actualClassLabel, predClassLabel):
    '''
    Function to compute the overall AUC.
    
    Parameters:
    ----------
    data: DataFrame
        Dataset containing the CV or Test dataset to be evaluated.
    actualClassLabel: str
        Name of the Class Label. Note that the Class Label should have binary values i.e., 0 and 1 and not toxicity scores.
    predClassLabel: str
        Name of the Class Label containing the Predicted probabilities, as returned by a model's predict_proba method.
    '''
    
    # Get the actual class labels.
    yActual = data[actualClassLabel]
    
    # Get the predicted class probabilities.
    yPredProb = data[predClassLabel]
    
    # Return the AUC Score.
    return computeAUC(yActual=yActual, yPredProb=yPredProb)


def computeFinalMetric(biasDF, overallAUC, p=-5, overallModelWeight=0.25):
    '''
    Function to compute the final metric score.
    
    Parameters:
    ----------
    biasDF: DataFrame
        Dataset containing the final performance metric scores for all identity subgroup.
    overallAUC: float    
        Overall AUC computed from the computeOverallAUC() function.
    p: float
        Value to be used for the power on bias metrics.
    overallModelWeight: float
        Weight value for the relative importance of each submetric; all four w values set to 0.25
    '''
    
    # Define few constants to be used for dictionary keys.
    subgroupAUC, bpsnAUC, bnspAUC = 'Subgroup AUC', 'BPSN AUC', 'BNSP AUC'
    
    # Get the average of the generalized mean of each metrics.
    biasScore = np.average([
        computePowerMean(biasDF[subgroupAUC], p),
        computePowerMean(biasDF[bpsnAUC], p),
        computePowerMean(biasDF[bnspAUC], p)
    ])
    
    # Return the final metric score.
    return (overallModelWeight * overallAUC) + ((1 - overallModelWeight) * biasScore)