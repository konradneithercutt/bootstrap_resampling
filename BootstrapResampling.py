# BootstrapResampling.py

import random

class BootstrapResampling():
    """ This class  implements the non-parametric bootstrap resampling procedure discussed in class.
    """

    def getAverageBaselineScore(self, dataIn:list):
        """Given a list of dictionaries (dataIn) with key
            'baselineScore' (float), calculate the average baselineScore
            Example: [ {'question':"Question Text", 'answer':"Answer Text",
            'baselineScore':0.0, 'experimentalScore':1.0}, ... ]

            :param dataIn: List of dictionaries with key 'baselineScore'
            :return: Average 'baselineScore' across all elements in list.
        """
        sum = 0
        for dict in dataIn:
            sum += dict['baselineScore']
        return sum/len(dataIn)

    def getAverageExperimentalScore(self, dataIn:list):
        """Given a list of dictionaries (dataIn) with key
            'experimentalScore' (float), calculate the average baselineScore
            Example: [ {'question':"Question Text", 'answer':"Answer Text",
            'experimentalScore':0.0, 'experimentalScore':1.0}, ... ]

            :param dataIn: List of dictionaries with key 'experimentalScore'
            :return: Average 'experimentalScore' across all elements in list.
        """
        sum = 0
        for dict in dataIn:
            sum += dict['experimentalScore']
        return sum/len(dataIn)

    def createDifferenceScores(self, dataIn:list):
        """Given a list of dictionaries (dataIn) with keys 'baselineScore'
            and 'experimentalScore', calculate their difference scores
            (experimentalScore - baselineScore).
            Example: [ {'question':"Question Text", 'answer':"Answer Text",
            'experimentalScore':0.0, 'experimentalScore':1.0}, ... ]
            Example output: [1.0, ...]

            :param dataIn: List of dictionaries with float keys 'baselineScore', 'experimentalScore'
            :return: List of floats representing difference scores (experimental - baseline)
        """
        differenceScores = []
        for dict in dataIn:
            differenceScores.append(dict['experimentalScore']-dict['baselineScore'])
        return differenceScores

    def generateOneResample(self, differenceScores:list):
        """Randomly resamples the difference scores, to make a bootstrapped resample
            Example input: [0, 1, 0, 0, 1, 0, 1, 1, 0]
            Example output: [1, 0, 1, 0, 0, 1, 0, 1, 1]

            :param differenceScores: A list of difference scores (floats).
            :return: A list of randomly resampled difference scores (floats),
                of the same length as the input, populated using random
                sampling with replacement.
        """
        dummy = differenceScores
        random.shuffle(dummy)
        return dummy

    def calculatePValue(self, dataIn:list, numResamples=10000):
        """Calculate the p-value of a dataset using the bootstrap resampling procedure.
            Example: [ {'question':"Question Text", 'answer':"Answer Text",
            'baselineScore':0.0, 'experimentalScore':1.0}, ... ]
            Example output: 0.01

            :param dataIn: List of dictionaries with float keys 'baselineScore', 'experimentalScore' populated
            :param numResamples: The number of bootstrap resamples to use (typically 10,000 or higher)
            :return: A value representing the p-value using the bootstrap resampling procedure (float)
        """
        count = 0
        for i in range(numResamples):
            newDS = self.generateOneResample(self.createDifferenceScores(dataIn))
            randomScores = []
            for i in range(len(dataIn)):
                randomScores.append(random.choice(newDS))
            if sum(randomScores) > 0:
                count += 1
        return 1-(count/numResamples)
