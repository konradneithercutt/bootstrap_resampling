# test_bootstrap.py

import pytest
import random
from BootstrapResampling import BootstrapResampling


# Sample data (smaller set)
scores1 = []
scores1.append( {'question':"Question Text", 'answer':"Answer Text", 'baselineScore':0.0, 'experimentalScore':1.0} )
scores1.append( {'question':"Question Text", 'answer':"Answer Text", 'baselineScore':1.0, 'experimentalScore':0.0} )
scores1.append( {'question':"Question Text", 'answer':"Answer Text", 'baselineScore':0.0, 'experimentalScore':0.0} )
scores1.append( {'question':"Question Text", 'answer':"Answer Text", 'baselineScore':1.0, 'experimentalScore':1.0} )
scores1.append( {'question':"Question Text", 'answer':"Answer Text", 'baselineScore':1.0, 'experimentalScore':1.0} )
scores1.append( {'question':"Question Text", 'answer':"Answer Text", 'baselineScore':0.0, 'experimentalScore':1.0} )
scores1.append( {'question':"Question Text", 'answer':"Answer Text", 'baselineScore':1.0, 'experimentalScore':0.0} )
scores1.append( {'question':"Question Text", 'answer':"Answer Text", 'baselineScore':1.0, 'experimentalScore':1.0} )
scores1.append( {'question':"Question Text", 'answer':"Answer Text", 'baselineScore':0.0, 'experimentalScore':1.0} )

# Sample data (larger set)
scores2 = []
scores2.append( {'question':"Question Text", 'answer':"Answer Text", 'baselineScore':0.0, 'experimentalScore':1.0} )
scores2.append( {'question':"Question Text", 'answer':"Answer Text", 'baselineScore':1.0, 'experimentalScore':1.0} )
scores2.append( {'question':"Question Text", 'answer':"Answer Text", 'baselineScore':0.0, 'experimentalScore':0.0} )
scores2.append( {'question':"Question Text", 'answer':"Answer Text", 'baselineScore':1.0, 'experimentalScore':1.0} )
scores2.append( {'question':"Question Text", 'answer':"Answer Text", 'baselineScore':1.0, 'experimentalScore':1.0} )
scores2.append( {'question':"Question Text", 'answer':"Answer Text", 'baselineScore':0.0, 'experimentalScore':1.0} )
scores2.append( {'question':"Question Text", 'answer':"Answer Text", 'baselineScore':1.0, 'experimentalScore':1.0} )
scores2.append( {'question':"Question Text", 'answer':"Answer Text", 'baselineScore':1.0, 'experimentalScore':1.0} )
scores2.append( {'question':"Question Text", 'answer':"Answer Text", 'baselineScore':0.0, 'experimentalScore':1.0} )
scores2.append( {'question':"Question Text", 'answer':"Answer Text", 'baselineScore':1.0, 'experimentalScore':1.0} )
scores2.append( {'question':"Question Text", 'answer':"Answer Text", 'baselineScore':0.0, 'experimentalScore':1.0} )
scores2.append( {'question':"Question Text", 'answer':"Answer Text", 'baselineScore':0.0, 'experimentalScore':0.0} )
scores2.append( {'question':"Question Text", 'answer':"Answer Text", 'baselineScore':1.0, 'experimentalScore':1.0} )
scores2.append( {'question':"Question Text", 'answer':"Answer Text", 'baselineScore':1.0, 'experimentalScore':1.0} )
scores2.append( {'question':"Question Text", 'answer':"Answer Text", 'baselineScore':0.0, 'experimentalScore':0.0} )
scores2.append( {'question':"Question Text", 'answer':"Answer Text", 'baselineScore':1.0, 'experimentalScore':1.0} )
scores2.append( {'question':"Question Text", 'answer':"Answer Text", 'baselineScore':1.0, 'experimentalScore':0.0} )
scores2.append( {'question':"Question Text", 'answer':"Answer Text", 'baselineScore':0.0, 'experimentalScore':0.0} )
scores2.append( {'question':"Question Text", 'answer':"Answer Text", 'baselineScore':1.0, 'experimentalScore':1.0} )
scores2.append( {'question':"Question Text", 'answer':"Answer Text", 'baselineScore':0.0, 'experimentalScore':1.0} )
scores2.append( {'question':"Question Text", 'answer':"Answer Text", 'baselineScore':0.0, 'experimentalScore':0.0} )
scores2.append( {'question':"Question Text", 'answer':"Answer Text", 'baselineScore':0.0, 'experimentalScore':1.0} )
scores2.append( {'question':"Question Text", 'answer':"Answer Text", 'baselineScore':1.0, 'experimentalScore':1.0} )
scores2.append( {'question':"Question Text", 'answer':"Answer Text", 'baselineScore':1.0, 'experimentalScore':1.0} )
scores2.append( {'question':"Question Text", 'answer':"Answer Text", 'baselineScore':0.0, 'experimentalScore':0.0} )



# Tests whether averages are computed correctly
def test_averages():
    bsr = BootstrapResampling()

    assert( bsr.getAverageBaselineScore(scores2) == 0.52 )    
    assert( bsr.getAverageExperimentalScore(scores2) == 0.72 )        

    assert( bsr.getAverageBaselineScore(scores1) == pytest.approx(0.5555, abs=0.01) )  
    assert( bsr.getAverageExperimentalScore(scores1) == pytest.approx(0.6666, abs=0.01) )        


# Tests whether difference scores are computed correctly
def test_difference_scores():
    bsr = BootstrapResampling()

    assert (bsr.createDifferenceScores(scores1) == [1.0, -1.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 1.0])
    assert (bsr.createDifferenceScores(scores2) == [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0])

# Note: createDifferenceScores() is not directly tested as it is stochastic and there are many possible correct answers -- it's indirectly tested by testing calculatePValue().

# Tests whether the full bootstrap resampling procedure gives the expected p-value.
# Note that this is stochastic (given the random samples), so 100k resamples are used to make the value very stable (typically <0.01 variance, but 0.02 used for a cushion)
def test_bootstrap():
    bsr = BootstrapResampling()
    assert( bsr.calculatePValue(scores1, numResamples=100000) == pytest.approx(0.41, abs=0.02) )
    assert( bsr.calculatePValue(scores2, numResamples=100000) == pytest.approx(0.03, abs=0.02) )
