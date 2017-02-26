from vectorMath import *

def testModel( xMatrix, yVec, weights, callback ):
    numCorrect = 0
    total = len( xMatrix )
    if len( xMatrix[0] ) != len( weights ):
        print( "Error testing model: weight vector and data size dont match" )
        return False
    #Check each value of the dataset
    for i in range( total ):
        x = xMatrix[i]
        y = yVec[i]
        p = scalarProduct( weights, x )
        if p >= 0.5 and y == 1:
            numCorrect += 1
        if p < 0.5 and y == 0:
            numCorrect += 1
    #Run callback
    callback( numCorrect, total )
    return True
