from vectorMath import *

def testModel( xMatrix, yVec, weights ):
    numCorrect = 0
    numFalse = 0
    total = len( xMatrix )
    if len( xMatrix[0] ) != len( weights ):
        print( "Error testing model: weight vector and data size dont match" )
        return False
    for i in range( total ):
        x = xMatrix[i]
        y = yVec[i]
        p = scalarProduct( weights, x )
        if p >= 0 and y == 1:
            numCorrect += 1
        if p < 0 and y == 0:
            numCorrect += 1
    numFalse = total - numCorrect
    message = "Total Test Records: "+str(total)+"\n"
    message += "False Predictions: "+str(numFalse)+"\n"
    message += "Correct Predictions: "+str(numCorrect)+"\n"
    message += "Model Accuracy: "+str( round( ( numCorrect / total ) * 100, 2 ) )+"%"
    print( message )
    return True


