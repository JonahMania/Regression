import math
from vectorMath import *

def sigmoid( x ):
    return 1 / ( 1 + math.exp( -x ) )

def modelError( x, y, weights ):
    try:
        m = sigmoid( scalarProduct( weights, x ) )
    except OverflowError:
        if scalarProduct( weights, x ) < 0:
            m = 0
        else:
            m = 1
    return m - y

#Runs linear regression on a N by D matrix
def trainModel( xMatrix, yVec, learningRate, regularization, iterations, callback ):
    #Create a list of weights the size of one row of the datamatrix
    weights = list( 0 for i in range( len(xMatrix[0] ) ) )
    N = len( yVec )
    for itr in range( iterations ):
        g = list( 0 for i in range( len( weights ) ) )
        for i in range( len( yVec ) ):
            x = xMatrix[i]
            y = yVec[i]
            error = modelError( x, y, weights )
            g = addVectors( g, scaleVector( error, x ) )
        #Adjust g for regularization with weights
        g = addVectors( g, scaleVector( regularization, weights ) )
        #Scale weights for learning rate
        weights = addVectors( weights, scaleVector( -learningRate, g ) )
        #Run callback
        callback( itr+1, iterations, weights ) 
    return weights


