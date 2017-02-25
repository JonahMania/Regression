import math
from vectorMath import *

def probability( x ):
    return 1 / ( 1 + math.exp( x ) )

def modelError( x, y, weights ):
    m = probability( scalarProduct( scaleVector( -1, weights ), x ) )
    return m - y

#Runs linear regression on a N by D matrix
def trainModel( xMatrix, yVec, learningRate, regularization, iterations ):
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
        #Normalize gradient
        g = scaleVector( 1 / N, g )
        #Adjust g for regularization with weights
        g = addVectors( g, scaleVector( regularization, weights ) )
        #Scale weights for learning rate
        weights = addVectors( weights, scaleVector( -learningRate, g ) )
        #Print status update for user every 20 cycles
        if not(itr % 20 ):
            print( str( round( (itr/iterations)*100 ) ) +'%' )
    return weights


