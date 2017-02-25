#!/usr/bin/python
import sys
import getopt
import csv
import math

from vectorMath import *

usage = 'Usage: python regression.py -t <input training data> -s <input test data> [OPTIONS]\n\
Options:\n\
    -i ITERATIONS, -iterations=ITERATIONS   The number of iterations to run while training the data (default 1000).\n\
    -l RATE -learning-rate=RATE             Learning rate for each training iteration (default 0.1).\n\
    -r REGULARIZER -regularizer=REGULARIZER regularizer for the model (default 1).\n'

DEFAULT_ITERATIONS = 1000
DEFAULT_RATE = 0.1
DEFAULT_REGULARIZER = 1

#Gets data from a csv assuming that the first column is the response variable
def getDataMatrix( path, dataMatrix, responseVector ): 
    with open( path, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            dataMatrix.append( [ float(x) for x in row[1:] ] )
            responseVector.append( float(row[0]) )
    return True

def probability( x ):
    return 1 / ( 1 + math.exp( x ) )

def modelError( x, y, weights ):
    m = probability( scalarProduct( scaleVector( -1, weights ), x ) )
    return m - y

#Runs linear regression on a N by D matrix
def regression( xMatrix, yVec, learningRate, regularization, iterations ):
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

def main( argv ):
    trainingDataPath = ''
    testDataPath = ''
    iterations = DEFAULT_ITERATIONS
    rate = DEFAULT_RATE
    regularization = DEFAULT_REGULARIZER

    # Make sure the correct arguments have been passed
    if len( argv ) < 3:
        print( usage )
        sys.exit( 2 )
    try:
        opts, args = getopt.getopt( argv, 't:s:i:r:l:h', ["training-data=","test-data=","iterations=","learning-rate=","regularizer="] )
    except getopt.GetoptError:
        print( usage )
        sys.exit( 2 )

    for opt, arg in opts:
        if opt == '-h':
            print( usage )
            sys.exit()
        elif opt in ( '-t', "training-data=" ):
            trainingDataPath = arg
        elif opt in ( '-s', "test-data=" ):
            testDataPath = arg
        elif opt in ( '-i', "iterations=" ):
            iterations = int(arg)
        elif opt in ( '-l', "rate=" ):
            rate = float(arg)
        elif opt in ( '-r', "regularizer=" ):
            regularization = float(arg)

    #Make sure we got a test and training data path
    if trainingDataPath == '' or testDataPath == '':
        print( usage )
        sys.exit( 2 )

    #Get our test and training data
    trainingDataMatrix = list();
    trainingResponseVec = list();
    testDataMatrix = list();
    testResponseVec = list();
    print( "Parsing training data..." )
    getDataMatrix( trainingDataPath, trainingDataMatrix, trainingResponseVec );
    print( "Parsing test data..." )
    getDataMatrix( testDataPath, testDataMatrix, testResponseVec );
    #Make sure all our points are normalized
    for i in trainingDataMatrix:
        for j in range( len( i ) ):
            if i[j] < 0 or i[j] > 1:
                print( "Training data point out of range [0,1] at index:",j,"with value:",i[j] )
                sys.exit(2)
    for i in testDataMatrix:
        for j in range( len( i ) ):
            if i[j] < 0 or i[j] > 1:
                print( "Test data point out of range [0,1] at index:",j,"with value:",i[j] )
                sys.exit(2)
    print( "Running regression..." )
    weights = regression( trainingDataMatrix, trainingResponseVec, rate, regularization, iterations )
    print( weights )
    print( "Testing model..." )
    testModel( testDataMatrix, testResponseVec, weights )

if __name__ == "__main__":
    main( sys.argv[1:] )
