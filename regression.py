#!/usr/bin/python
import sys
import getopt
import csv

from trainModel import trainModel
from testModel import testModel

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
    weights = trainModel( trainingDataMatrix, trainingResponseVec, rate, regularization, iterations )
    print( weights )
    print( "Testing model..." )
    testModel( testDataMatrix, testResponseVec, weights )

if __name__ == "__main__":
    main( sys.argv[1:] )
