#!/usr/bin/python
import sys
import getopt
import shutil
import math

from trainModel import trainModel
from testModel import testModel
from parser import getDataMatrix

usage = 'Usage: python regression.py -t <input training data> -s <input test data> [OPTIONS]\n\
Options:\n\
    -i ITERATIONS, -iterations=ITERATIONS   The number of iterations to run while training the data (default 1000).\n\
    -l RATE -learning-rate=RATE             Learning rate for each training iteration (default 0.00001).\n\
    -r REGULARIZER -regularizer=REGULARIZER regularizer for the model (default 1).\n\
    -w PATH -save-weights=PATH              appends the weights to a csv file during every iteration of the training loop. Will create a new file if none exists\n'

DEFAULT_ITERATIONS = 1000
DEFAULT_RATE = 0.00001
DEFAULT_REGULARIZER = 1
#File descriptor to write weights into 
weightsFile = None;

#Prints a progress bar to the screen
def printProgressBar( currIteration, totalIterations ):
    #Get deminsions for the current terminal
    terminalDimensions = shutil.get_terminal_size( ( 80, 20 ) )
    #Get the percentage of iterations complete
    percentage = currIteration/totalIterations
    #Create bar string
    percentCols = math.ceil( ( terminalDimensions.columns - 8 ) * percentage )
    bar = '#' * percentCols + ' ' * ( ( terminalDimensions.columns - 8 ) - percentCols )
    sys.stdout.write('\r[{0}] {1}%'.format( bar, round( percentage * 100 ) ) )
    #If its the last iteration of the bar print a new line
    if currIteration == totalIterations:
        print()

#Callback that fires every of the training loop
def trainingCallback( currIteration, totalIterations, weights ):
    #Save weights to file
    if weightsFile != None:
        weightsFile.write( ','.join( [ str(x) for x in weights ] )+'\n' )
    #Only print new prgress bar when we are at t new percentage point
    if currIteration % math.ceil( totalIterations / 100 ) == 0:
        printProgressBar( currIteration, totalIterations )

#Callback that fires when the test is complete
def testCallback( numCorrect, totalTested ):
    numFalse = totalTested - numCorrect
    message = "Total Test Records: "+str(totalTested)+"\n"
    message += "False Predictions: "+str(numFalse)+"\n"
    message += "Correct Predictions: "+str(numCorrect)+"\n"
    message += "Model Accuracy: "+str( round( ( numCorrect / totalTested ) * 100, 2 ) )+"%"
    print( message )

def main( argv ):
    trainingDataPath = ''
    testDataPath = ''
    iterations = DEFAULT_ITERATIONS
    rate = DEFAULT_RATE
    regularization = DEFAULT_REGULARIZER
    weightsPath = ""
    global weightsFile

    # Make sure the correct arguments have been passed
    if len( argv ) < 3:
        print( usage )
        sys.exit( 2 )
    try:
        opts, args = getopt.getopt( argv, 't:s:i:r:l:w:h', ["training-data=","test-data=","iterations=","learning-rate=","regularizer=","save-weights="] )
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
        elif opt in ( '-w', "save-weights=" ):
            weightsPath = str( arg )

    #Make sure we got a test and training data path
    if trainingDataPath == '' or testDataPath == '':
        print( usage )
        sys.exit( 2 )

    if weightsPath != "":
        weightsFile = open( weightsPath, 'a' )

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
    weights = trainModel( trainingDataMatrix, trainingResponseVec, rate, regularization, iterations, trainingCallback )
    #Close weights file if it is open
    if weightsFile != None:
        weightsFile.close()
    print( weights )
    print( "Testing model..." )
    testModel( testDataMatrix, testResponseVec, weights, testCallback )

if __name__ == "__main__":
    main( sys.argv[1:] )
