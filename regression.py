#!/usr/bin/python
import sys
import getopt
import csv
import math
#import matplotlib.pyplot as plt 

usage = 'Usage: python regression.py -i <input file>'
dataPoints = {
    49: (lambda x: float(x) / 17),
    64: (lambda x: 0 if int(x) == 2 else 1 )
}
rIndex = 313
rFunction = (lambda x: 0 if int(x) == -2 else 1 )
# Parses a csv file and returns a matrix of its data as a 2d array ( N by D )
# Where N is the number of datapoints in the set and D is the number of dimensions
def getDataMatrix( csvPath, columns, responseIndex, responseFunction, responseVec, dataMatrix ):
    with open( csvPath ) as csvfile:
        reader = csv.reader( csvfile, delimiter=',' )
        for row in reader:
            mRow = list( ( columns[i](row[i]) ) for i in columns ) 
            mRow.append( 1 )
            dataMatrix.append( mRow )
            responseVec.append( responseFunction( row[responseIndex] ) );
    return True

def scaleVector( scalar, vec ):
    ret = list( 0 for i in range( len( vec ) ) )
    for i in range( len( ret ) ):
        ret[i] = vec[i] * scalar
    return ret

def addVectors( vecA, vecB ):
    ret = list( 0 for i in range( len( vecA ) ) )
    for i in range( len( vecA ) ):
        ret[i] = vecA[i] + vecB[i]
    return ret

def scalarProduct( vecA, vecB ):
    ret = 0
    for i in range( len( vecA ) ):
        ret += vecA[i] * vecB[i]
    return ret

def probability( x ):
    return 1 / ( 1 + math.exp( x ) )

def modelError( x, y, weights ):
    m = probability( scalarProduct( scaleVector( -1, weights ), x ) )
    return m - y

#Runs linear regression on a N by D matrix
def regression( xMatrix, yVec, learningRate, iterations ):
    #Create a list of weights the size of one row of the datamatrix
    weights = list( 0 for i in range( len(xMatrix[0] ) ) )
    
    for itr in range( iterations ):
        g = list( 0 for i in range( len( weights ) ) ) 
        for i in range( len( yVec ) ):
            x = xMatrix[i]
            y = yVec[i]
            error = modelError( x, y, weights )
            g = addVectors( g, scaleVector( 1/len(xMatrix), scaleVector( error, x ) ) )
        #g = addVectors( g, scaleVector( l, weights ) )   
        weights = addVectors( weights, scaleVector( -learningRate, g ) )
        print( weights )
    return weights
def main( argv ):
    csvFilePath = ''
  
    # Make sure the correct arguments have been passed 
    if len( argv ) < 2:
        print( usage )
        sys.exit( 2 )
    try:
        opts, args = getopt.getopt( argv, 'hi:', ["inputFile="] )
    except getopt.GetoptError:
        print( usage )
        sys.exit( 2 )
    for opt, arg in opts:
        if opt == '-h':
            print( usage )
            sys.exit() 
        elif opt in ( '-i', "inputFile=" ):
            csvFilePath = arg
    
    print( "Parsing csv file..." ) 
    # Parse csv file
    dataMatrix = list();
    responseVec = list();
    getDataMatrix( csvFilePath, dataPoints, rIndex, rFunction, responseVec, dataMatrix )
    print( "Running regression..." )
   
    for i in dataMatrix:
        for j in i:
            if j < 0 or j > 1:
                print( j )
                sys.exit()
   
    regression( dataMatrix, responseVec, 0.1, 1000 )

if __name__ == "__main__":
    main( sys.argv[1:] )

