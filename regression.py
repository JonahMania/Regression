#!/usr/bin/python
import sys
import getopt
import csv
import math
#import matplotlib.pyplot as plt 

usage = 'Usage: python regression.py -i <input file>'
dataPoints = {
    49: (lambda x: float(x) / 17),
}
rIndex = 313
rFunction = (lambda x: 0 if int(x) == -2 else 1 )
# Parses a csv file and returns a matrix of its data as a 2d array ( N by D )
# Where N is the number of datapoints in the set and D is the number of dimensions
def getDataMatrix( csvPath, columns, responseIndex, responseFunction, responseVec, dataMatrix ):
    with open( csvPath ) as csvfile:
        reader = csv.reader( csvfile, delimiter=',' )
        for row in reader:
            dataMatrix.append( list( ( columns[i](row[i]) ) for i in columns ) )
            responseVec.append( responseFunction( row[responseIndex] ) );
    return True

"""
def vecAdd( vectorA, vectorB ):
    ret = list()
    if len( vectorA ) != len( vectorB ):
        return False
    for i in range( len( vectorA ) ):
        ret.append( vectorA[i] + vectorB[i] )
    return ret

def vecMult( vector, scalar ):
    ret = list()
    for i in vector:
        ret.append(i*scalar)
    return ret 

def transposeMatrix( matrix ):
    ret = list()
    for i in range( len( matrix[0] ) ):
        ret.append( list( 0 for i in range( len( matrix ) ) ) )
    
    for i in range( len(matrix) ):
        for j in range(len(matrix[0])):
            ret[j][i] = matrix[i][j]    
    return ret

def multiplyMatrix( matrixA, matrixB ):
    ret = list()
    for i in range( max( len( matrixA ), len( matrixB ) ) ):
        ret.append( list( 0 for i in range( max( len( matrixA[0] ), len( matrixB[0] ) ) ) ) ) 

    for i in range( len(matrixA) ):
        for j in range( len(matrixB[0]) ):
            for k in range( len(matrixB) ):
                ret[i][j] += matrixA[i][k] * matrixB[k][j] 
    return ret
"""
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

def logisticVal( x ):
    return 1 / ( 1 + math.exp( -x ) )

def modelError( xMatrix, yVec, weights ):
    g = list( 0 for i in range( len( weights ) ) )
    for i in range( len( yVec ) ):
        x = xMatrix[i]
        y = yVec[i]
        m = logisticVal( scalarProduct( weights, x ) )
        g = addVectors( g, scaleVector( m - y, x ) )
        print( scaleVector( m-y, x ) )
    return g 

# Runs linear regression on a N by D matrix
def regression( xMatrix, yResponse, learningRate ):
    # Create a list of weights the size of one row of the datamatrix
    weights = list( 0 for i in range( len(xMatrix[0] ) ) )
    g = modelError( xMatrix, yResponse, weights )   
  
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
    regression( dataMatrix, responseVec, 0.1 )

if __name__ == "__main__":
    main( sys.argv[1:] )

