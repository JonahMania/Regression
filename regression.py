#!/usr/bin/python
import sys
import getopt
import csv

usage = 'Usage: python regression.py -i <input file>'

# Parses a csv file and returns a matrix of its data as a 2d array
def getDataMatrix( csvPath ):
    with open( csvPath ) as csvfile:
        reader = csv.reader( csvfile, delimiter=',' )
        for row in reader:
            print( row )

# Get command line arguments
if len( sys.argv ) < 1:
    print( usage )
    sys.exit()

def main( argv ):
    csvFilePath = ''
  
    # Make sure the correct arguments have been passed 
    if len( argv ) < 1:
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
            
    # Parse csv file
    getDataMatrix( csvFilePath )

if __name__ == "__main__":
    main( sys.argv[1:] )

