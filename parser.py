import csv

#Gets data from a csv assuming that the first column is the response variable
def getDataMatrix( path, dataMatrix, responseVector ): 
    with open( path, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            dataMatrix.append( [ float(x) for x in row[1:] ] )
            responseVector.append( float(row[0]) )
    return True


