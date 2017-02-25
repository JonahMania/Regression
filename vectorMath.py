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
