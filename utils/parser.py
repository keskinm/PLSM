import numpy as np

def parse_tdoc_file(filetdoc, document_length, nw):
    matrix = [[0 for x in range(document_length)] for x in range(nw)]
    # np.zeros((documentLength, W))  # Initialize Matrix of length 300x25 with Zeros   ,dtype=np.uint8

    Dict = {}  # Define Dictionary , Index of line position : List (that contain the Values)
    # Read the file tdoc
    infile = open(filetdoc, 'r')  # Read the tdoc file
    i = 0
    for data in infile:
        if len(data.strip()) != 0:  # take not empty line in tdoc and make some processing on Data
            Dict[i] = [data]  # Example Dict[113] = ['12:2 13:3 14:4 15:2 16:1 \n']
        i += 1
    infile.close()
    for j in Dict.keys():
        x = Dict[j]  # x is the Value in the Dict at Index j
        y = x[0].split(' ')  # Split at the space to get something like this ['17:1', '\n']
        for k in range(len(y)):
            z = y[k].split(':')  # split at : so we get ['17:1']['\n']
            for m in range(len(z) - 1):
                b = int(z[m])  # b is the position of Word W and j is the time , For Example b=17
                q = float(z[m + 1])  # q is the value should be filled in the matrix , For example q=1 (at position j,b)
                matrix[b][j] = q  # Fill the Matrix with v
    #                 matrix= np.transpose(matrix)         # To put the Matrix in the correct form

    return np.array(matrix)  # Return the Matrix