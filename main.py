def main():
    matrix = [
        [2.0, 7.0, 1.0, 5.0, 70.0],
        [1.0, 5.0, 3.0, 2.0, 45.0],
        [3.0, 2.0, 4.0, 1.0, 33.0],
        [8.0, 1.0, 5.0, 3.0, 56.0]
    ]

    # starts as the identity matrix, but the inverse gets computed later
    inverse = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ]

    soln = GaussJordanWithInverse(matrix, inverse)
    print("Solution:")
    printMatrix(soln)
    print()
    print("Inverse:")
    printMatrix(inverse)

    # copy matrix without augument
    matrixCopy = [row[:] for row in matrix]
    identity = Winograd(matrixCopy, inverse)
    print()
    print("Identity:")
    printMatrix(identity)


def Winograd(A, B):
    a = len(A)
    b = len(B)
    c = len(B[0])
    d = int(b / 2)
    C = [[0 for i in range(c)] for j in range(a)]
    rowFactor = [0] * a
    colFactor = [0] * c

    # calculate rowfactors for A
    for i in range(a):
        rowFactor[i] = A[i][0] * A[i][1]
        for j in range(1, d):
            rowFactor[i] += A[i][2 * j] * A[i][2 * j + 1]

    # calculate columnfactors for B
    for i in range(c):
        colFactor[i] = B[0][i] * B[1][i]
        for j in range(1, d):
            colFactor[i] += B[2 * j][i] * B[2 * j + 1][i]

    # calculate resulting matrix
    for i in range(a):
        for j in range(c):
            C[i][j] = -rowFactor[i] - colFactor[j]
            for k in range(d):
                C[i][j] += (A[i][2 * k] + B[2 * k + 1][j]) * (A[i][2 * k + 1] + B[2 * k][j])

    # add terms for odd shared dimension
    if (2 * (b / 2)) != b:
        for i in range(a):
            for j in range(c):
                C[i][j] += A[i][b - 1] * B[b - 1][j]

    return C;


def printMatrix(M):
    for i in M:
        for j in i:
            print(str(j) + "  ", end=" ")
        print()


def GaussJordanWithInverse(equations, inverse):
    # make sure we have N equations and N unknowns (inputs)
    if (len(equations) != len(equations[0]) - 1):
        raise Exception("invalid matrix")

    # make sure inverse is the correct size in relation to the input matrix
    if (len(equations) != len(inverse) or len(equations) != len(inverse[0])):
        raise Exception("invalid inverse")

    h = len(equations)
    solutions = [[0] * (h + 1)] * h

    solutions = list(map(list, equations))

    # perform the Gauss-Jordan elimination
    for i in range(h):  # loop through all rows
        # make the coefficient at the diagonal (location [i][i]) equal to 1
        #    by either multiplication or division
        pivot = float(solutions[i][i])
        pivotcandidate = pivot
        j = i
        candidaterow = i
        # look for a value in the ith column that is not 0
        while j < h and abs(float(pivotcandidate)) < 0.000001:
            pivotcandidate = solutions[j][i]
            candidaterow = j
            j += 1

        # there is no non-zero pivot value in this column
        if (abs(float(pivotcandidate)) < 0.000001):
            raise Exception("singular matrix")

        # swap rows if necessary
        if (candidaterow != i):
            for k in range(h):
                temp = solutions[candidaterow][k]
                solutions[candidaterow][k] = solutions[i][k]
                solutions[i][k] = temp
                temp = inverse[candidaterow][k]
                inverse[candidaterow][k] = inverse[i][k]
                inverse[i][k] = temp
            temp = solutions[candidaterow][h]
            solutions[candidaterow][h] = solutions[i][h]
            solutions[i][h] = temp

        # force the pivot value to 1 by division
        for k in range(h):
            solutions[i][k] /= pivotcandidate
            inverse[i][k] /= pivotcandidate
        solutions[i][h] /= pivotcandidate

        # set all values in this column (i) equal to 0 by subtracting multiples of row i
        for r in range(h):
            if r == i:
                continue
            factor = solutions[r][i]
            for c in range(h):
                value = solutions[i][c] * factor
                solutions[r][c] -= value
                inverse[r][c] -= inverse[i][c] * factor
            solutions[r][h] -= solutions[i][h] * factor


    return solutions


main()
