import numpy as np
cimport cython

# C function to generate the product of the denominators
# We turn off bounds checking an negative indicies to get C like speed
@cython.boundscheck(False)
@cython.wraparound(False)
cdef float calc_product(double[:] a, Py_ssize_t n, double d):

    cdef double s = a[n-1] + d   # Inner sum
    cdef double p = s            # Product
    cdef Py_ssize_t i
    cdef double result

    for i in range(n-2,-1,-1):
        s += a[i]
        p *= s

    result = 1.0/p

    return result

# Python function to calculate permutations and evaluate expression
# We turn off bounds checking and negative indicies to get C like speed
@cython.boundscheck(False)
@cython.wraparound(False)
def permuteexpression(t, dp):

    # We capture the length of the array as we will need that later
    cdef Py_ssize_t n = len(t)
    
    # Create a copy of our input tuple and get a memoryview of it
    # so we can manipulate it efficiently from C
    # This will be our first permutation.
    # Subsequent permutations will involve doing swaps of this memoryview
    cdef double[:] a = t.copy()

    # Convert python float to C float as we use this often
    cdef double d = dp

    # Reserve memory for local indexing array
    # Assume 64 bit pointers i.e. Py_ssize_t is same as np.int64
    cnp = np.zeros(n, dtype=np.int64)

    # Get a memoryview of our indexing array for efficient C manipulation
    cdef Py_ssize_t[:] c = cnp

    # Define remaining local variables
    cdef Py_ssize_t i
    cdef double tmp
    cdef double outer_sum = calc_product(a, n, d) # Process first permutation
    cdef double numerator = 1.0

    # Calculate factored out product
    for i in range(n):
        numerator *= a[i]

    # Generate the remaining permutations
    i = 1
    while i < n:
        if c[i] < i:
            if i % 2 == 0:
                # Swap a[0] with a[i]
                tmp = a[0]
                a[0] = a[i]
                a[i] = tmp

            else:
                # Swap a[c[i]] with a[i]
                tmp = a[c[i]]
                a[c[i]] = a[i]
                a[i] = tmp

            # Process new permutation
            outer_sum += calc_product(a, n, d)

            c[i] += 1
            i = 1

        else:
            c[i] = 0
            i += 1

    return numerator * outer_sum