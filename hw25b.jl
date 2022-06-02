using LinearAlgebra
using Printf
Q = [5 -1; -1 2];
eigvalues=(eigvals(Q))
print(eigvalues)
print("")
print(1/(eigvalues[2,1]/2))