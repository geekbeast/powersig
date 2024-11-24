# PowerSig
Using power series to compute signatures.

## Algorithms

This library implements three different power series approaches to computing the signature kernel
with varying levels of stability and performance. All of implemented the approaches in this library 
are currently based upon ADM. The first two schemes below work by repeatedly solving for the explicit 
truncated power series representation of 

### Truncated coefficients
This is the most accurate, but slowest scheme. It uses a coefficient vector and two exponent vectors
to maintain an arbitrarily precise power series representation of the 
### Self-truncating Coefficient Matrices
This approach is similar to the truncated coefficients scheme, where the linear integral operator
will automatically discard coefficients that grow two large.

### Fixed Point ADM Power Series
Unlike the other approaches which build up the solution by repeatedly solving the boundary conditions
for each tile. This approach is direct power series computation of the signature scheme based on an
analytic solution of the Goursat PDE.

