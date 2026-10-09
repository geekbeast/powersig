# Inactive diagonal cells and numerical safety

A fixed-width diagonal sweep contains lanes outside the rectangular path-pair
grid, especially at the first and last diagonals. These lanes are not kernel
tiles. Evaluating them with wrapped/clamped path indices can repeatedly reuse
a large increment, overflowing internal polynomial states even when every
real tile and the final kernel value are finite.

Mask the inputs **before** the static kernel and power-series recurrence:
use in-bounds gather indices, select boundary conditions with `where` rather
than multiplication by a Boolean, and give inactive lanes zero boundary
vectors and zero increments. Masking only the final output is insufficient
for differentiation: an unused infinity can still produce `0 * Inf = NaN`
in a pullback. Gradient clipping after backpropagation cannot recover those
NaNs.

The custom-VJP implementation, when present, must use the same rule in its
checkpointed forward sweep, forward replay, local adjoints, and path-gradient
scatter. Inactive lanes must never contribute to either boundary or path
adjoints. This is geometric masking, not gradient clipping or sanitizing
non-finite values in real tiles. Genuine overflow in an active tile remains
visible.

## Reproduction and verification

The tests use only synthetic FP32 paths: a 64-knot scalar query from zero to
one, and a bank path with one jump from zero to 256 followed by constant
knots, at numerical order 9. The unpatched sweep gives a finite kernel value
but non-finite path gradients; the masked implementation agrees with an
independent row-major scan that visits only real tiles.

The reference checks the same finite-order numerical recurrence, not the
accuracy of that recurrence against the exact signature kernel for large
increments. Cancellation in the stress-case gradient permits eight FP32 ULPs
at the largest component's scale; ordinary small-path tests retain strict
elementwise tolerances. Additional tests cover linear and RBF kernels,
unequal lengths in both orientations, one-increment paths, chunked and
unchunked sweeps, and genuinely overflowing active tiles.
