import numpy as np
from numba import njit

## To unable njit
# def njit(x,**kwargs):
# 	return x

@njit(fastmath=True)
def random_translate_vector ( dr_max, old ):
    """Returns a vector translated by a random amount."""

    # A randomly chosen vector is added to the old one

    zeta = np.random.rand(3)   # Three uniform random numbers in range (0,1)
    zeta = 2.0*zeta - 1.0      # Now in range (-1,+1)
    return old + zeta * dr_max # Move to new position

@njit(fastmath=True)
def overlap_particle(ri,box,r):
	"""Takes a coordinate of an particle and signals any overlap
	"""

	# In general, r will be a subset of the complete set of simulation coordinates
	# and none of its rows should be identical to ri

	# It is assumed that positions are in units where box = 1

	nj, d = r.shape
	assert d==3, 'Dimension error for r in overlap_particle'
	assert ri.size==3, 'Dimension error for ri in overlap_particle'

	inv_box_sq = 1.0 / box ** 2

	rij = ri - r         # Get all separation vectors from partners
	rij = rij - np.rint(rij) # Periodic boundary conditions in box=1 units
	rij_sq = np.sum(rij**2,axis=1)  # Squared separations
	return np.any(rij_sq<inv_box_sq)

@njit(fastmath=True)
def overlap(box, r):
    """Takes in box and coordinate array, and signals any overlap."""

    # Actual calculation is performed by function overlap_1

    n, d = r.shape
    assert d==3, 'Dimension error for r in overlap'

    for i in range(n-1):
        if overlap_particle( r[i,:], box, r[i+1:,:] ):
            return True # Immediate return on detection of overlap

    return False