import numpy as np


def local_min_1d(A):
    if A is None or len(A) == 0:
        return None
    return local_min_1d_helper(A, 0, len(A)-1)

def local_min_1d_helper(A, lo, hi):
    mid = (lo+hi)//2
    l,m,r = max(0,mid-1), mid, min(mid+1,hi)
    if A[l]<A[m]: #local min on left side
        return local_min_1d_helper(A, lo, m-1)
    elif A[r]<A[m]: #local min on right side
        return local_min_1d_helper(A, m+1,hi)
    else: #m is local min - functions as base case
        return m

def local_min_2d(A):
    if A is None or len(A)==0:
        return None
    return local_min_2d_helper(A, 0, len(A)-1)

def local_min_2d_helper(A, lo, hi):
    mid = (lo+hi)//2
    min_col_idx = np.argmin(A[mid,:])
    #find the row (above or below mid with smallest value in
    #the column
    l,m,r = max(0,mid-1), mid, min(mid+1,hi)
    #col index is guaranteed as local min, so just check the
    #rows above and below mid (i.e. we are in 1d case)
    tmp = A[:,min_col_idx]
    if tmp[l]<tmp[m]: #local min below
        return local_min_2d_helper(A, lo, m-1)
    elif tmp[r]<tmp[m]: #local min above
        return local_min_2d_helper(A, m+1,hi)
    else: #at a local min
        return m, min_col_idx



def two_d_neighbors(A, r, c):
    m,n = A.shape
    cl,ch,rl,rh = max(0, c - 1), min(n-1, c + 1), max(0, r - 1), min(m - 1, r + 1)
    return (r,cl),(r,ch),(rl,c),(rh,c)
#Test
n = 100
x,y = 10,10
for _ in range(n):
    A = np.random.randint(0,100,(x,y))
    r,c = local_min_2d(A)
    neighbors = two_d_neighbors(A, r,c)
    assert np.alltrue(np.array([A[i,j] for (i,j) in neighbors])>=A[r,c])
    print(np.array([A[i,j] for (i,j) in neighbors]),A[r,c])

from typing import List
from random import randint

import numpy as np


# Part A - Deterministic strategy

def strategy_a(w: List[int]) -> int:
    """
    :param m: The number of metal detectors available to process guests.
    :param w: A list of m integers, where w[i] contains the # of guests
    currently waiting at detector i.
    :return: Index of the detector where the next guest should be sent.
    Should be an integer in the range {0, 1, ..., m-1}.
    """
    return np.argmin(w)


# Part A theory answer(s)
"""
Key assumptions:
- All guests take the same amount of time to process
- All detectors process guests at the same rate

Deterministic strategies do not perform well when used concurrently with other guards.
You and the k other guards will assign k+1 guests to the same line simultaneously.

For argmin strategy:
w' = max(w_max, w_min + k + 1)

For first line smaller than some other line strategy:
w' = max(w_max, w_j + k + 1) where j is this index
"""


# Part B - Randomized strategy

def strategy_b(m: int) -> int:
    """
    :param m: The number of metal detectors available to process guests.
    :return: Index of the detector where the next guest should be sent.
    Should be an integer in the range {0, 1, ..., m-1}.
    """
    return randint(0, m - 1)


# todo: Part B theory answer(s)
"""
The number of guests sent to detector i is a binomial random variable with probability of success p=1/40, n=60 independent trials. 
=> Pr(X_i > 5) =  1 - Pr(X_i <= 5) = 0.00385 for any detector i.

Note that the X_i's are not independent - they are negatively correlated.
Applying the union bound,
Pr(X_i > 5 for any i) <= sum_{i=0}^{m-1} P(X_i) = 40*0.00385 = 0.154

The union bound double-counts any intersections, and is only completely tight when the events are disjoint.
In this case, the events are negatively correlated, so the intersections are very unlikely.
As a result, the union bound does a very good job.

Simple randomized strategy has at least an 84.6% chance of preventing very long lines.
"""


# Part C - Smarter randomized strategy

def strategy_c(m: int, w: List[int]) -> int:
    """
    :param m: The number of metal detectors available to process guests.
    :param w: A list of m integers, where w[i] contains the # of guests
    currently waiting at detector i.
    :return: Index of the detector where the next guest should be sent.
    Should be an integer in the range {0, 1, ..., m-1}.
    """
    # Power of two random choices - sample two indices without replacement
    i,j = np.random.choice(range(m),2,replace=False)
    return i if w[i]<w[j] else j


# todo: Part C theory answer(s)
"""
There are two ways we might increase w_j:
- j is picked on the first draw and one of the t gates with a longer line is picked on the second
- j is picked on the second draw and one of the t gates with a longer line is picked on the first.

The probability that j is picked on the first random draw is 1/m
The probability that j is picked on the second random draw is 1/(m-1)
The probability that one of the t gates with a longer line is picked on the first draw is t/m
The probability that such a gate is picked on the second draw is t/(m-1)

The two events are disjoint, so the total probability is (1/m)(t/(m-1)) + (1/(m-1))(t/m) = 2t/(m(m-1))
"""

# record beginning of OH for soln to programming prblm
# zoom link on canvas
# update OH on site and ed
# meet with gerry thursday (4pm)
# write up template for DP + proof of correctness
# write rubric for HW2 (slack)


