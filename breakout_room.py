#Largest Sum contigous subarray A[a..b] of A[1..n]
#Base case:
#Array of length 0 has a largest sum contigous subarray of value 0

#Subproblem:
#T[i] is largest sum in a contigous subarray of A[1..i], ending at A[i]

#Recurrence:
#T[i] = max(T[i-1] + A[i], A[i])

#Order:
#for i=1..n

#Solution:
#return max(T[0..n])

#Example: A= [-5,3,4,-6,11]

#T=[0,-5,3,7,1,12]

#Run time:
#Tables have exactly n entries, each entry is computed exactly once, and each takes O(1) time to compute.
#computing table takes O(n) time, and finding solution takes O(n) time, so O(n) overall

#proof
#let a* and b* be the "optimal" start and end points
#so A[a*]+...+A[b*] is maximal for all contigous subarrays.
#guaranteed that 0<=b*<=n
#=> T[b*] is the solution

#---------------------------------------------------------------

#Problem : Largest Product contigous subarray

# Base case: (what to do with empty array)
# maximum product of empty array in 1.
#m[0] = M[0] = 1

# Subproblem: (define what the elements of table represent)
#WANT TO KNOW : Maximum product of contigous subarray of A[1..i] ending at A[i]
#Observations :
#We actually want to know both the maximum and minimum product we can make with A[1..i]
#M[i] : The maximum product of elements in a contigous subarray of A[1..i] ending with A[i]
#m[i] : The minimum product of elements in a contigous subarray of A[1..i] ending with A[i]

#Recurrence:
#If A[i] is negative, then max(A[i], m[i-1]*A[i]) is the largest possible product
#If A[i] is positive, then max(A[i], M[i-1]*A[i]) is the largest possible product
# M[i] = max(A[i], m[i-1]*A[i], M[i-1]*A[i])
# m[i] = min(A[i], m[i-1]*A[i], M[i-1]*A[i])

#Order to fill table:
#i=1..n

#Solution:
#Max(M[0..n])

#Run time:
#Tables have exactly n entries, each entry is computed exactly once, and each takes O(1) time to compute.
#computing table takes O(n) time, and finding solution takes O(n) time, so O(n) overall

#run time =(size of table) * (time to compute an entry) + (time to find solution)

#Example: [-6, 12, -7, 0, 14, -7, 5]

#m = [1, -6,  -72,  -84, 0,  0, -98, -490]
#M = [1, -6,   12,  504, 0, 14,  0,  5]










