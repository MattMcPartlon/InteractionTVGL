from typing import List, Tuple
import numpy as np
PLAYER, SCORE = 0, 1
from abc import ABC, abstractmethod
from collections import defaultdict

class DataStructure(ABC):
    def __init__(self, S: List[Tuple[str, int]]):
        self.S = S

    @abstractmethod
    def lowest_player(self, start, end) -> str:
        pass

    @abstractmethod
    def fill_dp_table(self):
        pass


class Datastructure1(DataStructure):
    """
    TODO : answer parts 1-4
    """

    def __init__(self, S):
        super().__init__(S)
        self.T = defaultdict(lambda : ('NONE',1e10))
        self.n = len(self.S)
        if S:
            self.fill_dp_table()

    def lowest_player(self, start: int, end: int) -> str:
        print('------------------')
        return self.lowest_player_helper(start,end)[PLAYER]

    def lowest_player_helper(self, start, end):
        start,end = int(start),int(end)
        assert start<=end<self.n
        # split up the interval by powers of 2
        interval_len = (end - start) + 1
        #Base case
        if interval_len <=2:
            print(start,end)
            print(self.T[(0,start)], self.T[(0,end)])
            print('lowest',self.lowest(self.T[(0,start)], self.T[(0,end)]))
            return self.lowest(self.T[(0,start)], self.T[(0,end)])

        x = max(0,np.floor(np.log2(interval_len))-1)
        #search for the lowest score in the largest interval contained here
        #i*2**x, (i+1)*2**x
        i = (end-1)//(2**x)
        print('left ',start, int(max(start, (i-1)*(2**x)-1)),start,end)
        left_min = self.lowest_player_helper(start, max(start, (i-1)*(2**x)-1))
        mid_min = self.T[(x,max(0,i))]
        print('mid :',mid_min)
        print('mid',int((i-1)*(2**x)),int(i*2**x),start,end)
        right_min = self.lowest_player_helper(max(end,i*(2**x)+1), end)
        print('right :',int(max(end,i*(2**x))+1), end,start,end)
        return self.lowest(left_min,right_min,mid_min)

    def lowest(self, *args):
        scores = [a[SCORE] for a in args]
        return args[int(np.argmin(scores))]


    def fill_dp_table(self):
        for i in range(self.n):
            self.set_table_entry(0,i,self.S[i])
        for x in range(1,int(np.ceil(np.log2(self.n)))):
            idx = 0
            while idx < self.n/(2**(x-1))+1:
                p1,p2 = self.T[(x-1,idx)],self.T[(x-1,idx+1)]
                if p1[SCORE]<p2[SCORE]:
                    self.T[(x,idx//2)]=p1
                else:
                    self.T[(x,idx//2)]=p2
                idx+=2

    def set_table_entry(self, x: int, i: int, value: Tuple[str, int]):
        self.T[(x, i)] = value


def _bin(x : int):
    bits = []
    while x:
        x, rmost = divmod(x, 2)
        bits.append(rmost)
    return bits