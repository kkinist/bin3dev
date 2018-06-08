#!/usr/bin/python3
# test use of multiprocessing within class methods
#
import multiprocessing
import numpy as np
#
class VecList(object):
    def __init__(self, rect_matrix):
        self.mat = np.array(rect_matrix)
    def rowsum(self, irow):
        s = self.mat[irow,:].sum()
        print('SSS s = ', s)
        return s
    def nrow(self):
        return self.mat.shape[0]
    def print(self):
        print(self.mat)
        return
    def sumrows(self):
        # sum all rows, return a vector
        print('sum each rows')
        if __name__ == '__main__':
            #tasks = np.arange(self.nrow()).astype(int)
            tasks = [self.mat[i,:] for i in range(self.nrow())]
            print('tasks = ', tasks)
            pool = multiprocessing.Pool(2)
            #results = [pool.apply_async(self.rowsum, t) for t in tasks]
            results = [pool.apply_async(simplesum, t) for t in tasks]
            pool.close()
            pool.join()
            s = []
            for result in results:
                s1 = result.get()
                s.append(s1)
        return np.array(s)
##
def simplesum(vec):
    # return sum of elements
    s = vec.sum()
    return s
##
vl = VecList(np.arange(15).reshape((5,3)))
vl.print()
if False:
    # do serially
    for irow in range(vl.nrow()):
        s = vl.rowsum(irow)
        print('{:d} : {:d}'.format(irow, s))
print('now try in parallel')
s = vl.sumrows()
print('s = ', s)
