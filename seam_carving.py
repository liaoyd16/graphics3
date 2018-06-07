
# coding: utf-8

'''
# seam_carving.py
1. 读入图片 -> numpy.array
2. 计算energy_distribution
3. DP缓存
4(debug). 求出carving线，打印到numpy.array中
4(test). 剪切carving线上的像素
'''

import numpy as np
import cv2
from cv2 import imread, imwrite
import copy

import time

# picture object
class Picture(object):
    def __init__(self, picname):
        super(Picture, self).__init__()
        self.picture = imread(picname) # numpy ?
        self.x_range = len(self.picture)
        self.y_range = len(self.picture[0])

    @property
    def rangex(self):
        return x_range

    @property
    def rangey(self):
        return y_range

    def carve_y(self, seam_list):
        for row_s in seam_list:
            self.picture[ row_s[0] ][ row_s[1]:-1 ][:] = self.picture[ row_s[0] ][ row_s[1]+1: ][:]

        self.y_range -= 1

    def carve_x(self, seam_list):
        for pair in seam_list:
            self.picture[ pair[0]:-1, pair[1], :] = self.picture[ pair[0]+1:, pair[1], :]

        self.x_range -= 1

    def modify(self, x, y): # color it red
        self.picture[x][y][:] = [255, 0, 0]

    def write(self, filename):
        imwrite(filename, self.picture)

    def pad1(self):
        new_picture = np.zeros((self.rangex+2, self.rangey+2, 3))
        new_picture[ 1:self.rangex+1 , 1:self.rangey+1, : ] = self.picture
        return new_picture

# energy estimator
class EnergyEstimator(object):
    def __init__(self, picture):
        super(EnergyEstimator, self).__init__()
        self._pic = picture
        self._energy_distr = np.zeros((picture.rangex, picture.rangey))

    def estimate(self):
        pass

    def get_distr(self):
        return self._energy_distr


class Sobel_Estimator(EnergyEstimator):
    def __init__(self, picture):
        super(Sobel_Estimator, self).__init__(picture)

    def convo(self, padded, x, y, filt):
        ans = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                ans += np.sum(padded[x+i][y+i]) * filt[i+1][j+1]
        return ans

    def estimate(self):
        # do padding = 1
        # using np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
        # abs() / squared()
        padded = self._pic.pad1()

        filt1 = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
        filt2 = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
        for x in range(self._pic.rangex):
            for y in range(self._pic.rangey):
                self._energy_distr[x][y] += max(abs(self.convo(padded, x, y, filt1)), abs(self.convo(padded, x, y, filt2))) #self._pic.picture[x-1:x+2][y-1:y+2]

class E1_Estimator(EnergyEstimator):
    def __init__(self, picture):
        super(E1_Estimator, self).__init__(picture)

    def estimate(self):
        # using E1
        pass

class Entropy_Estimator(EnergyEstimator):
    def __init__(self, picture):
        super(Entropy_Estimator, self).__init__(picture)

    def estimate(self):
        # using entropy method
        pass

def carve_one_y(picture):
    x_range = picture.rangex
    y_range = picture.rangey

    estimator = Sobel_Estimator(picture)
    estimator.estimate()
    energy_distr = copy.copy(estimator.get_distr())

    # cv2.imwrite("energy.png", energy_distr)

    # 'from' table
    from_table = np.zeros((x_range, y_range), dtype='uint16')

    # get from_table[x][y]
    for x in range(1, x_range):
        for y in range(y_range):
            candid = [energy_distr[x-1][y]]
            if y-1 >= 0:
                candid.insert(0, energy_distr[x-1][y-1])
            if y+1 < y_range:
                candid.append(energy_distr[x-1][y+1])
            energy_distr[x][y] += min(candid)

            from_table[x][y] = y + np.argmin(candid) - 1

    # sort index
    # priority = zip(energy_distr[x_range-1], np.arange(0, y_range))
    # priority = sorted(priority, key=lambda x: x[0])
    yrun = 0
    min_energy = energy_distr[x_range-1][0]
    for i in range(1, y_range):
        if energy_distr[x_range-1][i] < min_energy:
            min_energy = energy_distr[x_range-1][i]
            yrun = i

    # # 4 - test:
    # for del_id in range(deleted):
    xrun = x_range - 1
    seam_list = [(xrun, yrun)]

    while xrun > 0:
        yrun = from_table[xrun][yrun]
        xrun -= 1
        seam_list.append((xrun, yrun))

    picture.carve_y(seam_list)

    print("done")

def carve_one_x(picture):
    x_range = picture.rangex
    y_range = picture.rangey


    t1 = time.time()
    estimator = Sobel_Estimator(picture)
    estimator.estimate()
    energy_distr = copy.copy(estimator.get_distr())
    t2 = time.time()
    print("estimate_time = {}".format(t2 - t1))

    # 'from' table
    from_table = np.zeros((x_range, y_range), dtype='uint16')

    # get from_table[x][y]
    for y in range(1, y_range):
        for x in range(x_range):
            candid = [energy_distr[x][y-1]]
            if x-1 >= 0:
                candid.insert(0, energy_distr[x-1][y-1])
            if x+1 < x_range:
                candid.append(energy_distr[x+1][y-1])
            energy_distr[x][y] += min(candid)

            from_table[x][y] = x + np.argmin(candid) - 1

    xrun = 0
    min_energy = energy_distr[xrun][y_range-1]
    for i in range(1, x_range):
        if energy_distr[i][y_range-1] < min_energy:
            min_energy = energy_distr[i][y_range-1]
            xrun = i

    # # 4 - test:
    yrun = y_range - 1
    seam_list = [(xrun, yrun)]

    while yrun > 0:
        xrun = from_table[xrun][yrun]
        yrun -= 1
        seam_list.append((xrun, yrun))

    picture.carve_x(seam_list)

    print("done")

def deal_pic(picname, direction):
    written_name = "cut-" + picname + "-" + direction + ".png"

    picture = Picture("./Images/" + picname + ".jpg")

    if direction=="x":
        x_range0 = picture.rangex
        times = int(0.2*x_range0)
        for i in range(times):
            carve_one_x(picture)
    else:
        y_range0 = picture.rangey
        times = int(0.2*y_range0)
        for i in range(times):
            carve_one_y(picture)

    picture.write(written_name)

# main procedure
if __name__=="__main__":
    
    picnames = ["1", "2", "3", "4", "5", "6"]
    for pic in picnames:
        deal_pic(pic, "x")
        deal_pic(pic, "y")

