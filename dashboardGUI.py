import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import os
import os.path
from os.path import join


dir_path = os.path.dirname(os.path.realpath(__file__))
path = join(dir_path, 'graph')
file_name1 = 'translation.txt'
file_name2 = 'rotational.txt'
file_name3 = 'motion.txt'

filename1 = join(path, file_name1)
filename2 = join(path, file_name2)
filename3 = join(path, file_name3)


fig = plt.figure()
ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)

ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 3)
ax3 = plt.subplot(1, 2, 2)


def animate(i):
    pullData1 = open(filename1,"r").read()
    pullData2 = open(filename2,"r").read()
    pullData3 = open(filename3,"r").read()
    dataArray1 = pullData1.split('\n')
    dataArray2 = pullData2.split('\n')
    dataArray3 = pullData3.split('\n')
    xar_d1  = []
    yar_d1  = []
    xar_d2  = []
    yar_d2  = []
    xar_d3  = []
    yar_d3  = []
    x1ar = []
    x2ar = []
    x3ar = []
    x4ar = []
    x5ar = []
    x6ar = []
    x7ar = []
    x8ar = []

    for eachLine in dataArray1:
        if len(eachLine)>1:
            x,x1,x2,x3 = eachLine.split(',')
            xar_d1.append(int(x))
            yar_d1.append(0)
            x1ar.append(float(x1))
            x2ar.append(float(x2))
            x3ar.append(float(x3))
            
    for eachLine in dataArray2:
        if len(eachLine)>1:
            x,x4,x5,x6 = eachLine.split(',')
            xar_d2.append(int(x))
            yar_d2.append(0)
            x4ar.append(float(x4))
            x5ar.append(float(x5))
            x6ar.append(float(x6))
            
    for eachLine in dataArray3:
        if len(eachLine)>1:
            x,x7,x8 = eachLine.split(',')
            xar_d3.append(int(x))
            yar_d3.append(0)
            x7ar.append(float(x7))
            x8ar.append(float(x8))

    ax1.clear()
    ax1.plot(xar_d1, x1ar, label = "Translation in X Direction")
    ax1.plot(xar_d1, x2ar, label = "Translation in Y Direction")
    ax1.plot(xar_d1, x3ar, label = "Translation in Z Direction")
    #ax1.plot(xar_d1, yar_d1, label = "Zero Axis")
    ax1.set_title('Translation')
    ax2.clear()
    ax2.plot(xar_d2, x4ar, label = "Rotation in X Direction")
    ax2.plot(xar_d2, x5ar, label = "Rotation in Y Direction")
    ax2.plot(xar_d2, x6ar, label = "Rotation in Z Direction")
    #ax2.plot(xar_d2, yar_d2, label = "Zero Axis")
    ax2.set_title('Rotation')
    ax3.clear()
    ax3.plot(xar_d2, x7ar, label = "Absolute Displacement")
    ax3.plot(xar_d2, x8ar, label = "Relative Displacement")
    #ax2.plot(xar_d2, yar_d2, label = "Zero Axis")
    ax3.set_title('Head Motion')
    
ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()
