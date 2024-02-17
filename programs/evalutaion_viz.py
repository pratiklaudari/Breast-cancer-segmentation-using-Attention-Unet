import csv
import matplotlib.pyplot as plt
import numpy as np
file_path=r"C:\Users\sapko\Downloads\segmentation_eval.csv"
e=[]
d_c=[]
j_c=[]
l=[]
a=[]
v_l=[]
v_a=[]
with open(file_path, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        e.append(float(row[0]))
        d_c.append(float(row[1]))
        j_c.append(float(row[2]))
        l.append(float(row[3]))
        a.append(float(row[4]))
        v_l.append(float(row[5]))
        v_a.append(float(row[6]))

plt.plot(e, d_c)
plt.show()

plt.plot(e, j_c)
plt.show()

plt.plot(e, l)
plt.show()

plt.plot(e, a)
plt.show()

plt.plot(e, v_l)
plt.show()

plt.plot(e, v_a)
plt.show()


