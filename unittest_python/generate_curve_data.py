import math
import sys

import numpy as np
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import pandas as pd


x = np.array([-150 , -75, 0 , 75, 150])
y = np.array([0, -2.5, -5, -2.5, 0])
t, c, k = interpolate.splrep(x, y, s=0, k=2)
# print('''\
# t: {}
# c: {}
# k: {}
# '''.format(t, c, k))
# N = 300
# xmin, xmax = x.min(), x.max()
# xx = np.linspace(xmin, xmax, N)
# print(xx)
# spline = interpolate.BSpline(t, c, k, extrapolate=False)
# yy = spline(xx)
# print(np.shape(xx))
# # np.reshape(xx, newshape=[np.shape(xx)[1], np.shape(xx)[0]])
# print(xx)
# xx = np.reshape(xx, (xx.shape[0], 1))
# yy = np.reshape(yy, (yy.shape[0], 1))
# z = np.concatenate([xx, yy], axis=1)
# print(z)
# print(spline(xx))
# plt.plot(x, y, 'bo', label='Original points')
# plt.plot(xx, spline(xx), 'r', label='BSpline')
# plt.grid()
# plt.legend(loc='best')
# plt.show()

def generate_1d_sdf_curve(xmin, xmax, dx, ymin, ymax, dy):
    # steps
    # 1) Find mid the point on bspline data to use as reference
    # 2) then for each y co ordinate of the point calculate distance from bspline y
    # x = np.array([-150, -75, 0, 75, 150])
    # y = np.array([0, -2.5, -5, -2.5, 0])
    # -5 max
    # x = np.array([-150, -100, -60, -20, 0, 20, 60, 100, 150])
    # y = np.array([0, 0, -1.67, -3.33, -5,  -3.33, -1.67, 0, 0])
    # -10 max
    # x = np.array([-150, -100, -60, -20, 0, 20, 60, 100, 150])
    # y = np.array([0, -1, -3.33, -6.67, -10, -6.67, -3.33, -1, 0])

    x = np.array([-150, 0, 150])
    y = np.array([0, -20, 0])
    N = 300
    t, c, k = interpolate.splrep(x, y, s=0, k=2)
    # xmin, xmax = x.min(), x.max()
    xx = np.linspace(xmin, xmax, N)
    spline = interpolate.BSpline(t, c, k, extrapolate=False)
    yy = spline(xx)
    # print(xx)
    # print(yy)
    xx = np.reshape(xx, (xx.shape[0], 1))
    yy = np.reshape(yy, (yy.shape[0], 1))
    z = np.concatenate([xx, yy], axis=1)

    y_values = np.linspace(ymin, ymax, (ymax - ymin)//dy)
    # y_values = y_values * (-1)

    # y1 = y_values
    x_values = np.linspace(xmin, xmax, (xmax - xmin)//dx)
    temp = pd.DataFrame()
    norm_out = pd.DataFrame()
    for x in x_values:
    #     find closest bspline point
        arr_val = np.array([])
        for y in y_values:
            index_closest = 0
            min_x = 0
            min_y = 0
            min_val = sys.maxsize
            for j in range(len(z)):
                dist = math.sqrt(pow(x-z[j][0], 2) + pow(y-z[j][1], 2))
                if dist < min_val:
                    min_val = dist
                    min_x = z[j][0]
                    min_y = z[j][1]
                    index_closest = j
            if y >= min_y:
                arr_val = np.append(arr_val, -min_val)
            else:
                arr_val = np.append(arr_val, min_val)
            norm_v = np.array([(min_x, min_y)])
            norm_v = pd.DataFrame(norm_v)
            norm_out = pd.concat([norm_out, norm_v])

        arr_val = np.reshape(arr_val, (1, len(arr_val)))
        co_ordinate_df = pd.DataFrame(arr_val)
        temp = pd.concat([temp, co_ordinate_df], axis=0)

    # print(temp.head())
    temp.to_csv("sdf.csv", header=False, index=False)
    norm_out.to_csv("normal_data.csv", header=False, index=False)



generate_1d_sdf_curve(-150, 150, 1, -50, 50, 1)

r = 4.330127018922193
def generate_cell_data(x_start, x_end, y_start, y_end, cellId):
    # layer 1

    dv = 0.5
    x1 = np.linspace(x_start, x_end, num=8)
    x2 = np.linspace(x_start + dv, x_end - dv, num=8)
    x3 = np.linspace(x_start, x_end, num=8)
    x4 = np.linspace(x_start + dv, x_end - dv, num=8)
    x5 = np.linspace(x_start, x_end, num=8)

    y1 = np.linspace(y_start + r, y_end + r, num=8)
    y2 = np.linspace(y_start + 2*r, y_end + 2*r, num=8)
    y3 = np.linspace(y_start + 3*r, y_end + 3*r, num=8)
    y4 = np.linspace(y_start + 4*r, y_end + 4*r, num=8)
    y5 = np.linspace(y_start + 5*r, y_end + 5*r, num=8)

    # y1 = np.array([r] * 8)
    # y2 = np.array([2 * r] * 8)
    # y3 = np.array([3 * r] * 8)
    # y4 = np.array([4 * r] * 8)
    # y5 = np.array([5 * r] * 8)

    x = np.concatenate((x1, x2, x3, x4, x5), axis=0)
    y = np.concatenate((y1, y2, y3, y4, y5), axis=0)
    x = np.reshape(x, (x.shape[0], 1))
    y = np.reshape(y, (y.shape[0], 1))
    df_data = np.concatenate((x, y), axis=1)

    df = pd.DataFrame(df_data, columns=['x', 'y'])
    df['z'] = 0.0
    df['type'] = 0
    df['id'] = cellId
    return df

# x = np.array([-150, -100, -60, -20, 0, 20, 60, 100, 150])
# y = np.array([0, 0, -1.67, -3.33, -5,  -3.33, -1.67, 0, 0])

# -10 max
# x = np.array([-150, -100, -60, -20, 0, 20, 60, 100, 150])
# y = np.array([0, -1, -3.33, -6.67, -10, -6.67, -3.33, -1, 0])
x = np.array([-150, 0, 150])
y = np.array([0, -20, 0])
N = 300
t, c, k = interpolate.splrep(x, y, s=0, k=2)
xmin, xmax = x.min(), x.max()
xx = np.linspace(xmin, xmax, N)
spline = interpolate.BSpline(t, c, k, extrapolate=False)
yy = spline(xx)


df1 = generate_cell_data(-100, -60, yy[-100 + 150], yy[-60 + 150], 1)
df2 = generate_cell_data(-60, -20, yy[-60 + 150], yy[-20 + 150], 2)
# print(df1)
df3 = generate_cell_data(-20, 20, yy[-20 + 150], yy[20 + 150], 3)
df4 = generate_cell_data(20, 60, yy[20 + 150], yy[60 + 150], 4)
df5 = generate_cell_data(60, 100, yy[60 + 150], yy[100 + 150], 5)

df1 = pd.concat([df1, df2, df3, df4, df5], axis=0)
# df1 = pd.concat([df1, df2, df3], axis=0)
# df1 = pd.concat([ df2, df3, df4], axis=0)
df1.to_csv("unittest_data.csv", header=False, index=False)

plt.plot(x, y, 'bo', label='Original points')
plt.plot(xx, spline(xx), 'r', label='BSpline')
plt.grid()
plt.legend(loc='best')
plt.show()



'''
    for i in range(len(x_values)):
    #     find closest bspline point
        x = x_values[i]
        index_closest = 0
        min_val = sys.maxsize
        for j in range(len(z)):
            dist = math.sqrt((x))
            if abs(x-z[j][0]) < min_val:
                min_val = abs(x-z[j][0])
                index_closest = j

        arr_val = np.array([])
        y_sdf = z[index_closest][1]
        for y_val in y_values:
            arr_val = np.append(arr_val, y_sdf - y_val)
            norm_v = np.array([(x, y_sdf)])
            norm_v = pd.DataFrame(norm_v)
            norm_out = pd.concat([norm_out, norm_v])
        arr_val = np.reshape(arr_val, (1, len(arr_val)))
        co_ordinate_df = pd.DataFrame(arr_val)
        # print(co_ordinate_df.head(), i)
        temp = pd.concat([temp, co_ordinate_df], axis=0)
        '''