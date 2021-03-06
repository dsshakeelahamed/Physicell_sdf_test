import numpy as np
import pandas as pd

r = 4.330127018922193
# r = 3
x_val = []
r_val = []

def generate_cell_data(start, end, cellId):
    # layer 1
    dv = 2.5
    x1 = np.linspace(start, end, num=8)
    x2 = np.linspace(start + dv, end - dv, num=8)
    x3 = np.linspace(start, end , num=8)
    x4 = np.linspace(start + dv, end - dv, num=8)
    x5 = np.linspace(start, end, num=8)



    y1 = np.array([r] * 8)
    y2 = np.array([2 * r] * 8)
    y3 = np.array([3 * r] * 8)
    y4 = np.array([4 * r] * 8)
    y5 = np.array([5 * r] * 8)

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
# df1 = generate_cell_data(-110, -70, 1)
# df2 = generate_cell_data(-65, -25, 2)
# # print(df1)
# df3 = generate_cell_data(-20, 20, 3)
# df4 = generate_cell_data(25, 65, 4)
# df5 = generate_cell_data(70, 110, 5)

# df1 = generate_cell_data(-90, -50, 1)
# df2 = generate_cell_data(-55, -15, 2)
# # print(df1)
# df3 = generate_cell_data(-20, 20, 3)
# df4 = generate_cell_data(15, 55, 4)
# df5 = generate_cell_data(50, 90, 5)


# df1 = generate_cell_data(-96, -56, 1)
# df2 = generate_cell_data(-58, -18, 2)
# # print(df1)
# df3 = generate_cell_data(-20, 20, 3)
# df4 = generate_cell_data(18, 58, 4)
# df5 = generate_cell_data(56, 96, 5)

df1 = generate_cell_data(-100, -60, 1)
df2 = generate_cell_data(-60, -20, 2)
# # print(df1)
df3 = generate_cell_data(-20, 20, 3)
df4 = generate_cell_data(20, 60, 4)
df5 = generate_cell_data(60, 100, 5)
#
#
df1 = pd.concat([df1, df2, df3, df4, df5], axis=0)
# df1 = pd.concat([df1, df2, df3], axis= 0)
df1.to_csv("unittest_data.csv", header=False, index=False)

# normal_vector
# df_norm = df1[['x', 'y']]
# df_norm['y'] = 0
# print(df_norm.shape)
# print(df_norm)


def generate_1d_sdf(xmin, xmax, dx, ymin, ymax, dy):

    y_values = np.linspace(ymin, ymax, (ymax - ymin)//dy)
    y_values = y_values * (-1)

    y1 = y_values
    x_values = np.linspace(xmin, xmax, (xmax - xmin)//dx)

    y_values = np.reshape(y_values, (1, len(y_values)))
    df1 = pd.DataFrame(y_values)
    print(df1.shape)
    temp = pd.DataFrame()

    # temp = []
    for i in range(len(x_values)):
        temp = pd.concat([temp, df1], axis=0)

    temp.to_csv("sdf.csv", header=False, index=False)

    # norm
    out = pd.DataFrame()
    print(x_values.shape, y1.shape)
    for i in x_values:
        for j in y1:
            val = np.array([(i, 0)])
            val = pd.DataFrame(val)
            out = pd.concat([out, val])
    # print(out)

    out.to_csv("normal_data.csv", header=False, index=False)
    # return temp

    # print(y_values)
    # print(temp.shape)
    # print(temp)

generate_1d_sdf(-150, 150, 1, -50, 50, 1)
# sdf.to_csv("sdf.csv", header=False, index=False)



