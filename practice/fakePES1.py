from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

# 関数の定義
x = np.linspace(-1.5, 2.0, 50)
y = np.linspace(-3.0, 1.7, 50)
x, y = np.meshgrid(x, y)
# z = x**3 - 3*x*y + np.exp((x**2 + y**2)/2.0)
# X, Y, Z = axes3d.get_test_data(0.2)

# 関数の定義
def f(x, y):
    return (1.0*np.sin(3*x)+1.0*np.sin(3*y))*np.exp(-((x+0.2)**2 + (y-0.2)**2)/4) / 4 + np.exp(-(x**2 + y**2)/6)/1 + (0.2*x+y)/6 + np.exp((((x-1)**2)/2 + ((y+1)**2)/2)/6)/3 - np.exp(-((x-1)**2 + (y+1.0)**2)/10)/1 + np.exp(-((x-1.5)**2 + (y+2.5)**2)/3)/1
    # return (1*np.sin(2*x)+np.cos(4*y))*np.exp(-((x+0.1)**2 + (y+0.2)**2)/3) / 3 + np.exp(-(x**2 + y**2)/7)/1 + (-x+y)/10 + np.exp(((x**2)/2 + (y**2)/3)/12)/2  + np.exp(-((x-3)**2 + (y+2.5)**2)/5)/3

z = f(x, y)
z_clipped = np.clip(z, -2, 10)
z_max = 3.0

# # 勾配の定義
# def grad_f(x, y):
#     df_dx = 3*x**2 - 3*y + x*np.exp((x**2 + y**2)/2.0)
#     df_dy = -3*x + y*np.exp((x**2 + y**2)/2.0)
#     return np.array([df_dx, df_dy])

# 勾配の定義（数値微分）
def grad_f(x, y):
    h=1e-8
    df_dx = (f(x + h, y) - f(x - h, y)) / (2 * h)
    df_dy = (f(x, y + h) - f(x, y - h)) / (2 * h)
    return np.array([df_dx, df_dy])

# # Example usage
# point = (1, 2)
# numerical_partial_derivative(f, point)

# 最急降下法の実装
def gradient_descent(start_point, learning_rate, num_steps):
    path = [start_point]
    current_point = start_point
    for _ in range(num_steps):
        gradient = grad_f(*current_point)
        current_point = current_point - learning_rate * gradient
        path.append(current_point)
    return np.array(path)

# Normalize to [0,1]
norm = plt.Normalize(z_clipped.min(), z_max)
colors = cm.Blues_r(norm(z_clipped))
# colors = cm.jet_r(norm(z_clipped))
rcount, ccount, _ = colors.shape

fig = plt.figure()
ax = fig.gca(projection='3d')

# パラメータの設定
start_point = np.array([-0.466, 0.675+0.01])
learning_rate = 0.01
num_steps = 5000

# 最急降下経路の計算 1
path1 = gradient_descent(start_point, learning_rate, num_steps)

# 最急降下経路の3Dプロット
path_z1 = f(path1[:,0], path1[:,1])+0.002
# ax.plot(path1[:,0], path1[:,1], path_z1, color='r', linestyle='-', linewidth=2)

# パラメータの設定
start_point = np.array([-0.466, 0.675-0.01])
learning_rate = 0.01
num_steps = 5000

# 最急降下経路の計算 2
path2 = gradient_descent(start_point, learning_rate, num_steps)

# 最急降下経路の3Dプロット
path_z2 = f(path2[:,0], path2[:,1])+0.002
# ax.plot(path2[:,0], path2[:,1], path_z2, color='r', linestyle='-', linewidth=2)


# パラメータの設定 3
start_point = np.array([0.475+0.01, -0.63])
learning_rate = 0.01
num_steps = 4800

# 最急降下経路の計算
path3 = gradient_descent(start_point, learning_rate, num_steps)

# 最急降下経路の3Dプロット
path_z3 = f(path3[:,0], path3[:,1])+0.002
# ax.plot(path3[:,0], path3[:,1], path_z3, color='r', linestyle='-', linewidth=2)

# パラメータの設定 4
start_point = np.array([0.475-0.01, -0.63])
learning_rate = 0.01
num_steps = 5000

# 最急降下経路の計算
path4 = gradient_descent(start_point, learning_rate, num_steps)

# 最急降下経路の3Dプロット
path_z4 = f(path4[:,0], path4[:,1])+0.002
# ax.plot(path4[:,0], path4[:,1], path_z4, color='r', linestyle='-', linewidth=2)

# パラメータの設定 5
start_point = np.array([-0.544+0.000, -1.343+0.009])
learning_rate = 0.01
num_steps = 5000

# 最急降下経路の計算
path5 = gradient_descent(start_point, learning_rate, num_steps)

# 最急降下経路の3Dプロット
path_z5 = f(path5[:,0], path5[:,1])+0.002
# ax.plot(path5[:,0], path5[:,1], path_z5, color='r', linestyle='-', linewidth=2)

# パラメータの設定 6
start_point = np.array([-0.544-0.000, -1.343-0.009])
learning_rate = 0.01
num_steps = 3200

# 最急降下経路の計算
path6 = gradient_descent(start_point, learning_rate, num_steps)

# 最急降下経路の3Dプロット
path_z6 = f(path6[:,0], path6[:,1])+0.002
# ax.plot(path6[:,0], path6[:,1], path_z6, color='r', linestyle='-', linewidth=2)

# print(path1[-1,0],path1[-1,1],path_z1[-1],sep="\t")
# print(path2[-1,0],path2[-1,1],path_z2[-1],sep="\t")
# print(path3[-1,0],path3[-1,1],path_z3[-1],sep="\t")
# print(path4[-1,0],path4[-1,1],path_z4[-1],sep="\t")
# print(path5[-1,0],path5[-1,1],path_z5[-1],sep="\t")
# print(path6[-1,0],path6[-1,1],path_z6[-1],sep="\t")

# print(path1[0,0],path1[0,1],path_z1[0],sep="\t")
# print(path2[0,0],path2[0,1],path_z2[0],sep="\t")
# print(path3[0,0],path3[0,1],path_z3[0],sep="\t")
# print(path4[0,0],path4[0,1],path_z4[0],sep="\t")
# print(path5[0,0],path5[0,1],path_z5[0],sep="\t")
# print(path6[0,0],path6[0,1],path_z6[0],sep="\t")

# surf = ax.plot_surface(x, y, z_clipped, rcount=rcount, ccount=ccount, facecolors=colors, shade=False)
surf = ax.plot_surface(x, y, z_clipped, rcount=rcount, ccount=ccount, facecolors=colors, shade=False, zorder=0.5, cmap='Blues_r')
# surf = ax.plot_surface(x, y, z_clipped, rcount=rcount, ccount=ccount, facecolors=colors, shade=False, alpha=0.0, zorder=0.5) # PESを透明化
surf.set_facecolor((0,0,0,0))

# サーフェスプロットに対して色の範囲を設定
surf.set_clim(0.0, z_max)
# ax.set_zlim(z_clipped.min(), z_clipped.max()) # z軸の範囲の設定
ax.set_zlim(top=z_max) # z軸の範囲の設定

# 背景と軸の削除
# ax.set_axis_off()
# カラーバーの追加
# fig.colorbar(surf, shrink=0.9, aspect=10)
ax.view_init(elev=30, azim=230)
ax.set_axis_off()

ax.plot(path1[:,0], path1[:,1], path_z1, color='r', linestyle='-', linewidth=2, zorder=1.5)
ax.plot(path2[:,0], path2[:,1], path_z2, color='r', linestyle='-', linewidth=2, zorder=1.5)
ax.plot(path3[:,0], path3[:,1], path_z3, color='r', linestyle='-', linewidth=2, zorder=1.5)
ax.plot(path4[:,0], path4[:,1], path_z4, color='r', linestyle='-', linewidth=2, zorder=1.5)
ax.plot(path5[:,0], path5[:,1], path_z5, color='r', linestyle='-', linewidth=2, zorder=1.5)
ax.plot(path6[:,0], path6[:,1], path_z6, color='r', linestyle='-', linewidth=2, zorder=1.5)
# ax.scatter(path1[:, 0], path1[:, 1], path_z1, color='r', s=20)

# # 軸の枠を非表示にする
# ax.grid(False)  # グリッドを非表示
# ax.set_xticks([])  # x軸の目盛りを非表示
# ax.set_yticks([])  # y軸の目盛りを非表示
# ax.set_zticks([])  # z軸の目盛りを非表示

# # 軸の枠線を非表示にする
# ax.w_xaxis.line.set_visible(False)
# ax.w_yaxis.line.set_visible(False)
# ax.w_zaxis.line.set_visible(False)

# 背景のパッチを非表示にする
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# # 軸の枠線の色を透明にする（完全に非表示にする）
# ax.xaxis.pane.set_edgecolor('w')
# ax.yaxis.pane.set_edgecolor('w')
# ax.zaxis.pane.set_edgecolor('w')

plt.show()


# 2Dプロットの生成
fig, ax = plt.subplots(figsize=(7, 7))

# 関数の等高線プロット
# contour = ax.contourf(x, y, z_clipped, 50, cmap=cm.Blues_r)
contour = ax.contourf(x, y, z_clipped, 30, cmap=cm.Blues_r, alpha=0.8, vmax=z_max)
# contour = ax.contourf(x, y, z_clipped, 30, cmap=cm.jet_r, alpha=0.8, vmax=z_max)

# 最急降下経路の2Dプロット
ax.plot(path1[:,0], path1[:,1], color='r', linestyle='-', linewidth=2)
# 最急降下経路の2Dプロット
ax.plot(path2[:,0], path2[:,1], color='r', linestyle='-', linewidth=2)
ax.plot(path3[:,0], path3[:,1], color='r', linestyle='-', linewidth=2)
ax.plot(path4[:,0], path4[:,1], color='r', linestyle='-', linewidth=2)
ax.plot(path5[:,0], path5[:,1], color='r', linestyle='-', linewidth=2)
ax.plot(path6[:,0], path6[:,1], color='r', linestyle='-', linewidth=2)
ax.set_axis_off()
plt.show()
