import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import art3d
import numpy as np
from scipy.signal import argrelmin, argrelmax, argrelextrema
import sys

def surface_plotting(f, xc_list, yc_list, zc_list, linecolor, cmapname):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot3D(xc_list, yc_list, zc_list, linewidth=1, color=linecolor)
    ax.scatter(xc_list[-1], yc_list[-1], zc_list[-1], marker=".", color="r", linestyle='None')

    surf_x = np.linspace(-1.1, 0.9, 50)
    surf_y = np.linspace(-0.8, 1.8, 50)
    xmesh, ymesh = np.meshgrid(surf_x, surf_y)
    zmesh = f(xmesh, ymesh)
    X, Y, Z = xmesh, ymesh, zmesh
    wire = ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2)

    # Retrive data from internal storage of plot_wireframe, then delete it
    nx, ny, _  = np.shape(wire._segments3d)
    wire_x = np.array(wire._segments3d)[:, :, 0].ravel()
    wire_y = np.array(wire._segments3d)[:, :, 1].ravel()
    wire_z = np.array(wire._segments3d)[:, :, 2].ravel()
    wire.remove()

    # create data for a LineCollection
    wire_x1 = np.vstack([wire_x, np.roll(wire_x, 1)])
    wire_y1 = np.vstack([wire_y, np.roll(wire_y, 1)])
    wire_z1 = np.vstack([wire_z, np.roll(wire_z, 1)])
    to_delete = np.arange(0, nx*ny, ny)
    wire_x1 = np.delete(wire_x1, to_delete, axis=1)
    wire_y1 = np.delete(wire_y1, to_delete, axis=1)
    wire_z1 = np.delete(wire_z1, to_delete, axis=1)
    scalars = np.delete(wire_z, to_delete)

    segs = [list(zip(xl, yl, zl)) for xl, yl, zl in \
                     zip(wire_x1.T, wire_y1.T, wire_z1.T)]

    my_wire = art3d.Line3DCollection(segs, alpha = 0.5, cmap=cmapname)
    my_wire.set_array(scalars)
    ax.add_collection(my_wire)
    # plt.colorbar(my_wire) # カラーバーを表示する
    # ax.set_xticks([]) # ３Dプロットの軸目盛を消す
    # ax.set_yticks([])
    # ax.set_zticks([])
    # ax.grid(False) # ３Dプロットの背景のグリッドを消す
    # ax.w_xaxis.set_pane_color((0., 0., 0., 0.)) # ３Dプロットの背景色を消す
    # ax.w_yaxis.set_pane_color((0., 0., 0., 0.))
    # ax.w_zaxis.set_pane_color((0., 0., 0., 0.))
    # ax.axis("off") # option
    # ax.axis("off") # 軸と背景全消し
    plt.show()


def surface_plotting2(f, f2, xc_list, yc_list, zc_list, xc_list2, yc_list2, zc_list2, linecolor, linecolor2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot3D(xc_list, yc_list, zc_list, linewidth=1, color=linecolor)
    ax.scatter(xc_list[-1], yc_list[-1], zc_list[-1], marker=".", color="r", linestyle='None')
    surf_x = np.linspace(-1.1, 0.9, 50)
    surf_y = np.linspace(-0.8, 1.8, 50)
    xmesh, ymesh = np.meshgrid(surf_x, surf_y)
    zmesh = f(xmesh, ymesh)
    X, Y, Z = xmesh, ymesh, zmesh
    wire = ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2)

    # Retrive data from internal storage of plot_wireframe, then delete it
    nx, ny, _  = np.shape(wire._segments3d)
    wire_x = np.array(wire._segments3d)[:, :, 0].ravel()
    wire_y = np.array(wire._segments3d)[:, :, 1].ravel()
    wire_z = np.array(wire._segments3d)[:, :, 2].ravel()
    wire.remove()

    # create data for a LineCollection
    wire_x1 = np.vstack([wire_x, np.roll(wire_x, 1)])
    wire_y1 = np.vstack([wire_y, np.roll(wire_y, 1)])
    wire_z1 = np.vstack([wire_z, np.roll(wire_z, 1)])
    to_delete = np.arange(0, nx*ny, ny)
    wire_x1 = np.delete(wire_x1, to_delete, axis=1)
    wire_y1 = np.delete(wire_y1, to_delete, axis=1)
    wire_z1 = np.delete(wire_z1, to_delete, axis=1)
    scalars = np.delete(wire_z, to_delete)

    segs = [list(zip(xl, yl, zl)) for xl, yl, zl in \
                     zip(wire_x1.T, wire_y1.T, wire_z1.T)]

    my_wire = art3d.Line3DCollection(segs, alpha = 0.5, cmap="coolwarm")
    my_wire.set_array(scalars)
    ax.add_collection(my_wire)

    ### 2nd function
    shift_ene = zc_list[-1] - zc_list2[-1] + 20
    ax.plot3D(xc_list2, yc_list2, zc_list2 + shift_ene, linewidth=1, color=linecolor2)
    ax.scatter(xc_list2[-1], yc_list2[-1], zc_list2[-1] + shift_ene, marker=".", color="r", linestyle='None')

    surf_x = np.linspace(-1.1, 0.9, 50)
    surf_y = np.linspace(-0.8, 1.8, 50)
    xmesh, ymesh = np.meshgrid(surf_x, surf_y)
    zmesh = f2(xmesh, ymesh) + shift_ene
    X, Y, Z = xmesh, ymesh, zmesh
    wire = ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2)

    # Retrive data from internal storage of plot_wireframe, then delete it
    nx, ny, _  = np.shape(wire._segments3d)
    wire_x = np.array(wire._segments3d)[:, :, 0].ravel()
    wire_y = np.array(wire._segments3d)[:, :, 1].ravel()
    wire_z = np.array(wire._segments3d)[:, :, 2].ravel()
    wire.remove()

    # create data for a LineCollection
    wire_x1 = np.vstack([wire_x, np.roll(wire_x, 1)])
    wire_y1 = np.vstack([wire_y, np.roll(wire_y, 1)])
    wire_z1 = np.vstack([wire_z, np.roll(wire_z, 1)])
    to_delete = np.arange(0, nx*ny, ny)
    wire_x1 = np.delete(wire_x1, to_delete, axis=1)
    wire_y1 = np.delete(wire_y1, to_delete, axis=1)
    wire_z1 = np.delete(wire_z1, to_delete, axis=1)
    scalars = np.delete(wire_z, to_delete)

    segs = [list(zip(xl, yl, zl)) for xl, yl, zl in \
                     zip(wire_x1.T, wire_y1.T, wire_z1.T)]

    my_wire = art3d.Line3DCollection(segs, alpha = 0.2, cmap="plasma")
    my_wire.set_array(scalars)
    ax.add_collection(my_wire)
    # plt.colorbar(my_wire) # カラーバーを表示する
    # ax.set_xticks([]) # ３Dプロットの軸目盛を消す
    # ax.set_yticks([])
    # ax.set_zticks([])
    # ax.grid(False) # ３Dプロットの背景のグリッドを消す
    # ax.w_xaxis.set_pane_color((0., 0., 0., 0.)) # ３Dプロットの背景色を消す
    # ax.w_yaxis.set_pane_color((0., 0., 0., 0.))
    # ax.w_zaxis.set_pane_color((0., 0., 0., 0.))
    # ax.axis("off") # 軸と背景全消し
    plt.show()

def fx(f, x, y):
    h = 0.0000001
    return (f(x+h, y)-f(x-h, y))/(2*h)

def fy(f, x, y):
    h = 0.0000001
    return (f(x, y+h)-f(x, y-h))/(2*h)

def fxx(f, x, y):
    h = 0.0000001
    return (fx(f, x+h, y)-fx(f, x-h, y))/(2*h)

def fxy(f, x, y):
    h = 0.0000001
    return (fx(f, x, y+h)-fx(f, x, y-h))/(2*h)

def fyy(f, x, y):
    h = 0.0000001
    return (fy(f, x, y+h)-fy(f, x, y-h))/(2*h)

def pes(x, y):
    # Müller & Brown potential, https://doi.org/10.1007/BF00547608
    A=[-200,-100,-170,15]
    a=[-1,-1,-6.5,0.7]
    b=[0,0,11,0.6]
    c=[-10,-10,-6.5,0.7]
    p=[1,0,-0.5,-1]
    q=[0,0.5,1.5,1]
    s = []
    for i in range(4):
        s.append(A[i]*np.exp(a[i]*(x-p[i])**2+b[i]*(x-p[i])*(y-q[i])+c[i]*(y-q[i])**2))
    return sum(s)

# def pes(x, y):
#     # NB potential 1
#     A=[0.1, 0.1, 0.1, 0.001]
#     a=[np.sin(x-2) + np.cos(y), np.sin(x-y) - np.cos(y), np.sin(y) + 3 * np.cos(x+y), x**2 + y**2 /6]
#     s = []
#     for i in range(4):
#         s.append(A[i]*np.exp(a[i]))
#     return sum(s)

def addint(x, y):  # 人工力を加える操作（なんちゃってAFIR関数）
    a = 150.0   # for MB potential
    b = -150.0
    # a = -0.6  # for NB potential 1
    # b = -0.4
    return pes(x, y) + a*x + b*y  # x, y方向に定比例する項を付け加える


##############################################################
#                       input section                        #
##############################################################
(xc, yc) = ( 0.623 , 0.028 )
xc_list = []; yc_list = []
ene_list = []; original_ene_list = []; addint_ene_list = []
addint_xc_list = []; addint_yc_list = []
stepsize = 0.0001
maxtimes = 10000

# MBポテンシャル曲面 (トラジェクトリ無し)
# surf_x = np.linspace(-3.5, 3.5, 300)
# surf_y = np.linspace(-7.5, 7.5, 300)
# xmesh, ymesh = np.meshgrid(surf_x, surf_y)
# z = addint(xmesh, ymesh)
# level = []
# for i in range(0,100):
#     level.append(np.min(z) + 0.1*i)
# cont = plt.contourf(surf_x, surf_y, z, levels=level, cmap='coolwarm')
# plt.colorbar()
# plt.show()


##############################################################
#                         最急降下法                          #
##############################################################

# steepest descent method 最急降下法 on 人工力を加えたPES
for i in range(1, maxtimes):
    times = i
    diffx1 = fx(addint, xc, yc)
    diffy1 = fy(addint, xc, yc)
    if np.sqrt(diffx1 ** 2 + diffy1 ** 2) < 1e-10:
        addint_xc_list.append(xc); addint_yc_list.append(yc)  # 座標をリストに追加
        original_ene_list.append(pes(xc, yc))  # AFIR経路に沿ったエネルギー値をリストに追加
        addint_ene_list.append(addint(xc, yc))  # AFIR経路に沿ったエネルギー値をリストに追加 (AFIR関数追加)
        break
    else:
        addint_xc_list.append(xc); addint_yc_list.append(yc)  # 座標をリストに追加
        original_ene_list.append(pes(xc, yc))  # AFIR経路に沿ったエネルギー値をリストに追加
        addint_ene_list.append(addint(xc, yc))  # AFIR経路に沿ったエネルギー値をリストに追加 (AFIR関数追加)
        xc = xc - stepsize * diffx1   # 次のx座標を生成
        yc = yc - stepsize * diffy1   # 次のy座標を生成

print("Optimization on modified PES finished!\n( itr =", times, ")\n(", xc, ",", yc,")")
print("Energy (a.f.) =", addint(xc, yc))
print("Grad (a.f.)   =", np.sqrt(fx(addint, xc, yc) ** 2 + fy(addint, xc, yc) ** 2))
print("Energy (bare) =", pes(xc, yc))
print("Grad (bare)   =", np.sqrt(fx(pes, xc, yc) ** 2 + fy(pes, xc, yc) ** 2))

# 人工力を加えたPES
plt.plot(addint_xc_list, addint_yc_list, 'y.-', alpha=0.2)            # trajectory
plt.plot(addint_xc_list[0], addint_yc_list[0], 'b.-', alpha=0.2)      # initial point
plt.plot(addint_xc_list[-1], addint_yc_list[-1], 'r.-', alpha=0.2)    # terminal point

# surf_x = np.linspace(-3.5, 3.5, 300)
# surf_y = np.linspace(-7.5, 7.5, 300)
surf_x = np.linspace(-3.0, 1.5, 300)
surf_y = np.linspace(-1.0, 3.5, 300)
xmesh, ymesh = np.meshgrid(surf_x, surf_y)
z = addint(xmesh, ymesh)
# level = []
# for i in range(0,100):
#     level.append(np.min(z) + (10-np.min(z))*0.01*i)
level = []
for i in range(0,25):
    level.append(np.min(z) + (100-np.min(z))*0.04*i)
cont = plt.contourf(surf_x, surf_y, z, levels=level, cmap='coolwarm')
plt.colorbar()
plt.show()


# AFIR経路上のピークを検出する
local_max_id = argrelmax(np.array(original_ene_list), order=10)
local_min_id = argrelmin(np.array(original_ene_list), order=10)
# order: How many points on each side to use for the comparison to consider;
# この値を大きくすることでノイズを回避する
local_max_id_list = local_max_id[0]
local_min_id_list = local_min_id[0]
# print(maxid[0])
print("-----------------\nThe # of App. EQ found = ", len(local_max_id_list))
for i in range(len(local_min_id_list)):
    print("App. EQ", i, "(", addint_xc_list[local_min_id_list[i]], ",", addint_yc_list[local_min_id_list[i]], ")")
    print("Energy (bare) =", pes(addint_xc_list[local_min_id_list[i]], addint_yc_list[local_min_id_list[i]]))

print("-----------------\nThe # of App. TS found = ", len(local_max_id_list))
for i in range(len(local_max_id_list)):
    print("App. TS", i, "(", addint_xc_list[local_max_id_list[i]], ",", addint_yc_list[local_max_id_list[i]], ")")
    print("Energy (bare) =", pes(addint_xc_list[local_max_id_list[i]], addint_yc_list[local_max_id_list[i]]))

# print(len(ene_list))
plt.plot([i for i in range(len(original_ene_list))], original_ene_list)
plt.plot([i for i in range(len(addint_ene_list))], addint_ene_list)
for i in range(len(local_min_id_list)):   # App. EQ point(s)
    plt.plot(local_min_id_list[i], original_ene_list[local_min_id_list[i]], 'rx')
    plt.plot(local_min_id_list[i], addint_ene_list[local_min_id_list[i]], 'rx')
for i in range(len(local_max_id_list)):   # App. TS point(s)
    plt.plot(local_max_id_list[i], original_ene_list[local_max_id_list[i]], 'kx')
    plt.plot(local_max_id_list[i], addint_ene_list[local_max_id_list[i]], 'kx')
plt.show()

# steepest descent method 最急降下法 on オリジナルのPES
# (xc, yc) = ( 0.0 , 0.0 )
xc_list.append(xc); yc_list.append(yc)  # 人工力を加えたPES上での終点をリストに追加
for i in range(1, maxtimes):
    times = i
    diffx1 = fx(pes, xc, yc)
    diffy1 = fy(pes, xc, yc)
    if np.sqrt(diffx1 ** 2 + diffy1 ** 2) < 1e-10:
        xc_list.append(xc); yc_list.append(yc)  # 座標をリストに追加
        break
    else:
        xc_list.append(xc); yc_list.append(yc)  # 座標をリストに追加
        xc = xc - stepsize * diffx1   # 次のx座標を生成
        yc = yc - stepsize * diffy1   # 次のy座標を生成

print("-----------------\nOptimization on original PES finished!\n( itr =", times, ")\n(", xc, ",", yc,")")
print("Energy (bare) =", pes(xc, yc))
print("Grad (bare)   =", np.sqrt(fx(pes, xc, yc) ** 2 + fy(pes, xc, yc) ** 2))

# 元のPES
plt.plot(addint_xc_list, addint_yc_list, 'y.-', alpha=0.1)            # trajectory
plt.plot(addint_xc_list[0], addint_yc_list[0], 'b.-', alpha=0.2)      # initial point
plt.plot(addint_xc_list[-1], addint_yc_list[-1], 'r.-', alpha=0.2)    # terminal point
for i in range(len(local_min_id_list)):   # App. TS point(s)
    plt.plot(addint_xc_list[local_min_id_list[i]], addint_yc_list[local_min_id_list[i]], 'rx', alpha=0.7)
for i in range(len(local_max_id_list)):   # App. TS point(s)
    plt.plot(addint_xc_list[local_max_id_list[i]], addint_yc_list[local_max_id_list[i]], 'kx', alpha=0.7)
plt.plot(xc_list, yc_list, 'g.-', alpha=0.3)            # trajectory
plt.plot(xc_list[0], yc_list[0], 'b.-', alpha=0.8)      # initial point
plt.plot(xc_list[-1], yc_list[-1], 'r.-', alpha=0.5)    # terminal point
# surf_x = np.linspace(-3.5, 3.5, 300)
# surf_y = np.linspace(-7.5, 7.5, 300)
surf_x = np.linspace(-3.0, 1.5, 300)
surf_y = np.linspace(-1.0, 3.5, 300)
xmesh, ymesh = np.meshgrid(surf_x, surf_y)
z = pes(xmesh, ymesh)
# level = []
# for i in range(0,100):
#     level.append(np.min(z) + (10-np.min(z))*0.01*i)
level = []
for i in range(-15,10):
    level.append(10.0 * i)
cont = plt.contourf(surf_x, surf_y, z, levels=level, cmap='coolwarm')
plt.colorbar()
plt.show()

surface_plotting(pes, addint_xc_list, addint_yc_list, original_ene_list, 'g', 'coolwarm')

surface_plotting(addint, addint_xc_list, addint_yc_list, addint_ene_list, 'r', 'plasma')

surface_plotting2(pes, addint, addint_xc_list, addint_yc_list, original_ene_list, addint_xc_list, addint_yc_list, addint_ene_list, 'g', 'r')

# print("sa")

# sys.exit()

##############################################################
#                        ニュートン法                         #
##############################################################

# print(len(local_max_id_list))

for h in range(len(local_max_id_list)):
    ### 初期設定
    (xc, yc) = ( addint_xc_list[local_max_id_list[h]] , addint_yc_list[local_max_id_list[h]] )
    xc_list = []; yc_list = []  # 座標を格納するリストを用意する
    stepsize = 1.0   # STEP幅
    maxitr = 10000  # 座標更新サイクルの上限回数

    ### ニュートン法による停留点の探索
    for i in range(1, maxitr):
        diffx = fx(pes, xc, yc)
        diffy = fy(pes, xc, yc)
        iteration = i       # 更新回数を保存
        if np.sqrt(diffx ** 2 + diffy ** 2) < 1e-10:  # gradientの大きさが10^(-10)未満のとき
            xc_list.append(xc)  # 座標をリストに追加
            yc_list.append(yc)
            break
        else:
            xc_list.append(xc)  # 座標をリストに追加
            yc_list.append(yc)

            det = fxx(pes, xc, yc) * fyy(pes, xc, yc) - fxy(pes, xc, yc) ** 2                         # ヘシアンの行列式
            x_element = (fx(pes, xc, yc) * fyy(pes, xc, yc) - fy(pes, xc, yc) * fxy(pes, xc, yc)) / det    # ヘシアンの逆行列とグラジエントの積（x成分）
            y_element = (- fx(pes, xc, yc) * fxy(pes, xc, yc) + fxx(pes, xc, yc) * fy(pes, xc, yc)) / det  # ヘシアンの逆行列とグラジエントの積（y成分）
            xc = xc - stepsize * x_element  # 次の座標を生成･更新
            yc = yc - stepsize * y_element

    ### ニュートン法による最適化の結果を出力
    print("Terminal point = (", xc, ",", yc,") , Iteration =", iteration)
    print("Energy =", pes(xc, yc), ", Gradient =", np.sqrt(diffx ** 2 + diffy ** 2))

    ### 初期点からニュートン法で辿った経路の図示
    plt.plot(xc_list, yc_list, 'g.-', alpha=0.3)            # trajectory
    plt.plot(xc_list[0], yc_list[0], 'b.-', alpha=0.8)      # initial point
    plt.plot(xc_list[-1], yc_list[-1], 'r.-', alpha=0.5)    # terminal point

    # surf_x = np.linspace(-3.5, 3.5, 300)
    # surf_y = np.linspace(-7.5, 7.5, 300)
    surf_x = np.linspace(-3.0, 1.5, 300)
    surf_y = np.linspace(-1.0, 3.5, 300)
    xmesh, ymesh = np.meshgrid(surf_x, surf_y)
    z = pes(xmesh, ymesh)
    # level = []
    # for i in range(0,100):
    #     level.append(np.min(z) + 0.1*i)
    level = []
    for i in range(-15,10):
        level.append(10.0 * i)
    cont = plt.contourf(surf_x, surf_y, z, levels=level, cmap='coolwarm')
    plt.colorbar()
    plt.show()


### stationary points on MB potential
# ( 0.623499404925542 , 0.02803775852509033 , -108.16672411685236 )
# ( -0.5582236344831244 , 1.4417258419458847 , -146.69951720995402 )
# ( -0.050010823607759436 , 0.46669410492931074 , -80.76781812965903 )
# TS ( 0.2124865821449053 , 0.2929883250861434 , -72.24894011232522 )
# TS ( -0.8220015586868213 , 0.6243128028157512 , -40.6648435086574 )

### stationary points on NB potential 1
# ( -0.5603915290035836 , 4.971004915794321 )
# ( 0.6549871734236349 , 2.0767806008410785 )
# ( -0.8455223807123371 , -1.0928376521695613 )
# ( 0.536057764437937 , -4.03291810258038 )
# ( -0.6518785121541462 , -6.347274786210717 )
# TS ( 0.19751229061430708 , 4.515137953507503 )
# TS ( -2.2588813539053993 , 2.1285074825588763 )
# TS ( 1.6153116009676123 , -1.6000995843171115 )
# TS ( 0.2678579034943539 , -1.7699957607379708 )
# TS ( 0.07939567396724026 , -6.338899598796261 )
# 2nd order saddle ( -1.6799011303506017 , 1.658383831987869 )
