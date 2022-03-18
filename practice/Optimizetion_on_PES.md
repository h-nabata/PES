# Optimizetion on Potential Energy Surface
> *produced by H. Nabata*
> 
> *- 2022/03/18 uploaded*


## What is potential energy surface (PES)?


> Atoms in a molecule are held together by chemical bonds. When the atom is distorted, the bonds are stretched or compressed, in which increases the potential energy of its system. As the new geometry is formed, the molecule stays stationary. Therefore, the energy of the system is not caused by the kinetic energy, but depending on the position of the atoms (potential). Energy of a molecule is a function of the position of the nuclei. When nuclei moves, electron readjusts quickly. The relationship between this molecular energy and molecular geometry (position) is mapped out with potential energy surface.
> (refer to [this page](https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Supplemental_Modules_(Physical_and_Theoretical_Chemistry)/Quantum_Mechanics/11%3A_Molecules/Potential_Energy_Surface))


## Purpose of this text
* Implement the steepest descent method to find a minimum on PES from an arbitrary initial point.
* Implement Newton's method to find a saddle point from an initial point near the saddle point.


In this text, we adopt the Müller-Brown potential as a sample PES. The Müller-Brown potential is one of the simple 2D potentials proposed by K. Miiller and L. D. Brown in [their paper in 1979](https://link.springer.com/content/pdf/10.1007/BF00547608.pdf).
The Müller-Brown potential is given as below.

![\begin{align*}
\sum_{i} A_{i} \exp \left[a_{i}\left(x-p_{i}\right)^{2}+b_{i}\left(x-p_{i}\right)\left(y-q_{i}\right)+c_{i}\left(y-q_{i}\right)^{2}\right]
\end{align*}](https://render.githubusercontent.com/render/math?math=%5Ccolor%7Bblack%7D%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0A%5Csum_%7Bi%7D+A_%7Bi%7D+%5Cexp+%5Cleft%5Ba_%7Bi%7D%5Cleft%28x-p_%7Bi%7D%5Cright%29%5E%7B2%7D%2Bb_%7Bi%7D%5Cleft%28x-p_%7Bi%7D%5Cright%29%5Cleft%28y-q_%7Bi%7D%5Cright%29%2Bc_%7Bi%7D%5Cleft%28y-q_%7Bi%7D%5Cright%29%5E%7B2%7D%5Cright%5D%0A%5Cend%7Balign%2A%7D)

where (A) = (-200/-100/-170/15), (a) = (-1/-1/-6.5/0.7), (b) = (0/0/11/0.6), (c) = (-10/-10/-6.5/0.7), (p) =(1/0/-0.5/-1), (q) = (0/0.5/1.5/1).


The 2D image of this potential is shown below.
<div align="center">
  
![MDpot](https://github.com/h-nabata/image_storage/blob/fa44d488018f68358cabe15ffc16881bb0e061d7/MBpot1.svg "Müller-Brown potential")

</div>

We use ***[Python](https://www.python.org/)*** to implement the optimization with gradient methods. Python is a high-level and versatile programming language, and Python’s ecosystem provides a rich set of frameworks, tools, and libraries that allow you to write almost any kind of application.

One of the advantages of writing programs in Python is the availability of many open source libraries such as Numpy and Matplotlib. This means that optimization methods can be implemented in very simple steps.

## Implementation

This program needs to import two libraries, Numpy and Matplotlib.

```py
### importing libraries, ライブラリのインポート
import matplotlib.pyplot as plt
import numpy as np
```

The Müller-Brown potential is defined as below.

```py
### Müller-Brownポテンシャルの定義
def f(x, y):
    # Müller & Brown potential, https://doi.org/10.1007/BF00547608
    A=[-200.0, -100.0, -170.0, 15.0]
    a=[-1.0, -1.0, -6.5, 0.7]
    b=[0.0, 0.0, 11.0, 0.6]
    c=[-10.0, -10.0, -6.5, 0.7]
    p=[1.0, 0.0, -0.5, -1.0]
    q=[0.0, 0.5, 1.5, 1.0]
    s = []
    for i in range(4):
        s.append(A[i]*np.exp(a[i]*(x-p[i])**2+b[i]*(x-p[i])*(y-q[i])+c[i]*(y-q[i])**2))
    return sum(s)
```

Definition of partial derivatives and those of the second order by numerical differentiation.

```py
### 数値微分による偏導関数の定義
def fx(x, y):
    h = 1e-7
    return (f(x+h, y)-f(x-h, y))/(2*h)

def fy(x, y):
    h = 1e-7
    return (f(x, y+h)-f(x, y-h))/(2*h)
```

```py
### 数値微分による2階の偏導関数の定義
def fxx(x, y):
    h = 1e-7
    return (fx(x+h, y)-fx(x-h, y))/(2*h)

def fxy(x, y):
    h = 1e-7
    return (fx(x, y+h)-fx(x, y-h))/(2*h)

def fyy(x, y):
    h = 1e-7
    return (fy(x, y+h)-fy(x, y-h))/(2*h)
```

Now, let's implement the steepest descent method. First, the initial coordinates must be determined. In general PES, this point corresponds to the initial structure of the molecule(s). The appropriate step size for this case is 1e-4 (=0.0001), and the maximum number of iterations for updating coordinates should be set to 10000.

```py
### 初期設定
xc = -0.8; yc = 0.75      # 初期座標（＝初期構造）
xc_list = []; yc_list = []  # 座標を格納するリストを用意する
stepsize = 1e-4   # STEP幅
maxitr = 10000  # 座標更新サイクルの上限回数
```

The steepest descent method requires the gradient vector at each point on the PES to calculate the direction of descent. The termination condition for updating coordinates is that the magnitude of the gradient falls below an appropriate threshold.

```py
### 最急降下法による極小点の探索
for i in range(1, maxitr):  # 上限回数まで座標更新を続ける
    diffx = fx(xc, yc)  # gradient（勾配）の計算
    diffy = fy(xc, yc)
    iteration = i       # 更新回数を保存
    if np.sqrt(diffx ** 2 + diffy ** 2) < 1e-10:  # gradientの大きさが10^(-10)未満のとき
        xc_list.append(xc)  # 座標をリストに追加
        yc_list.append(yc)
        break              # loopをbreakする
    else:                                         # gradientの大きさが10^(-10)以上のとき
        xc_list.append(xc)  # 座標をリストに追加
        yc_list.append(yc)
        xc = xc - stepsize * diffx  # 次の座標を生成･更新
        yc = yc - stepsize * diffy
```

Then, print out the output of the program.

```py
### 最急降下法による最適化の結果を出力
print("Terminal point = (", xc, ",", yc,") , Iteration =", iteration)
print("Energy =", f(xc, yc), ", Gradient =", np.sqrt(diffx ** 2 + diffy ** 2))
```

> Terminal point = ( -0.5582236349238372 , 1.4417258414809169 ) , Iteration = 532
> 
> Energy = -146.69951720995402 , Gradient = 0.0

Let's visualize the trajectory of descending course from the initial point.

```py
### 初期点からの最急降下経路の図示
plt.plot(xc_list, yc_list, 'g.-', alpha=0.3)            # trajectory
plt.plot(xc_list[0], yc_list[0], 'b.-', alpha=0.8)      # initial point
plt.plot(xc_list[-1], yc_list[-1], 'r.-', alpha=0.5)    # terminal point

# z軸のlevelを定義
level = []
for i in range(-8,20):
    level.append(20*i)

# ポテンシャル面の描画
surf_x = np.linspace(-2.0, 1.5, 300)
surf_y = np.linspace(-1.0, 2.5, 300)
xmesh, ymesh = np.meshgrid(surf_x, surf_y)
surf_z = f(xmesh, ymesh)
cont = plt.contourf(surf_x, surf_y, surf_z, levels=level, cmap='coolwarm')
plt.show()
```

The 2D image of the trajectory is shown below.
<div align="center">
  
![MDpot](https://github.com/h-nabata/image_storage/blob/518beb88011362d257143411b2f4c748eb9c9412/MBpot2.svg "trajectory with the steepest descent method")

</div>

* * *

Next, let's implement Newton's method. The concept of optimizetion by using Newton's method is explained in [this wiki page "Newton's method in optimization"](https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization).

Firstly, you should determine the initial coordinates. In Newton's method, coordinates are update by using the gradient vector and the hessian matrix, so it does not require step size.

```py
### 初期設定
xc = -0.8; yc = 0.75      # 初期座標（＝初期構造）
xc_list = []; yc_list = []  # 座標を格納するリストを用意する
maxitr = 10000  # 座標更新サイクルの上限回数
```

```py
### ニュートン法による停留点の探索
for i in range(1, maxitr):
    diffx = fx(xc, yc)
    diffy = fy(xc, yc)
    if np.sqrt(diffx ** 2 + diffy ** 2) < 1e-10:  # gradientの大きさが10^(-10)未満のとき
        xc_list.append(xc)  # 座標をリストに追加
        yc_list.append(yc)
        times = i
        break
    else:        
        xc_list.append(xc)  # 座標をリストに追加
        yc_list.append(yc)
        
        det = fxx(xc, yc) * fyy(xc, yc) - fxy(xc, yc) ** 2                         # ヘシアンの行列式
        x_element = (fx(xc, yc) * fyy(xc, yc) - fy(xc, yc) * fxy(xc, yc)) / det    # ヘシアンの逆行列とグラジエントの積（x成分）
        y_element = (- fx(xc, yc) * fxy(xc, yc) + fxx(xc, yc) * fy(xc, yc)) / det  # ヘシアンの逆行列とグラジエントの積（y成分）
        xc = xc - stepsize * x_element  # 次の座標を生成･更新
        yc = yc - stepsize * y_element
```

Then, print out the output.

```py
### ニュートン法による最適化の結果を出力
print("Terminal point = (", xc, ",", yc,") , Iteration =", iteration)
print("Energy =", f(xc, yc), ", Gradient =", np.sqrt(diffx ** 2 + diffy ** 2))
```

> Terminal point = ( -0.8220015587830566 , 0.6243128028186382 ) , Iteration = 532
> 
> Energy = -40.664843508657405 , Gradient = 0.0

```py
### 初期点からニュートン法で辿った経路の図示
plt.plot(xc_list, yc_list, 'g.-', alpha=0.3)            # trajectory
plt.plot(xc_list[0], yc_list[0], 'b.-', alpha=0.8)      # initial point
plt.plot(xc_list[-1], yc_list[-1], 'r.-', alpha=0.5)    # terminal point

# z軸のlevelを定義
level = []
for i in range(-15,10):
    level.append(10.0 * i)

# ポテンシャル面の描画
surf_x = np.linspace(-2.0, 1.5, 300)
surf_y = np.linspace(-1.0, 2.5, 300)
xmesh, ymesh = np.meshgrid(surf_x, surf_y)
surf_z = f(xmesh, ymesh)
cont = plt.contourf(surf_x, surf_y, surf_z, levels=level, cmap='coolwarm')
plt.show()
```

The 2D image of the trajectory is shown below.
<div align="center">
  
![MDpot](https://github.com/h-nabata/image_storage/blob/daf15bc091000caff5d523c328c818d8df5f6887/MBpot3.svg "trajectory with Newton's method")

</div>

* * *

```
### 以上、最急降下法とニュートン法を実装した。
### 
### ここで、同じ初期点でも到達する点が異なっていることに注意しよう。
### 最急降下法では極小点、ニュートン法では鞍点が得られている。
### 
### また、ニュートン法の収束性は初期値やステップ幅などのパラメータに大きく依存する。
### 各自でパラメータを変えて実行してみて欲しい。
### 
### ニュートン法はあくまで停留点を見つけ出すアルゴリズムであり、必ずしも極小点に到達するとは限らない。
### これを改善した手法として「準ニュートン法」がある。準ニュートン法ではセカント条件を満たすような近似ヘシアンが用いられる。
### これにより、勾配法のように適切な降下方向を選択しつつ、極小点付近ではニュートン法に匹敵する収束性を発揮する。

### 【課題】
### ① 初期点を変えることですべての極小点と鞍点を特定する。
### ② 鞍点において固有値と固有ベクトルを求めて虚の振動方向を特定する。
### ③ ②で求めた虚の振動方向に対して、最急降下法を用いて得られるIRC経路を図示する。
### ④ 以下の式で定義されるポテンシャル面についても同様に解析してみよう。

# def f(x, y):
#     # nabata potential 1
#     A=[0.1, 0.1, 0.1, 0.001]
#     a=[np.sin(x-2) + np.cos(y), np.sin(x-y) - np.cos(y), np.sin(y) + 3 * np.cos(x+y), x**2 + y**2 /6]
#     s = []
#     for i in range(4):
#         s.append(A[i]*np.exp(a[i]))
#     return sum(s)

```

* * *

## Acknowledge
* https://tex-image-link-generator.herokuapp.com/
