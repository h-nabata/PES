# Optimizetion on Potential Energy Surface
> *created by H. Nabata*
> 
> *- 2022/03/18 Uploaded*
> 
> *- 2022/03/20 Chapter 2 was added 2*

<div align="center">
  
●　　　●　　　●

</div>

## What is potential energy surface (PES)?

> Atoms in a molecule are held together by chemical bonds. When the atom is distorted, the bonds are stretched or compressed, in which increases the potential energy of its system. As the new geometry is formed, the molecule stays stationary. Therefore, the energy of the system is not caused by the kinetic energy, but depending on the position of the atoms (potential). Energy of a molecule is a function of the position of the nuclei. When nuclei moves, electron readjusts quickly. The relationship between this molecular energy and molecular geometry (position) is mapped out with potential energy surface.
> (refer to [this page](https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Supplemental_Modules_(Physical_and_Theoretical_Chemistry)/Quantum_Mechanics/11%3A_Molecules/Potential_Energy_Surface))

<div align="center">
  
●　　　●　　　●

</div>

化合物の構造は立体的、つまり3次元的な構造であり、各原子の位置（座標）を決めれば化合物の構造が一意に定まります。化合物の持っているエネルギーの値は各原子の3次元座標によって変化し、一種の多変数関数として扱うことができます。これは化合物の構造とエネルギーとが1対1で対応しているためです。

> ここでは断熱近似を暗に仮定しており、原子核の運動に対して即座に電子状態が追随するものと見なしています。
>> 「励起状態を考えると1価関数でないのでは？」と疑問に思われた方は鋭いですね。確かに、より高いエネルギー準位にある電子状態まで含めると、エネルギーは3次元構造に対して多価な関数と捉えられますが、ここでは基底状態にのみ注目し、励起状態の話を一旦置いておきましょう。同様にスピン状態の話も脇に置いておきます。

N原子分子のエネルギー（ポテンシャル）の関数は、分子の並進と回転の次元を除いた「3N-6次元の曲面」を成すものと考えます。このような多次元（一般には3次元以上）の曲面は「超曲面」と呼ばれています。「構造最適化」の計算はこの「超曲面」（「ポテンシャルエネルギー超曲面」とも言います）の上にある極小点を探す数学的操作に対応します。*なかなか頭の中では想像できませんが…。*

化学反応はポテンシャルエネルギー曲面（PES）に基づいて理論的に調べることができます。いきなり実際の分子を相手にして多次元の空間を考えるのは難しいので、練習として2次元のポテンシャルを考えることにしましょう。

<div align="center">
  
●　　　●　　　●

</div>

# Chapter 1

2次元ポテンシャルの場合、停留点には極小点、極大点、鞍点の3種類が存在し、鞍点には遷移状態に相当する「1次の鞍点」と、["Monkey saddle"](https://en.wikipedia.org/wiki/Monkey_saddle) と呼ばれる「2次の鞍点」が存在します。PES上に存在するこれらの点を特定することは化学反応を理論的に議論する上で重要です。

今回は理論化学の世界で良く知られている2次元ポテンシャルである ["Müller-Brown potential"]((https://link.springer.com/content/pdf/10.1007/BF00547608.pdf))（ミューラー･ブラウン ポテンシャル）を題材として、最急降下法とニュートン法によって停留点を探索するプログラムを書いてみます。このMüller-Brown potentialは3つの極小点（minima）、2つの1次の鞍点が（saddles）を有する多峰性のポテンシャル面です。

ここではPythonを使用して、勾配法による最適化を実装することにします。Pythonは汎用性の高いプログラミング言語であり、様々なライブラリを利用できます。これにより、最適化手法を簡単な手順で実装することができます。

<!-- * Implement the steepest descent method to find a minimum on PES from an arbitrary initial point.
* Implement Newton's method to find a saddle point from an initial point near the saddle point.


In this text, we adopt the Müller-Brown potential as a sample PES. The Müller-Brown potential is one of the simple 2D potentials proposed by K. Miiller and L. D. Brown in [their paper in 1979](https://link.springer.com/content/pdf/10.1007/BF00547608.pdf).
The Müller-Brown potential is given as below. -->

<div align="center">
  
![\begin{align*}
\sum_{i} A_{i} \exp \left[a_{i}\left(x-p_{i}\right)^{2}+b_{i}\left(x-p_{i}\right)\left(y-q_{i}\right)+c_{i}\left(y-q_{i}\right)^{2}\right]
\end{align*}](https://render.githubusercontent.com/render/math?math=%5Ccolor%7Bblack%7D%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0A%5Csum_%7Bi%7D+A_%7Bi%7D+%5Cexp+%5Cleft%5Ba_%7Bi%7D%5Cleft%28x-p_%7Bi%7D%5Cright%29%5E%7B2%7D%2Bb_%7Bi%7D%5Cleft%28x-p_%7Bi%7D%5Cright%29%5Cleft%28y-q_%7Bi%7D%5Cright%29%2Bc_%7Bi%7D%5Cleft%28y-q_%7Bi%7D%5Cright%29%5E%7B2%7D%5Cright%5D%0A%5Cend%7Balign%2A%7D)
  
</div>

where (A) = (-200/-100/-170/15), (a) = (-1/-1/-6.5/0.7), (b) = (0/0/11/0.6), (c) = (-10/-10/-6.5/0.7), (p) =(1/0/-0.5/-1), (q) = (0/0.5/1.5/1).

このポテンシャルの2D画像を以下に示します。
<!-- The 2D image of this potential is shown below. -->
<div align="center">
  
![MDpot](https://github.com/h-nabata/image_storage/blob/fa44d488018f68358cabe15ffc16881bb0e061d7/MBpot1.svg "Müller-Brown potential")

</div>

<!-- We use ***[Python](https://www.python.org/)*** to implement the optimization with gradient methods. Python is a high-level and versatile programming language, and Python’s ecosystem provides a rich set of frameworks, tools, and libraries that allow you to write almost any kind of application.

One of the advantages of writing programs in Python is the availability of many open source libraries such as Numpy and Matplotlib. This means that optimization methods can be implemented in very simple steps. -->

<div align="center">
  
●　　　●　　　●

</div>

## Implementation (the steepest descent method)

このプログラムでは、NumpyとMatplotlibという2つのライブラリをインポートする必要があります。
<!-- This program needs to import two libraries, Numpy and Matplotlib. -->

```py
### importing libraries, ライブラリのインポート
import matplotlib.pyplot as plt
import numpy as np
```

ミュラー・ブラウン・ポテンシャルは以下のように定義されます。
<!-- The Müller-Brown potential is defined as below. -->

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

数値微分による1次の偏微分と2次の偏微分を定義します。
<!-- Definition of partial derivatives and those of the second order by numerical differentiation. -->

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

では、最急降下法を実装してみましょう。まず、初期座標を決定する必要があります。一般的なPESでは、この点が分子の初期構造に相当しています。今回の最適化計算における適切なステップ幅は1e-4 (=0.0001)で、座標更新の上限回数は10000回に設定します。
<!-- Now, let's implement the steepest descent method. First, the initial coordinates must be determined. In general PES, this point corresponds to the initial structure of the molecule(s). The appropriate step size for this case is 1e-4 (=0.0001), and the maximum number of iterations for updating coordinates should be set to 10000. -->

```py
### 初期設定
xc = -0.8; yc = 0.75      # 初期座標（＝初期構造）
xc_list = []; yc_list = []  # 座標を格納するリストを用意する
stepsize = 1e-4   # STEP幅
maxitr = 10000  # 座標更新サイクルの上限回数
```

最急降下法では降下方向を計算するためにPES上の各点における勾配ベクトルを計算する必要があります。座標の更新の終了条件は「勾配の大きさが適切な閾値を下回ること」とします。
<!-- The steepest descent method requires the gradient vector at each point on the PES to calculate the direction of descent. The termination condition for updating coordinates is that the magnitude of the gradient falls below an appropriate threshold. -->

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

プログラムの実行結果を出力します。
<!-- Then, print out the output of the program. -->

```py
### 最急降下法による最適化の結果を出力
print("Terminal point = (", xc, ",", yc,") , Iteration =", iteration)
print("Energy =", f(xc, yc), ", Gradient =", np.sqrt(diffx ** 2 + diffy ** 2))
```

> Terminal point = ( -0.5582236349238372 , 1.4417258414809169 ) , Iteration = 532
> 
> Energy = -146.69951720995402 , Gradient = 0.0

初期点から下降していく軌跡を可視化してみましょう。
<!-- Let's visualize the trajectory of descending course from the initial point. -->

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
plt.colorbar()
plt.show()
```

2Dの画像は以下のようになります。
<!-- The 2D image of the trajectory is shown below. -->
<div align="center">
  
![MDpot](https://github.com/h-nabata/image_storage/blob/518beb88011362d257143411b2f4c748eb9c9412/MBpot2.svg "trajectory with the steepest descent method")

</div>

3Dの画像は以下のようになります。
<!-- The 3D image of the trajectory is shown below. -->
<div align="center">
  
![MDpot_3d](https://github.com/h-nabata/image_storage/blob/cbce1495f56ea868535c5819385a977c8623ad6f/MBpot6.svg "trajectory with the steepest descent method")

</div>


## Implementation (Newton's method)

次に、ニュートン法を実装してみましょう。ニュートン法を用いた最適化の考え方は、["Newton's method in optimization"](https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization)で説明されています。詳しくはこちらのページを参照してください。

まず、初期座標を決めます。ニュートン法では勾配ベクトルとヘシアン行列のみを用いて座標を更新するので、ステップ幅の指定は必要ありません。

<!-- Next, let's implement Newton's method. The concept of optimizetion by using Newton's method is explained in [this wiki page "Newton's method in optimization"](https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization).

Firstly, you should determine the initial coordinates. In Newton's method, coordinates are updated by using the gradient vector and the hessian matrix, so it does not require step size. -->

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

結果を出力します。
<!-- Then, print out the output. -->

```py
### ニュートン法による最適化の結果を出力
print("Terminal point = (", xc, ",", yc,") , Iteration =", iteration)
print("Energy =", f(xc, yc), ", Gradient =", np.sqrt(diffx ** 2 + diffy ** 2))
```

> Terminal point = ( -0.8220015587830566 , 0.6243128028186382 ) , Iteration = 12
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
plt.colorbar()
plt.show()
```

トラジェクトリの2D画像は以下の通りです。
<!-- The 2D image of the trajectory is shown below. -->
<div align="center">
  
![MDpot](https://github.com/h-nabata/image_storage/blob/daf15bc091000caff5d523c328c818d8df5f6887/MBpot3.svg "trajectory with Newton's method")

</div>

* * *

以上、最急降下法とニュートン法を実装しました。

ここで、初期点が同じでも到達する停留点の種類が異なっていることに注意しましょう。最急降下法では極小点、ニュートン法では鞍点が得られています。

また、ニュートン法の収束性は初期値やステップ幅などのパラメータに大きく依存します。各自でパラメータを変えて実行してみて下さい。

ニュートン法はあくまで停留点を見つけ出すアルゴリズムであり、必ずしも極小点に到達するとは限りません。
これを改善した手法として「準ニュートン法」があります。準ニュートン法ではセカント条件を満たすような近似ヘシアンが用いられます。
これにより、勾配法のように適切な降下方向を選択しつつ、極小点付近ではニュートン法に匹敵する収束性を発揮することができます。

<div align="center">
  
●　　　●　　　●

</div>

以上の内容を踏まえ、以下の課題に挑戦してみよう。

## Exercise 1
1. 初期点を変えることですべての極小点と鞍点を特定する。
2. 鞍点でのヘシアンを対角化し、固有値と固有ベクトルを求めて虚の振動方向を特定する。
3. 鞍点を初期点として2.で求めた虚の振動方向に対して僅かに動かした点を初期点として、最急降下法を用いてポテンシャル面を下ってみる。このときの軌跡はIRC経路に一致するので、これを図示してみよう。
4. 以下の式で定義されるポテンシャル面についても同様の手順で解析してみよう。
```
def f(x, y):
    # nabata potential 1
    A=[0.1, 0.1, 0.1, 0.001]
    a=[np.sin(x-2) + np.cos(y), np.sin(x-y) - np.cos(y), np.sin(y) + 3 * np.cos(x+y), x**2 + y**2 /6]
    s = []
    for i in range(4):
        s.append(A[i]*np.exp(a[i]))
    return sum(s)
```

<div align="center">

![f(x, y)=\frac{1}{10} e^{\sin (x-2)+\cos y}+\frac{1}{10} e^{\sin (x-y)-\cos y}+\frac{1}{10} e^{\sin y+3 \cos (x+y)}+\frac{1}{1000} e^{x^{2}+\frac{y^{2}}{6}}
](https://render.githubusercontent.com/render/math?math=%5Ccolor%7Bblack%7D%5Cdisplaystyle+f%28x%2C+y%29%3D%5Cfrac%7B1%7D%7B10%7D+e%5E%7B%5Csin+%28x-2%29%2B%5Ccos+y%7D%2B%5Cfrac%7B1%7D%7B10%7D+e%5E%7B%5Csin+%28x-y%29-%5Ccos+y%7D%2B%5Cfrac%7B1%7D%7B10%7D+e%5E%7B%5Csin+y%2B3+%5Ccos+%28x%2By%29%7D%2B%5Cfrac%7B1%7D%7B1000%7D+e%5E%7Bx%5E%7B2%7D%2B%5Cfrac%7By%5E%7B2%7D%7D%7B6%7D%7D%0A)

![NBpot1](https://github.com/h-nabata/image_storage/blob/bd9d763ae00ae4f287cd7dad38807350c3fbf2f2/MBpot4.svg "Nabata potential no. 1")
  
</div>

# Chapter 2
ポテンシャル面が多峰性の場合、停留点は複数存在します。それらをすべて見つけるには多数の初期点を用意して最適化計算を行わなければなりません。例えば、初期点を格子状に1.0間隔で100個用意して、そこからポテンシャル面を下るという「グリッド探索」が考えられます。実際、この方法によって所期の目的は達せられますが、100回も計算しなければいけないという点で非効率です。また、今回のような2次元のポテンシャルならば現実的な時間内で計算できますが、これが例えば10次元空間になるとグリッド点は10の10乗（=100億）個も必要となり、これでは到底計算が終わりません。

そこで、以下ではグリッド探索よりも効率良くPES上の停留点を見つける方法を考えてみましょう。

<div align="center">
  
●　　　●　　　●

</div>
  
多峰性のPES上の極小点は、別の極小点と最小エネルギー経路（Minimum Energy Path; MEP）で結ばれています。これは、盆地に位置する2つの町が峠道で繋がっている様子を想像すると、何となくイメージできると思います。簡単に言えば、MEPは2次元のPESで言うところの「谷底の経路」"valley path" に相当しています（※）。

> （※）
> ただし、MEPは "valley path" と表現するよりも "path of least resistance"（最小抵抗経路）と表現するのが適切と言えます。通常、谷底は曲面の曲率が小さい方向に伸びていますが、これを辿ったからといって、峠（ここでは「遷移状態」）に到達するとは限らないからです。実際の反応経路は「***峠を経由するような***最小エネルギー経路」と言えます。
> 
> (cf.) [Dunitz, J. D.: *Phil. Trans. R. Soc. Lond.* B272, 99 (1975)](https://www.jstor.org/stable/pdf/2417520.pdf)

MEPは2点間法を用いて求めるのが一般的です。以下に幾つかの2点間法を列挙します。

- [The nudged elastic band (NEB) method](https://aip.scitation.org/doi/pdf/10.1063/1.1323224)
- [The string method](https://journals.aps.org/prb/pdf/10.1103/PhysRevB.66.052301)
- [The locally updated planes (LUP) method](https://aip.scitation.org/doi/pdf/10.1063/1.460343)
- [The ReaDuct method](https://pubs.acs.org/doi/pdf/10.1021/acs.jctc.8b00169)

ただ、これらの手法の実装は本章の範囲を超えてしまうので今は取り上げません。そこで代わりの戦略を用意します。

<div align="center">
  
●　　　●　　　●

</div>

一般に、反応経路は Minimum～Saddle～Minimum の3点を端点、または経由するPES上の(超)曲線として定義されます。このような曲線のうち、エネルギーのロスが一番小さいものがもっともらしい反応経路と考えられます。

化学反応を特徴づける反応経路は、福井謙一（フロンティア軌道理論を考案した業績により、1981年にノーベル化学賞を受賞）によって「固有反応座標」として数学的に定式化されました。これを "IRC"（Intrinsic Reaction Coordinate）と呼びます。

IRC経路は簡単に言ってしまうと、PES上における最急降下経路として定義できます（正確には、質量荷重座標空間における最小エネルギー経路）。IRC経路は遷移状態を始点とし、遷移状態における虚の振動モードに沿ってポテンシャル面を降下した先の2つの極小点を接続します。

> （※）IRC は、遷移状態を始点として、荷重ヘシアン行列の負の固有値に対応する方向に原子座標を変位させ、ポテンシャル勾配の負の方向に軌跡をたどることによりポテンシャル曲面の極小点に至る仮想的な反応経路、と定義されます。

遷移状態が見つかれば、IRC経路、つまり反応経路を求めることができるのです。

<div align="center">
  
●　　　●　　　●

</div>

残念ながら、ポテンシャル面の極小点における情報のみから鞍点を見つけ出すことは**数学的に不可能**であることが証明されています。そのため、遷移状態の決定には計算者による予測が必要です。

> プログラムによる「反応経路探索」が主流になる前は、計算者が「遷移状態っぽい構造」を手で作って初期構造としていました。遷移状態は停留点なのでニュートン法によって求めることができますが、初期構造が悪いと全く鞍点に収束しません。遷移状態を求める計算は職人技が要求される作業だったのです。

さて、ここで少し、PES上の既知の極小点から**未知の極小点**を見つける手法について考えてみます。何らかの方法で既知の極小点から新しい極小点を見つけることができれば、上述した2点間法を適用して遷移状態を見つけることができそうです。
しかし、PES上の既知の極小点から未知の極小点を狙って見つけ出すという操作は、これまで不可能とされてきました。仮にできたとしてもランダムなサンプリングか、グラフ理論に基づく構造生成がせいぜいでした。

そんな折、2003年に大野公一と前田理は「[極座標内挿法](https://www.sciencedirect.com/science/article/pii/S0009261403017135)」という手法を発表しました。これは後にADDF法（非調和下方歪み追跡法）と呼ばれる反応経路自動探索アルゴリズムの原型となるコンセプトでした。

"ADD" というのは非調和下方歪み（Anharmonic Downward Distortion）の頭文字です。大野らは「極小点から離れるにつれて調和ポテンシャルよりも下側に歪む」というPESの性質に着目し、このADDが化学反応経路の進行方向に相当することを見出しました。極小点周辺のADDを検出して追跡（Following）することで、計算者の予測に依らない反応経路探索を実現しました。

> なお、化学反応の全面探索の試みそのものは以前から為されています。
> - 最小の固有値に着目して反応経路を探す Eigenvector Following法：EF法
> - 勾配が極値をとる条件を満たす点を停留点から辿る Gradient Extremal Following法：GEF法
> - 極小点を中心とする球面上のエネルギー最小点を球面を拡大しながら追跡する Sphere Optimization法：SO法
> (cf.) Pancíř, J. *Collect. Czech. Chem. Commun.* 1975, **40**, 1112–1118 / Abashkin, Y.; Russo, N. *J. Chem. Phys.* 1994, **100**, 4477–4483.

ADDF法は優れた手法ですが、PESの3次微分の情報が必要となるため計算コストが馬鹿になりません。原子数の多い大きな系ではその制約がより顕著に現れてしまいます。

これを克服した画期的な手法として、2010年に前田理と諸熊奎治が開発した[AFIR法（Artificial Force Induced Reaction method；人工力誘起反応法）](https://aip.scitation.org/doi/10.1063/1.3457903)があります。これは、化学反応における反応物同士に仮想的な引力や斥力に相当するポテンシャルを加えることで、PES上の遷移状態の構造を特定する、という手法です。


<div align="center">
　
<img src="https://github.com/h-nabata/image_storage/blob/16c09a0a30d4cc5a0d7a7dda386fb5198b30893d/AFIRmethod.svg" width="400" title="a schematic illustration of AFIR method">
  
</div>


図を見て分かる通り、AFIR法では人工力を加えて改変したポテンシャル面上における構造最適化のみを必要とすることから、これまでの手法に比べて格段に計算コストが削減されています。改変後のPESがバリアレスになるような適切なパラメータを選択することで、既存の極小点（平衡構造；EQ）から未知の生成物を第一原理的に得ることができます。

なお、この最適化計算で得られる構造は人工力を加えて改変したPES上の極小点（Approximate EQ）なので、真のEQではありません。この App. EQ を初期構造としてオリジナルのPES上で再び最適化を行うことで、真のEQを得ます。

また、構造最適化と同時にオリジナルのPESを追跡し、得られるエネルギープロファイルのピークから遷移状態の候補（Path Top）を特定することができます。これにより得られる近似的遷移状態（Approximate TS）をGuessとすれば、遷移状態が効率的に求められます。

<div align="center">
  
●　　　●　　　●

</div>

実際のAFIR法では、絶妙なバランスで設計された「AFIR関数」によって、上手い具合にフラグメント間に人工力が加えられます。ここでは、再びMüller-Brownポテンシャルを題材として「なんちゃってAFIR法」を実装してみることにします。

どのようにPESを改変するかがポイントとなる訳ですが、ここでは大雑把に考えて、x軸方向、y軸方向のそれぞれにPESを傾けるような「なんちゃってAFIR関数」を加えてみましょう。

```py
### importing libraries, ライブラリのインポート
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelmax
```

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

人工力関数を以下のように定義します。

```py
### なんちゃってAFIR関数の定義
def addint(x, y):
    a = 60.0
    b = -120.0
    return pes(x, y) + a*x + b*y
```

人工力を加えたり取り除いたりしやすいように、偏導関数には関数を引数に加えています。

```py
### 数値微分による偏導関数、2階の偏導関数の定義
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
```

初期構造は極小点の一つとしています。

```py
### 初期設定
(xc, yc) = ( 0.6234994 , 0.0280378 )    # 初期座標（＝初期構造）
xc_list = []; yc_list = []; ene_list = []  # 座標を格納するリストを用意する
ene_list = []                              # エネルギーを格納するリストを用意する
addint_xc_list = []; addint_yc_list = []   # 改変したPES用に座標を格納するリストを用意する
stepsize = 1e-4                            # STEP幅
maxitr = 10000                             # 座標更新サイクルの上限回数
```

最急降下法では降下方向を計算するためにPES上の各点における勾配ベクトルを計算する必要があります。座標の更新の終了条件は「勾配の大きさが適切な閾値を下回ること」とします。
<!-- The steepest descent method requires the gradient vector at each point on the PES to calculate the direction of descent. The termination condition for updating coordinates is that the magnitude of the gradient falls below an appropriate threshold. -->

```py
### steepest descent method 最急降下法 on 人工力を加えたPES
for i in range(1, maxtimes):
    times = i
    diffx1 = fx(addint, xc, yc)
    diffy1 = fy(addint, xc, yc)
    if np.sqrt(diffx1 ** 2 + diffy1 ** 2) < 1e-10:
        addint_xc_list.append(xc); addint_yc_list.append(yc)  # 座標をリストに追加
        ene_list.append(pes(xc, yc))  # AFIR経路に沿ったエネルギー値をリストに追加
        break
    else:
        addint_xc_list.append(xc); addint_yc_list.append(yc)  # 座標をリストに追加
        ene_list.append(pes(xc, yc))  # AFIR経路に沿ったエネルギー値をリストに追加
        xc = xc - stepsize * diffx1   # 次のx座標を生成
        yc = yc - stepsize * diffy1   # 次のy座標を生成

print("Optimization on modified PES finished!\n( itr =", times, ")\n(", xc, ",", yc,")")
print("Energy (a.f.) =", addint(xc, yc))
print("Grad (a.f.)   =", np.sqrt(fx(addint, xc, yc) ** 2 + fy(addint, xc, yc) ** 2))
print("Energy (bare) =", pes(xc, yc))
print("Grad (bare)   =", np.sqrt(fx(pes, xc, yc) ** 2 + fy(pes, xc, yc) ** 2))
```

> Optimization on modified PES finished!
> 
> ( itr = 2137 )
> 
> ( -0.5102233301489391 , 1.534199743630532 )
> 
> Energy (a.f.) = -357.3215991434462
> 
> Grad (a.f.)   = 0.0
> 
> Energy (bare) = -142.60423009884602
> 
> Grad (bare)   = 134.16407875612578


ここで、「なんちゃってAFIR関数」を加えて改変したPES上における最適化の結果を図示してみます。

```py
### 人工力を加えたPES
plt.plot(addint_xc_list, addint_yc_list, 'y.-', alpha=0.2)            # trajectory
plt.plot(addint_xc_list[0], addint_yc_list[0], 'b.-', alpha=0.2)      # initial point
plt.plot(addint_xc_list[-1], addint_yc_list[-1], 'r.-', alpha=0.2)    # terminal point

surf_x = np.linspace(-3.0, 1.5, 300)
surf_y = np.linspace(-1.0, 3.5, 300)
xmesh, ymesh = np.meshgrid(surf_x, surf_y)
z = addint(xmesh, ymesh)
level = []
for i in range(0,25):
    level.append(np.min(z) + (100-np.min(z))*0.04*i)
cont = plt.contourf(surf_x, surf_y, z, levels=level, cmap='coolwarm')
plt.colorbar()
plt.show()
```

<div align="center">
  
![MBpot_AFIR1](https://github.com/h-nabata/image_storage/blob/02ed88b9364c469e19524774bb0e844992e555e6/MBpot_AFIR1.svg "AFIR path on the modified Müller-Brown potential")

</div>

ここで得られた経路を「AFIR経路」と呼びます。この経路をオリジナルのPES上で辿ったときのエネルギープロファイルを図示してみます。ついでにピークも検出しておきましょう。

```py
### AFIR経路上のピークを検出する
local_max_id = argrelmax(np.array(ene_list), order=5)
# order: How many points on each side to use for the comparison to consider;
# この値を大きくすることでノイズを回避する

local_max_id_list = local_max_id[0]
print("-----------------\nThe # of App. TS found = ", len(local_max_id_list))
for i in range(len(local_max_id_list)):
    print("App. TS", i, "(", addint_xc_list[local_max_id_list[i]], ",", addint_yc_list[local_max_id_list[i]], ")")
    print("Energy (bare) =", pes(addint_xc_list[local_max_id_list[i]], addint_yc_list[local_max_id_list[i]]))

plt.plot([i for i in range(len(ene_list))], ene_list)
for i in range(len(local_max_id_list)):   # App. TS point(s)
    plt.plot(local_max_id_list[i], ene_list[local_max_id_list[i]], 'kx')
plt.show()
```

> The # of App. TS found =  2
> 
> App. TS 0 ( 0.23335787851086576 , 0.2994986937775166 )
> 
> Energy (bare) = -72.14008703460848
> 
> App. TS 1 ( -0.7314387998850193 , 0.6996159003161693 )
> 
> Energy (bare) = -37.696718520002925

横軸にiteration数、縦軸にEnergyをとると、以下のようになります。ピークが2つ出現しているのが分かります。

<div align="center">
  
![MBpot_AFIR2](https://github.com/h-nabata/image_storage/blob/02ed88b9364c469e19524774bb0e844992e555e6/MBpot_AFIR2.svg "energy profile along the AFIR path")

</div>

最後に、Approximate EQを初期構造としてオリジナルのPES上で再び最適化を行い、真のEQを求めてみます。トラジェクトリの図示まで一気にやってしまいましょう。

```py
### steepest descent method 最急降下法 on オリジナルのPES
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
for i in range(len(local_max_id_list)):   # App. TS point(s)
    plt.plot(addint_xc_list[local_max_id_list[i]], addint_yc_list[local_max_id_list[i]], 'kx', alpha=0.7)
plt.plot(xc_list, yc_list, 'g.-', alpha=0.3)            # trajectory
plt.plot(xc_list[0], yc_list[0], 'b.-', alpha=0.8)      # initial point
plt.plot(xc_list[-1], yc_list[-1], 'r.-', alpha=0.5)    # terminal point

surf_x = np.linspace(-3.0, 1.5, 300)
surf_y = np.linspace(-1.0, 3.5, 300)
xmesh, ymesh = np.meshgrid(surf_x, surf_y)
z = pes(xmesh, ymesh)
level = []
for i in range(-15,10):
    level.append(10.0 * i)
cont = plt.contourf(surf_x, surf_y, z, levels=level, cmap='coolwarm')
plt.colorbar()
plt.show()
```

> Optimization on original PES finished!
> 
> ( itr = 483 )
> 
> ( -0.5582236345379455 , 1.4417258419566479 )
> 
> Energy (bare) = -146.699517209954
> 
> Grad (bare)   = 0.0

ピークに該当する点にバツ印を付けてみました。遷移状態に近い点を通過しているのが分かりますね！
<div align="center">
  
![MBpot_AFIR3](https://github.com/h-nabata/image_storage/blob/02ed88b9364c469e19524774bb0e844992e555e6/MBpot_AFIR3.svg "AFIR path and final optimization")

</div>

このようにして見つかったAppriximate TSを初期構造としてニュートン法を適用すれば、遷移状態を得ることができます。さらにはIRCが得られ、Müller-Brownポテンシャルの完全な「反応経路ネットワーク」を得ることができます。

<div align="center">
  
●　　　●　　　●

</div>

ところで、上記の「なんちゃって反応経路探索」は偶然上手くいった例に過ぎません。というのも、AFIR関数のパラメータを変えると、とんでもないトラジェクトリが得られてしまう場合もあるからです。

そこで例えば人工力関数を以下のように定義するとどうなるでしょうか？

aとbの値を先ほどの**5倍**にしています。

```py
### なんちゃってAFIR関数の定義　その２
def addint(x, y):
    a = 300.0
    b = -600.0
    return pes(x, y) + a*x + b*y
```

同様に最適化を実行して結果をプロットしてみると以下のようになります。

<div align="center">
  
![MBpot_AFIR4](https://github.com/h-nabata/image_storage/blob/f8731b3ff76342972ac46c15e575d9995e5f7e52/MBpot_AFIR4.svg "AFIR path on the modified Müller-Brown potential")

![MBpot_AFIR5](https://github.com/h-nabata/image_storage/blob/76c8eda82d22a172c5023659b80adb8c8a04f5cc/MBpot_AFIR5.svg "energy profile along the AFIR path")

![MBpot_AFIR6](https://github.com/h-nabata/image_storage/blob/76c8eda82d22a172c5023659b80adb8c8a04f5cc/MBpot_AFIR6.svg "AFIR path and final optimization")
  
</div>

なかなかダイナミックな経路が得られてしまいました。このように、パラメータが不適切だと得られる経路も不適切なものになってしまいます。実際にGRRMプログラムで指定するGammaの値でも同様のことが起こります。Gammaの値はおおよその活性化障壁の高さに相当しています。100 kJ/mol 程度の反応経路に対して、例えば `Gamma=1000` などの巨大な値にしてしまうと、IRC経路からひどく逸脱したAFIR経路が得られてしまいます。

適切な人工力のパラメータを選ぶことも勿論重要なことですが、得られたAFIR経路をできるだけIRC経路に近付けるような最適化（エネルギーの緩和）の手法が必要となります。是非、2Dポテンシャルでの経路緩和も実装してみて下さい。

<div align="center">
  
●　　　●　　　●

</div>

## Notes
内容に関する著作権は作成者に帰属します。無断での複製や再配布などを禁じます。

## Acknowledgement and References
* https://tex-image-link-generator.herokuapp.com/
* [化学反応経路の自動探索](https://www.jstage.jst.go.jp/article/molsci/5/1/5_1_A0042/_pdf/-char/ja)
