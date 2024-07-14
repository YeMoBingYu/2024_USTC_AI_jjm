# HW6
## PB21111686_赵卓
### 8.24
- 词汇表如下：
  - $Student(x)$表示$x$是一个学生
  - $Take(x,y,s,c)$表示$x$在$y$年$s$学期上了$c$课
  - $Pass(x,c)$表示$x$通过了$c$课程考试
  - $Score(x,y,s,c)$表示表示$x$在$y$年$s$学期在$c$课的成绩
  - $Better(c_1,c_2)$表示$c_1$成绩好于$c_2$
  - $Person(x)$表示$x$是一个人
  - $Buy(x,i)$表示$x$买了保险$i$
  - $Insurance(i)$表示$i$是保险
  - $Smart(x)$表示$x$是聪明的
  - $Expensive(i)$表示保险$i$是昂贵的
  - $Agent(x)$表示$x$是代理
  - $Sell(x,y)$表示$x$买保险给$y$
  - $Tony(x)$表示$x$是理发师
  - $Shave(x,y)$表示$x$给$y$刮胡子
  - $Born(x,y)$表示$x$出生在$y$地
  - $Parents(x)$表示$x$的双亲
  - $People(x,y)$表示$x$是$y$的公民
  - $Live(x,y)$表示$x$是$y$的永久居住者
  - $Blood(x,y)$表示$x$在血统上是$y$的人
  - $Politics(x)$表示$x$是政治家
  - $Fool(x,y,t)$表示$x$在$t$时刻愚弄$y$
- 在以上词汇表的基础上，各句子表示如下：
  - a.
    $\exist x(Student(x)\wedge Take(x,2001,Spring,France))$
  - b.
    $\forall y\forall s\forall x(Student(x)\wedge Take(x,y,s,France))\Rightarrow Pass(x,France)$
  - c.
    $\exist x(Student(x)\wedge Take(x,2001,Spring,Greece))\wedge (\forall y(y\ne x)\Rightarrow\neg(Take(y,2001,Spring,Greece)))$
  - d.
    $\forall y\forall \exist x_2\forall x_1Better(Score(x_2,y,s,Greece),Score(x_1,y,s,France))$
  - e.
    $\forall i\forall x(Person(x)\wedge Buy(x,i))\Rightarrow Smart(x)$
  - f.
    $\forall i(Insurance(i)\wedge Expensive(i))\Rightarrow \neg \exist x(Person(x)\wedge Buy(x,i))$
  - h.
    $\forall x(Person(x)\wedge \neg Shave(x,x))\Rightarrow \exist y(Tony(y)\wedge Shave(y,x))$
  - i.
    $\forall x(Person(x)\wedge (People(Parents(x),UK)\vee Live(Parents(x),UK)))\Rightarrow People(x,UK)$
  - j.
    $\forall x(\neg Born(x,UK)\wedge People(Parents(x),UK))\Rightarrow Blood(x,UK)$
  - k.
    $\forall x(Politics(x)\wedge (\forall y_1Person(y_1)\Rightarrow \exist t_1Fool(x,y_1,t_1)) \wedge (\forall t_2 \exist y_2(Person(y_2)\wedge Fool(x,y_2,t_2))) \wedge \neg(\forall y_3Person(y_3)\Rightarrow \forall t_3Fool(x,y_3,t_3))  )$
<br>

### 8.17
- 该定义有两处错误：
  - 一方面，只定义了$[x,y]$的上方和右方相邻，而没有定义下方和左方，这样会导致我们可以得出$Adjacent([1,2],[2,1])$但无法得出$djacent([2,1],[1,2])$
  - 另一方面，我们无法根据这个定义得出两个结点是否不相邻，如$[1,3]$和$[3,1]$无法证明不相邻
- 正确定义：
  $\forall x,y,a,b Adjacent([x,y],[a,b])\Leftrightarrow (x=a\wedge (y=b-1\vee y=b+1))\vee (y=b\wedge (x=a-1\vee x=a+1))$


### 9.3
- b是存在量词实例化后的合法结果，将$x$置换成$Kiilimanjaro$
- a错误，因为$Everest$已经存在，不能置换
- c错误，因为存在量词实例化只能进行一次
<br>

### 9.4
- a.
  存在最一般合一置换$\{x/A,y/B,z/C\}$
- b.
  不存在最一般合一置换，这是因为$y$置换成$G(x,x)$后，$x$既要置换成$A$又要置换成$B$，这无法实现
- c.
  存在最一般合一置换$\{x/John,y/John\}$
- d.
  不存在最一般合一置换，这是因为$x$置换成$Father(y)$后，$y$也要置换成$Father(y)$，这无法实现
<br>

### 9.6
- 我们先给出词汇表：
  - $Horse(x)$表示$x$是马
  - $Pig(x)$表示$x$是猪
  - $Cattle(x)$表示$x$是牛
  - $Mammal(x)$表示$x$是哺乳动物
  - $Parents(x,y)$表示x是y的家长
  - $Child(x,y)$表示x是y的后代
- 各语句表示为：
  - a.
    $\forall x Horse(x)\Rightarrow Mammal(x)$
    $\forall x Pig(x)\Rightarrow Mammal(x)$
    $\forall x Cattle(x)\Rightarrow Mammal(x)$
  - b.
    $\forall x\forall y(Horse(y)\wedge Child(x,y)) \Rightarrow Horse(x)$
  - c.
    $Horse(Bluebeard)$
  - d.
    $Parents(Bluebeard,Charile)$
  - e.
    $\forall x \forall yParents(x,y)\Rightarrow Child(y,x)$
    $\forall x \forall yChild(x,y)\Rightarrow Parents(y,x)$
  - f.
    $\forall xMammal(x)\Rightarrow \exist yParents(y,x)$
<br>

### 9.13
- a.证明树如下：
  ```mermaid
  graph TB;
  1[Horse（h）]---2[Child（h,y）]
  1---3[Horse（Bluebeard）]
  2---4[Parents（y,h）]
  4---5[Parents（Bluebeard,Charile）
       y/Bluebeard,h/Charile]
  3---6[Horse（y）]
  3---7[Child（Bluebeard,y）]
  7---8[Parents（y,Bluebeard）]
  8---9[Child（Bluebeard,y）]
  9---10[...]
  ```
- b.本题中，反向链接算法失败，陷入了无限循环中，无法找到正确证明，这是因为对于某些知识库，反向链接无法证明被蕴含的语句，因此其是不完备的。
- c.实际上本题是存在两个h的解的，分别是$Horse(Bluebeard)$和$Horse(Charile)$
  