# HW5
## PB21111686_赵卓
### 7.13
- a.
  $\because \neg(P\wedge Q)=\neg P\vee \neg Q$
  $\therefore \neg P_1\vee \dots \vee \neg P_m \equiv \neg(P_1 \wedge \dots \wedge P_m)$
  $\therefore \neg P_1\vee \dots \vee \neg P_m\vee Q\equiv \neg(P_1 \wedge \dots \wedge P_m)\vee Q$
  $\because P\Rightarrow Q\equiv\neg P\vee Q$
  $\therefore \neg(P_1 \wedge \dots \wedge P_m)\vee Q \equiv (P_1 \wedge \dots \wedge P_m) \Rightarrow Q$
  $\therefore \neg P_1\vee \dots \vee \neg P_m\vee Q\equiv (P_1 \wedge \dots \wedge P_m) \Rightarrow Q$ 
<br>

- b.
  由每个子句的真值表我们可以将每个子句都可以写成$\neg P_1\vee \dots \vee \neg P_m \vee Q$的形式
  由a可知$\neg P_1\vee \dots \vee \neg P_m\vee Q\equiv (P_1 \wedge \dots \wedge P_m) \Rightarrow Q$
  而$Q$由真值表也可以写成$Q_1\vee \dots \vee Q_n$的形式
  因此每个子句都可以写成$(P_1 \wedge \dots \wedge P_m)\Rightarrow(Q_1\vee \dots \vee Q_n)$的形式
<br>

- c.
  蕴含范式的完整归结规则如下：
  $\frac{p_1\vee \dots \vee p_n \Rightarrow q_1\vee \dots \vee q_n,s_1\vee\dots \vee s_n \Rightarrow r_1\vee\dots \vee r_n}{p_1\vee \dots \vee p_{j-1}\vee p_{j+1}\vee\dots\vee p_n\vee s_1\vee \dots \vee s_n \Rightarrow q_1\vee \dots \vee q_{k-1}\vee q_{k+1}\vee \dots \vee q_n\vee r_1\vee \dots\vee r_n}$
  其中$p_j=q_k$


<br>

### Proof
- 假设FC到达不动点，此时不可能再出现新的推理
  对于所有KB蕴含的原子语句，假设存在b不能被推理，则b的蕴含式为$a_1\wedge \dots \wedge a_n \Rightarrow b$为false
  即$a_1\wedge \dots \wedge a_n$为true而$b$为false
  但根据FC算法，此时b会被继续推理而不是到达不动点，这与我们的假设违背
  因此FC到达不动点之后，不可能存在不能被推理的蕴含原子语句

  