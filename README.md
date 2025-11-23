# **Alpha Fund Project**

Collaborative quantitative finance + data analysis project.

This README describes:

1.  **Project Details & Structure** -- how the repository is organised and where different code belongs
2.  **Git Workflow** -- how the to use branches, commits, pulls, and pull requests

---

# **1. Project Details & Structure**

## **1.1 Project Theory**


### **1.1.1 Market Mechanism — Stop-Loss Cascades**

Stop-loss cascades occur when the market breaks through price levels where many traders have placed stop-loss orders.  
A stop-loss is an automatic instruction to *sell* if price falls below a threshold.  
When many traders place stops at similar levels, price breaks can trigger **simultaneous forced selling**.

Common levels where stop-losses cluster:

- **Recent lows:**  $L_t = \min(P_{t-k}, ..., P_{t-1})$
- **Round numbers:** $P_t \approx 100,\; 150,\; 200$
- **Technical thresholds:** moving averages, trendlines

When price breaks below a crowded level:

$$
P_t < L_t - \varepsilon,
$$

a large set of stop orders triggers, converting into **market sell orders** and creating a strong **one-sided flow imbalance**:

$$
\text{OrderImb}_t = \text{SellVol}_t - \text{BuyVol}_t \gg 0 .
$$

Due to limited order-book depth, this imbalance produces:

1. **Large downward price movement:**   $r_t = \log\left(\frac{P_t}{P_{t-1}}\right) \ll 0$
2. **Volume spike:**  $V_t \gg \mathbb{E}[V_t]$
3. **Short-term volatility surge**

These effects form the identifiable signature of a stop-loss cascade.

---

### **1.1.2 Price Dynamics Around Cascades**

Let the forward return over a short horizon $\delta t$ be:

$$
r_{t,t+\delta t} = \log\left(\frac{P_{t+\delta t}}{P_t}\right).
$$

After a cascade, markets typically show one of **two behaviours**:

---

#### **1.1.2.1 Liquidity-Driven Overshoot (Mean Reversion)**

If the cascade reflects *mechanical forced selling* rather than new information:

- Forced sales clear quickly  
- Liquidity providers restore depth  
- Price “snaps back” toward equilibrium  

Thus, expected forward returns become **positive**:

$$
\mathbb{E}[r_{t,t+\delta t} \mid \text{cascade, no news}] > 0 .
$$

This corresponds to a temporary **price overshoot**.

---

#### **1.1.2.2 Information-Driven Shock (Continuation)**

If the cascade is triggered by genuine negative information:

- Informed traders continue selling  
- The initial drop underreacts to the news  
- Price continues drifting downward  

Thus:

$$
\mathbb{E}[r_{t,t+\delta t} \mid \text{cascade, news shock}] < 0 .
$$

---

#### **Key Idea:**  
Outside cascades, short-horizon returns behave like a near-random walk:

$$
\mathbb{E}[r_{t,t+\delta t}] \approx 0 .
$$

Inside cascades, the distribution becomes **directionally biased**, making returns predictable.

---

### **1.1.3 Identifying Cascade Events (Proxy Method)**

Because stop-loss orders are unobservable, we detect cascades using observable patterns.

A bar $t$ is labeled a **cascade candidate** if:

---

#### **1. Large negative return relative to volatility**

$$
r_t < -k \sigma_{t-1}, \quad k \in [1.5,3].
$$

---

#### **2. Volume spike**

$$
Z^V_t = \frac{V_t - \mu^V}{\sigma^V} > z_{\min}.
$$

---

#### **3. Break of a crowded level**

$$
P_t < L_t - \varepsilon .
$$

---

#### **4. Local acceleration pattern**

Several small negative returns followed by a large one:

$$
r_{t-3}, r_{t-2}, r_{t-1} < 0, \quad |r_t| \text{ large}.
$$

These four conditions jointly provide a practical proxy for stop-loss cascades.

---

### **1.1.4 Regime-Conditioned Predictability**

Define the regime indicator:

$$
C_t = \mathbf{1}\{\text{cascade at } t\}.
$$

Under normal market conditions:

$$
\mathbb{E}[r_{t,t+\delta t}] \approx 0, \qquad 
\text{Cov}(r_t, r_{t,t+\delta t}) \approx 0.
$$

However, during cascades, forward returns decompose into:

$$
r_{t,t+\delta t}
= r^{\text{forced}}_{t,t+\delta t}
+ r^{\text{fundamental}}_{t,t+\delta t}.
$$

- $r^{\text{forced}}$ = predictable effects of temporary, one-sided forced flow  
- $r^{\text{fundamental}}$ = price reaction to genuine information  

Since forced flow is mechanical and temporary, conditioning on $C_t = 1$ creates **non-zero expected returns**:

$$
\mathbb{E}[r_{t,t+\delta t} \mid C_t = 1, X_{t-L:t}] = f(X_{t-L:t}) \neq 0.
$$

This is **regime-conditioned predictability**:  
returns become predictable *only* within the cascade regime.

---

### **1.1.5 Sequence Modelling & Feature Engineering**

Stop-loss cascades are **path-dependent** — the sequence leading to the break matters.

Useful features include:

#### **Price action**
- Recent returns $r_{t-i}$
- Realised volatility $\sigma_t$
- Return acceleration
- Drawdown depth

#### **Volume/flow**
- Raw volume  
- Volume z-score  
- Volume acceleration

#### **Distance to levels**
$$
d_t = P_t - L_t
$$

#### **Microstructure regime proxies**
- Volatility regime  
- Sector correlation stress  

---

#### **Sequence model inputs**

At time $t$, we form a feature vector:

$$
x_t = (r_t, \sigma_t, Z^V_t, d_t, \Delta\sigma_t, \Delta V_t, \dots).
$$

We feed the last $L$ feature vectors to the model:

$$
X_{t-L:t} = \{x_{t-L}, ..., x_t\}.
$$

The model estimates:

$$
p_t =
P(r_{t,t+\delta t} > 0 \mid C_t = 1,\; X_{t-L:t}).
$$

This represents the probability of **mean reversion** following a cascade.

---

### **1.1.6 Why Use an LSTM?**

LSTMs are well-suited because cascades involve:

- Gradual build-up in volatility  
- Rising volume before the break  
- Slow drift toward the key level  
- A final sudden break  

Static models cannot capture this **temporal structure**.

LSTMs model non-linear, path-dependent sequences, making them appropriate for detecting regime shifts and predicting outcomes after cascades.

---

### **1.1.7 Trading Interpretation**

The model output $p_t$ can be interpreted directly:

- **$p_t \approx 1$:**  
  High chance of **mean reversion** → go **long**
- **$p_t \approx 0$:**  
  High chance of **continuation** → go **short** or avoid longs  
- **$p_t \approx 0.5$:**  
  Uncertain → **no trade**

#### **Simple trading rule**

Enter trades *only* at cascade events:

- Long if $p_t > 0.5$  
- Short if $p_t < 0.5$  
- Close the position after $\delta t$ bars

This isolates the strategy to periods of **temporary market dislocation**.

---

### **1.1.8 Empirical Tests**

To validate the theory:

#### **1. Distributional differences**
Compare cascades vs normal periods in terms of:

- Mean forward returns  
- Volatility  
- Volume  
- Tail behaviour  

Cascades should display abnormal behaviour.

---

#### **2. Predictability tests**
Evaluate:

- Accuracy  
- ROC-AUC  
- Calibration curves  
- Comparison vs baseline models  

This confirms the model extracts real signal from cascade regimes.

---

#### **3. Trading performance**
Run a simple backtest:

- Only trade at cascade events  
- Hold for $\delta t$  
- Measure mean PnL, Sharpe, hit rate  

Even small positive performance validates the underlying mechanism.

---

#### **4. Feature ablation studies**
Remove features one at a time and observe performance degradation.  
This reveals which features carry most predictive power.

---

#### **5. Sensitivity analysis**
Vary sequence length $L$:

- Short memory (10 bars)  
- Medium (20–30)  
- Long (50+)  

This tests how far back the cascade signal extends.

---


## **1.2 Repository Layout**

    Alpha-Fund-Project/
    │
    ├── src/                    # All reusable code or functions
    │
    ├── notebooks/              # Jupyter notebooks
    │
    ├── data/
    │   ├── raw/                # Raw datasets (NOT in git)
    │   ├── processed/          # Processed data (NOT in git)
    │   └── README.md           # How to obtain datasets
    │
    ├── requirements.txt        # Shared dependencies
    ├── README.md               # This file
    └── .gitignore              # Files/folders git must ignore


## **1.3 Data Handling Policy**

`data/raw/` and `data/processed/` are **ignored by git**.
Put large files here --- NOT in the repository.

If a dataset is needed:

-   Document download instructions in `data/README.md`
-   Do NOT commit large `.csv`, `.parquet`, or raw data



## **1.4 Notebook Usage**

-   Put notebooks in `notebooks/`
-   For personal experiments, create:
    -   `notebooks/<name>_experiments.ipynb`


## **1.5 Code Organisation Rules**

-   All reusable code → `src/`
-   All one-off experiments → `notebooks/`
-   No big functions inside notebooks\
    (Notebooks should call functions from `src`)

---

# **2. Github Workflow**
## 2.1 Before You Start: Update `main`

Always start by getting the latest version of the project.

```bash
git checkout main
git pull
```

- `git checkout main` → switch to the `main` branch
- `git pull` → download and apply the latest changes from GitHub

---

## 2.2 Create a New Branch for Your Work

Never work directly on `main`.  
Create a new branch for each task or feature.

```bash
git checkout -b feature/<short-description>
```

Examples:

```bash
git checkout -b feature/readme-update
git checkout -b feature/data-cleaning
```

Now you are working on your own branch, separate from `main`.

---

## 2.3 Make Changes and Check Status

Edit files as needed.

To see what you changed:

```bash
git status
```

This shows:
- which files are modified
- which files are untracked (new)

---

## 2.4 Add Your Changes

When you are ready to save your work, **add** the files you changed:

```bash
git add <file1> <file2>
```

Example:

```bash
git add README.md src/utils.py
```

To add all changed files at once (be careful):

```bash
git add .
```

---

## 2.5 Commit Your Changes

After adding files, **commit** them with a short message:

```bash
git commit -m "Describe what you did"
```

Examples:

```bash
git commit -m "Add basic README setup instructions"
git commit -m "Implement data loading function"
```

A commit is like a save point in the project history.

---

## 2.6 Push Your Branch to GitHub

Send your branch to GitHub so others can see it:

```bash
git push -u origin feature/<short-description>
```

Example:

```bash
git push -u origin feature/readme-update
```

- `origin` = the GitHub copy of the repository
- `-u` links your local branch to the remote one (so later you can just use `git push`)

---

## 2.7 Open a Pull Request (PR)

1. Go to the project on GitHub
2. GitHub will suggest creating a **Pull Request** for your branch
3. Click **“Compare & pull request”**
4. Check that the base branch is `main` and the compare branch is your `feature/...` branch
5. Write a short description of your changes
6. Click **“Create pull request”**

Ask a teammate to review your PR if possible.

---

## 2.8 Merge the Pull Request

Once the PR is approved:

1. Click **“Merge pull request”** on GitHub
2. Confirm the merge
3. Optionally, delete the branch on GitHub after merging

Now your changes are part of `main`.

---

## 2.9 Update Your Local `main` After a Merge

After your PR (or someone else's) is merged:

```bash
git checkout main
git pull
```

This keeps your local copy in sync with GitHub.

---

## 2.10 Start a New Task

For every new task:

1. Update `main`:

   ```bash
   git checkout main
   git pull
   ```

2. Create a new branch:

   ```bash
   git checkout -b feature/<new-task>
   ```

Then repeat the same workflow: change → add → commit → push → PR → merge.

---

## Quick Summary

1. **Sync main**: `git checkout main` → `git pull`
2. **Branch**: `git checkout -b feature/<task>`
3. **Work**: edit files
4. **Add**: `git add <files>` or `git add .`
5. **Commit**: `git commit -m "Message"`
6. **Push**: `git push -u origin feature/<task>`
7. **PR on GitHub** → review → merge
8. **Update main** again: `git checkout main` → `git pull`

Stick to this flow and the repo will stay clean and easy for everyone to use.

