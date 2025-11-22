# **Alpha Fund Project**

Collaborative quantitative finance + data analysis project.

This README describes:

1.  **Setup** -- how to install everything and get the project running on your machine
2.  **Git Workflow** -- how the to use branches, commits, pulls, and pull requests
3.  **Project Details & Structure** -- how the repository is organised and where different code belongs

------------------------------------------------------------------------

# **1. Setup**

Follow these steps to set up the project **on your laptop**.

------------------------------------------------------------------------

## **1.1 Requirements**

Make sure you have:

### **Essential**

-   **Git**
-   **Python 3.10+**
-   **Any code editor** Recommended: **VS Code**\
(please use this I can't help if something goes wrong if you don't)


------------------------------------------------------------------------

## **1.2 Clone the Repository**

Open Terminal (macOS/Linux):

``` bash
git clone https://github.com/amitchell-thomson/Alpha-Fund-Project.git
cd Alpha-Fund-Project
```

------------------------------------------------------------------------

## **1.3 Create Your Python Environment**

Using a local `.venv` inside the project directory. (basically just a small self contained system)

### **macOS**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

You should now see `(.venv)` in your terminal prompt.

------------------------------------------------------------------------

## **1.4 Install Required Libraries**

``` bash
pip install --upgrade pip
pip install -r requirements.txt
```

All dependencies must be installed from `requirements.txt` so everyone uses the same library versions.

------------------------------------------------------------------------

## **1.5 VS Code Setup (Recommended)**

### **Step 1 --- Open the folder**

    File → Open Folder → select Alpha-Fund-Project

### **Step 2 --- Select the interpreter**

    Cmd + Shift + P → “Python: Select Interpreter”

Choose the `.venv` environment.

### **Step 3 --- Install Python on VS Code**

-   Python

------------------------------------------------------------------------

# **2. Git Workflow**

This section explains **how the to use GitHub**, including branches, commits, pulls, and pull requests.

------------------------------------------------------------------------

## **2.1 Basic Workflow**

-   **Do NOT commit directly to `main`**\
-   All work must be done on **feature branches**
-   Every change must go through a **Pull Request (PR)**
-   At least one teammate should review PRs before merging
-   **Pull updates frequently** to stay in sync

------------------------------------------------------------------------

## **2.2 Updating Your Local Main Branch**

Always start work by syncing with GitHub:

``` bash
git checkout main
git pull
```

------------------------------------------------------------------------

## **2.3 Creating a Feature Branch**

Use descriptive names:

``` bash
git checkout -b feature/<task-name>
```

Examples: - `feature/pnl-visualisation` - `feature/data-cleaning` - `feature/model-baseline`

------------------------------------------------------------------------

## **2.4 Making Changes & Committing**

Check what changed:

``` bash
git status
```

Add files:

``` bash
git add <file1> <file2>
```

Commit:

``` bash
git commit -m "Clear description of change"
```

------------------------------------------------------------------------

## **2.5 Push Your Branch**

``` bash
git push -u origin feature/<task-name>
```

------------------------------------------------------------------------

## **2.6 Open a Pull Request (PR)**

1.  Go to GitHub → the repository
2.  Click **"Compare & pull request"**
3.  Add a short description of your work
4.  Assign a reviewer
5.  Merge only after approval

------------------------------------------------------------------------

## **2.7 Keeping Your Branch Updated**

If someone else updates `main`, merge their changes into your branch:

``` bash
git checkout main
git pull
git checkout feature/<task-name>
git merge main
```

Resolve conflicts if needed, then continue working.

------------------------------------------------------------------------

## **2.8 Installing New Dependencies**

If you add a library:

``` bash
pip install <package>
pip freeze > requirements.txt
git add requirements.txt
git commit -m "Add <package> dependency"
git push
```

Teammates then update:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

# **📁 3. Project Details & Structure**

------------------------------------------------------------------------

## **3.1 Repository Layout**

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

------------------------------------------------------------------------

## **3.2 Data Handling Policy**

`data/raw/` and `data/processed/` are **ignored by git**.
Put large files here --- NOT in the repository.

If a dataset is needed:

-   Document download instructions in `data/README.md`
-   Do NOT commit large `.csv`, `.parquet`, or raw data

------------------------------------------------------------------------

## **3.3 Notebook Usage**

-   Put notebooks in `notebooks/`
-   For personal experiments, create:
    -   `notebooks/<name>_experiments.ipynb`

------------------------------------------------------------------------

## **3.4 Code Organisation Rules**

-   All reusable code → `src/`
-   All one-off experiments → `notebooks/`
-   No big functions inside notebooks\
    (Notebooks should call functions from `src`)

------------------------------------------------------------------------