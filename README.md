# **Alpha Fund Project**

Collaborative quantitative finance + data analysis project.

This README describes:

1.  **Project Details & Structure** -- how the repository is organised and where different code belongs
2.  **Git Workflow** -- how the to use branches, commits, pulls, and pull requests

---

# **1. Project Details & Structure**
## **1.1 Project Overview/Theory**
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

