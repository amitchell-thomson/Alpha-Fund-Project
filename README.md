# **Alpha Fund Project**
------------------------------------------------------------------------

# **1. Project Outline**

## **1.1 Overview**

- Describe project plan

## **1.2 Theory**

- Describe Theory Plan


------------------------------------------------------------------------

# **2. Project Structure**


## **2.1 Repository Layout**

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

## **2.2 Data Handling Policy**

`data/raw/` and `data/processed/` are **ignored by git**.
Put large files here --- NOT in the repository.

If a dataset is needed:

-   Document download instructions in `data/README.md`
-   Do NOT commit large `.csv`, `.parquet`, or raw data

------------------------------------------------------------------------

## **2.3 Notebook Usage**

-   Put notebooks in `notebooks/`
-   For personal experiments, create:
    -   `notebooks/<name>_experiments.ipynb`

------------------------------------------------------------------------

## **2.4 Code Organisation Rules**

-   All reusable code → `src/`
-   All one-off experiments → `notebooks/`
-   No big functions inside notebooks\
    (Notebooks should call functions from `src`)

------------------------------------------------------------------------