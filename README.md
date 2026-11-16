# DSC 478 Final Project

DePaul University<br>
DSC 478 Programming Machine Learning Applications<br>
2026-2027 Autumn

## 1. About

---

The objective of the project is to create a collaborative recommender system that will recommend books,
using data sourced from [Goodreads](https://www.goodreads.com/), 


## 2. Run

---

### 2.1 Setup

Install all required libraries:
<pre>
    python -m pip install -r requirements.txt
</pre>
Requires Python 3.12+. No bundled virtual environment.

### 2.2 Data Analysis

The file **data-download/targets.txt** contains a list of genres that will be targeted for analysis and for which relevant data will be downloaded.
Valid genres are 'children', 'comics_graphic', 'fantasy_paranormal', 'history_biography', 'mystery_thriller_crime', 'poetry', 'romance', and 'young_adult'.

#### Download the data

This step only needs to be executed once.

<pre>
    python -m data-download.DataDownload --out data/archive/ --genres data-download/targets.txt
</pre>

#### Start analysis and initialize recommender system
<pre>
    python -m recommender.Analyze --input data/archive/ --genres data-download/targets.txt
</pre>

## 3. Usage

---


## 4. Built With

---

Python 3.13

### 4.1 Required Libraries

[numpy](https://numpy.org/)<br>
[pandas](https://pandas.pydata.org/)


## 5. Contact

---

William Berthouex<br>
Email: wberthou@depaul.edu<br>
Project Link: [GitHub](https://github.com/StateControlled/dsc478-recommender-system)

## 6. Acknowledgements

---

### 6.1 Primary Data

Mengting Wan, Julian McAuley, "Item Recommendation on Monotonic Behavior Chains", in RecSys'18.<br>
Mengting Wan, Rishabh Misra, Ndapa Nakashole, Julian McAuley, "Fine-Grained Spoiler Detection from Large-Scale Review Corpora", in ACL'19.

Website: https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html

Some code is adapted from https://github.com/MengtingWan/goodreads as allowed by the [Apache License 2.0](http://www.apache.org/licenses/)


### 6.2 AI Usage
None
