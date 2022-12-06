# EE660 Final Project:  Predicting Edibility of Mushrooms
Author: Yunlin Tang


## Set Up
- python 3
- If needed, install the used modules included in the requirements.txt:
```
pip install -r requirements.txt
```


## Repository Structure
```
├── data
│   ├── mushroom.csv
├── qns3vm
├── notebook
│   ├── EDA.ipynb
├── README.md
├── requirements.txt
├── main.py
├── model.py
├── preprocess.py
└── .gitignore
```

- `data` includes the necessary dataset for training and testing model. After training, this dataset will also include some interim datasets.
- `qns3vm` is the S3VM source code which is provided by Fabian Gieseke, Antti Airola, Tapio Pahikkala, Oliver Kramer. The code can be found in this [GitHub Repo](https://github.com/NekoYIQI/QNS3VM/blob/master/qns3vm.py).
- `notebook` includes a jupyter notebook that shows all steps of this project. In particular, it also shows the EDA step with multiple plots and tables.
- `main.py`, `model.py` and `proprocess.py` are source code files that build for this project


## Training Models
- train
  - run **`python main.py train`** to train all models and report the training/validation scores

## Final Model Evaluation
- run **`python main.py`** to report the test scores of the 2 selected final systems (SL: decision tree, SSL: label propagation).
  - NO need to run **`python main.py train`** to report these test scores