# TropiPay Fraud Detection System 

## Skeleton 

```plain
- bin: Binaries
- cfg: Configurations
- src: Source Code
    - eda: Exploratory Data Analysis
    - lsc: Learning Classifier System
```

## Requirements 
- python v3.8.3
- pip v20.1.1

## Install
- git clone https://github.com/ameksike/kdd.fraud.detection.system.git
- virtualenv env
- .\env\Scripts\activate
- pip install -r requirements.txt
- pip list

## Develop
- git clone https://github.com/ameksike/kdd.fraud.detection.system.git
- pip install virtualenv
- virtualenv env
- .\env\Scripts\activate
- pip install Flask
- pip install pandas
- python -m pip freeze > requirements.txt

## Run 1 with heroku
- heroku local web -f Procfile.win2
- http://127.0.0.1:8000/

## Run 2 with python
- python bin/server.py 
- http://127.0.0.1:8000/


## Endopints

### Generate 
Request:
```
POST http://127.0.0.1:8000/api/lcs/generate
```
Response:
```json
{
  "data": [
    41872,
    77
  ]
}
```

### Train
Request:
```
POST http://127.0.0.1:8000/api/lcs/train
{
    "modelname": "dataMiningView"
}
```
Response:
```json
{
  "test": {
    "train": 0.8884152427450016,
    "validation": 0.8783281573682612,
    "value": 0.8859874809584283
  },
  "train": {
    "list": {
      "0.0001": 0.8783281573682612,
      "0.001": 0.8766600554209258,
      "0.01": 0.8605071922007237,
      "0.1": 0.6723685070867722,
      "1.0": 0.17649010264579928,
      "10.0": 0.5,
      "100.0": 0.5,
      "1000.0": 0.5,
      "10000.0": 0.5
    },
    "max": {
      "alpha": 0.0001,
      "score": 0.8783281573682612
    }
  }
}
```

### Classify
Request:
```
POST http://127.0.0.1:8000/api/lcs/classify
{

}
```
Response:
```json

```