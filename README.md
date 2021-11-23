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

### generate 
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

}
```