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
POST http://127.0.0.1:8000/api/lcs/traing
{
    "modelname": "datamining_view"
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

### Traing information
Request:
```
GET http://127.0.0.1:8000/api/lcs/traing
```
Response:
```json
{
  "algorithms": [
    "logisticRegression", 
    "ensembleClassify"
  ], 
  "dataMinings": [
    "datamining_view"
  ]
}
```

### Classify
Request:
```
POST http://127.0.0.1:8000/api/lcs/classify
{
  "model": "logisticRegression",
  "data": [
        {
          "tpv name": "PNP",
          "action type": "deposit",
          "affiliate id": "",
          "affiliate name": "",
          "billing country": "IT",
          "billing phone": 393494000000,
          "billing street": "Vía piave albese con Casano como italia",
          "billing user country": "IT",
          "card bin": "",
          "card brand": "MASTERCARD",
          "card fullname": "",
          "card last": 1504,
          "card sub brand": 79,
          "created at": "2021-05-19T09:37:12+00:00",
          "details url": "",
          "device id": "f375af1c25033e5f6f30bf4088ee12b20dff7b59",
          "email": "yisel.leyvapinon@gmail.com",
          "email domain": "gmail.com",
          "fraud score": 0.1,
          "fraud state": "APPROVE",
          "id": 17,
          "ip": "176.200.69.38",
          "ip geo city": "",
          "ip geo code": "US",
          "ip geo point lat": 25.7259,
          "ip geo point lon": -80.4036,
          "ip geo region": "FL",
          "ip geo timezone": "America/New_York",
          "merchant country": "ES",
          "merchant created at": "2021-04-13T10:03:53+00:00",
          "merchant id": 1,
          "order memo": "",
          "password hash": "f09384e0104ffc8e676c9ca2a8769c11a45aaf97a84594d89e1d6f7d38057904",
          "payment mode": "card_credit",
          "phone number": 393494000000,
          "same card operations": "",
          "same card operations day": "",
          "same card operations fraud": "",
          "same card shared ip": "",
          "same card shared ip day": "",
          "same card shared ip fraud": "",
          "same card shared user": "",
          "same card shared user day": "",
          "same device fingerprint": "",
          "same device fingerprint fraud": "",
          "same ip operations": "",
          "same ip operations day": "",
          "same ip operations fraud": "",
          "same user operations day": 2,
          "same user operations previus ko": 1,
          "same user operations previus ok": "",
          "session id": "ccb5d120d8ffb747c71c6bc29f639f1f",
          "tpp error reason": "Id:27 Desc:Awaiting review. GrayListBeneficiaryDoc NeedAction:false Action:null, ",
          "tpp type": "external",
          "tpv alert indicator": 3,
          "tpv card3d s2 partition": 5,
          "tpv card aut3 d secure2 method": "",
          "tpv card dcc": "",
          "tpv card same ip country": 1,
          "tpv id": "160344891-001",
          "tpv secure payment": "",
          "transaction amount": 80,
          "transaction currency": "EUR",
          "transaction id": 614586000000,
          "transaction type": "CHARGE_USER_CARDS",
          "updated at": "2021-06-17T02:30:46+00:00",
          "user account status": "KYC3_FULL_ALLOW",
          "user agent browser family": "CHROME",
          "user agent device family": "IPHONE",
          "user agent os family": "IOS",
          "user balance": 0.7,
          "user city": "Albese con Cassano ",
          "user country": "IT",
          "user created": "2021-02-24T09:17:56+00:00",
          "user fullname": "Yisel  Leyva Piñon",
          "user id": "2dddc720-7681-11eb-9e21-539a0b41ae48",
          "user name": "Yisel ",
          "user street": "Vía piave albese con Casano como italia",
          "user verification level": 4,
          "user zip": 22032,
          "email valid": "",
          "fraud score email": "",
          "fraud score ip": "",
          "fraud score phone": "",
          "ip proxy public": "",
          "ip proxy web": "",
          "ip tor": "",
          "ip vpn": "",
          "phone country": "none",
          "phone valid": "",
          "bank": "",
          "card hash": "a6b280dc6a68f9b39bb9366a1342ef1f",
          "user category": "WEB",
          "transaction state": "CHARGED",
          "fraud score tpp": 0.1,
          "user type": "PERSON",
          "transaction date": "2021-05-19T09:34:27+00:00"
        }
    ]
}

```
Response:
```json

```