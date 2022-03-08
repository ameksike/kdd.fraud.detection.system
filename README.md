# TropiPay Fraud Detection System 

## algorithms
- Light GBM
- Random Forest
- Logistic Regression

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
          "tpv_name": "PNP",
          "action_type": "deposit",
          "affiliate_id": "",
          "affiliate_name": "",
          "billing_country": "IT",
          "billing_phone": 393494000000,
          "billing_street": "Vía piave albese con Casano como italia",
          "billing_user_country": "IT",
          "card_bin": "",
          "card_brand": "MASTERCARD",
          "card_fullname": "",
          "card_last": 1504,
          "card_sub_brand": 79,
          "created_at": "2021-05-19T09:37:12+00:00",
          "details_url": "",
          "device_id": "f375af1c25033e5f6f30bf4088ee12b20dff7b59",
          "email": "yisel.leyvapinon@gmail.com",
          "email_domain": "gmail.com",
          "fraud_score": 0.1,
          "fraud_state": "APPROVE",
          "id": 17,
          "ip": "176.200.69.38",
          "ip_geo_city": "",
          "ip_geo_code": "US",
          "ip_geo_point_lat": 25.7259,
          "ip_geo_point_lon": -80.4036,
          "ip_geo_region": "FL",
          "ip_geo_timezone": "America/New_York",
          "merchant_country": "ES",
          "merchant_created_at": "2021-04-13T10:03:53+00:00",
          "merchant_id": 1,
          "order_memo": "",
          "password_hash": "f09384e0104ffc8e676c9ca2a8769c11a45aaf97a84594d89e1d6f7d38057904",
          "payment_mode": "card_credit",
          "phone_number": 393494000000,
          "same_card_operations": "",
          "same_card_operations_day": "",
          "same_card_operations_fraud": "",
          "same_card_shared_ip": "",
          "same_card_shared_ip_day": "",
          "same_card_shared_ip_fraud": "",
          "same_card_shared_user": "",
          "same_card_shared_user_day": "",
          "same_device_fingerprint": "",
          "same_device_fingerprint_fraud": "",
          "same_ip_operations": "",
          "same_ip_operations_day": "",
          "same_ip_operations_fraud": "",
          "same_user_operations_day": 2,
          "same_user_operations_previus_ko": 1,
          "same_user_operations_previus_ok": "",
          "session_id": "ccb5d120d8ffb747c71c6bc29f639f1f",
          "tpp_error_reason": "Id:27 Desc:Awaiting review. GrayListBeneficiaryDoc NeedAction:false Action:null, ",
          "tpp_type": "external",
          "tpv_alert_indicator": 3,
          "tpv_card3d_s2_partition": 5,
          "tpv_card_aut3_d_secure2_method": "",
          "tpv_card_dcc": "",
          "tpv_card_same_ip_country": 1,
          "tpv_id": "160344891-001",
          "tpv_secure_payment": "",
          "transaction_amount": 80,
          "transaction_currency": "EUR",
          "transaction_id": 614586000000,
          "transaction_type": "CHARGE_USER_CARDS",
          "updated_at": "2021-06-17T02:30:46+00:00",
          "user_account_status": "KYC3_FULL_ALLOW",
          "user_agent_browser_family": "CHROME",
          "user_agent_device_family": "IPHONE",
          "user_agent_os_family": "IOS",
          "user_balance": 0.7,
          "user_city": "Albese con Cassano ",
          "user_country": "IT",
          "user_created": "2021-02-24T09:17:56+00:00",
          "user_fullname": "Yisel  Leyva Piñon",
          "user_id": "2dddc720-7681-11eb-9e21-539a0b41ae48",
          "user_name": "Yisel ",
          "user_street": "Vía piave albese con Casano como italia",
          "user_verification_level": 4,
          "user_zip": 22032,
          "email_valid": "",
          "fraud_score_email": "",
          "fraud_score_ip": "",
          "fraud_score_phone": "",
          "ip_proxy_public": "",
          "ip_proxy_web": "",
          "ip_tor": "",
          "ip_vpn": "",
          "phone_country": "none",
          "phone_valid": "",
          "bank": "",
          "card_hash": "a6b280dc6a68f9b39bb9366a1342ef1f",
          "user_category": "WEB",
          "transaction_state": "CHARGED",
          "fraud_score_tpp": 0.1,
          "user_type": "PERSON",
          "transaction_date": "2021-05-19T09:34:27+00:00"
        }
    ]
}

```
Response:
```json

```