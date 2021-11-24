import os
from flask import Flask, request, jsonify, make_response
from server import app
from services.EtlService import EtlService
from services.MlService import MlService
from services.EdaService import EdaService

srvEda = EdaService()

srvEtl = EtlService()
srvEtl.setEda(srvEda)

srvMl = MlService()
srvMl.setEtl(srvEtl)

'''
    Generate Data Mining View
'''
@app.route("/api/lcs/generate", methods=["POST"])
def generate():
    result = srvEtl.generate()
    return jsonify({
        "data": result 
    })

'''
    Generate Model PKL
'''
@app.route("/api/lcs/traing", methods=["POST"])
def traing():
    path = os.path.dirname(__file__) + "../../../../data/"
    filename = request.json.get("modelname", "dataMiningView")
    filename = path + filename + ".csv"
    filename = os.path.abspath(filename)
    model = srvMl.train(filename)

    print('>>> LcsController:traing >>> model', model)
    return jsonify(model)

'''
    Classify Data
'''
@app.route("/api/lcs/classify", methods=["POST"])
def classify():
    path = os.path.dirname(__file__) + "../../../../data/"
    filename = request.json.get("modelname", "classifier_100_data_model")
    filename = path + filename + ".pkl"
    filename = os.path.abspath(filename)
    data = request.json.get("data", 10)

    result = srvMl.classify(filename, int(data))
    return jsonify(result)


