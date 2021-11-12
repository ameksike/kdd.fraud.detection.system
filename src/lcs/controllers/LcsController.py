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

@app.route("/api/lcs/generate", methods=["POST"])
def generate():
    result = srvEtl.generate()
    return jsonify({
        "data": result 
    })

@app.route("/api/lcs/traing", methods=["POST"])
def traing():
    path = os.path.dirname(__file__) + "../../../../data/"
    filename = request.json.get("modelname", "sample_data_100")
    filename = path + filename + ".csv"
    filename = os.path.abspath(filename)

    cross_validation_scores, cross_validation_score_mean, accuracy_score, precision_score, recall_score, f1_score, classifer_name = \
        srvMl.train(filename)

    payload = {
        "cross_validation_scores": cross_validation_scores.tolist(),
        "cross_validation_score_mean": cross_validation_score_mean,
        "accuracy_score": accuracy_score,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "f1_score": f1_score,
        "classifer_name": classifer_name
    }
    return jsonify(payload)

@app.route("/api/lcs/classify", methods=["POST"])
def classify():
    path = os.path.dirname(__file__) + "../../../../data/"
    filename = request.json.get("modelname", "classifier_100_data_model")
    filename = path + filename + ".pkl"
    filename = os.path.abspath(filename)
    data = request.json.get("data", 10)

    result = srvMl.classify(filename, int(data))
    return jsonify(result)


