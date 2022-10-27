from flask import Flask, request, jsonify

from stud.implementation import build_model_123, build_model_23, build_model_3

app = Flask(__name__)
model_123 = build_model_123("cpu")
model_23 = build_model_23("cpu")
model_3 = build_model_3("cpu")


@app.route("/", defaults={"path": ""}, methods=["POST", "GET"])
@app.route("/<path:path>", methods=["POST", "GET"])
def annotate(path):

    try:
        sentences = request.json["sentences"]
        predictions_123 = model_123.predict(sentences)
        predictions_23 = model_23.predict(sentences)
        predictions_3 = model_3.predict(sentences)

    except Exception as e:

        app.logger.error(e, exc_info=True)
        return {
            "error": "Bad request",
            "message": "There was an error processing the request. Please check logs/server.stderr",
        }, 400

    return jsonify(
        sentences=sentences,
        predictions_123=predictions_123,
        predictions_23=predictions_23,
        predictions_3=predictions_3,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=12345)
