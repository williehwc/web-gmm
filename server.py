from flask import Flask, request
from flask_restful import Resource, Api, reqparse
from Model import Model

app = Flask(__name__)
api = Api(app)

gmms = []

class NewModel(Resource):
  def post(self):
    # Validation
    parser = reqparse.RequestParser()
    parser.add_argument('maxNumTrainingVectors', type=int, location='json', required=True)
    parser.add_argument('numComponents', type=int, location='json', required=True)
    parser.parse_args()
    # Get data
    data = request.get_json()
    # Make new model
    new_gmm = Model(data['maxNumTrainingVectors'], data['numComponents'])
    gmms.append(new_gmm)
    return {'id': len(gmms) - 1}

class ExistingModel(Resource):
  def put(self, model_id):
    # Validation
    parser = reqparse.RequestParser()
    parser.add_argument('train', type=bool, location='json', required=True)
    parser.add_argument('vectors', type=list, location='json', required=True)
    parser.parse_args()
    # Get data
    data = request.get_json()
    vectors = data['vectors']
    if data['train'] and len(vectors) >= gmms[model_id].get_num_components():
      # Add vectors and train the model
      gmms[model_id].add_training_vectors(vectors)
      gmms[model_id].train()
      return
    else:
      # Just score the vectors
      score = gmms[model_id].score_testing_vectors(vectors)
      return {'score': score}

api.add_resource(NewModel, '/')
api.add_resource(ExistingModel, '/<int:model_id>')

if __name__ == '__main__':
  app.run(host='0.0.0.0', debug=True)