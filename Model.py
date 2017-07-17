from sklearn import mixture
import numpy

class Model:
  def __init__(self, max_num_training_vectors, num_components):
    self.max_num_training_vectors = max_num_training_vectors
    self.gmm = mixture.GaussianMixture(n_components=num_components, warm_start=True)
    self.training_vectors = []
  
  def add_training_vectors(self, vectors):
    self.training_vectors.extend(vectors)
    while len(self.training_vectors) > self.max_num_training_vectors:
      self.training_vectors.pop(0)
  
  def train(self):
    self.gmm.fit(self.training_vectors)
  
  def score_testing_vectors(self, vectors):
    scores = self.gmm.score_samples(vectors)
    return numpy.mean(scores)
  
  def get_num_components(self):
    return self.gmm.n_components