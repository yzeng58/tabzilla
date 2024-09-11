import torch
import numpy as np
from hyperfast import HyperFastClassifier
from models.basemodel import BaseModel
torch.set_num_threads(1)

class HyperFastModel(BaseModel):
    def __init__(self, params, args):
        super().__init__(params, args)

        if args.objective == "regression":
            raise NotImplementedError("Does not support")
        elif args.objective in ["classification", 'binary']:
            self.model = HyperFastClassifier(
                device='cuda'
            )
            
    def fit(self, X, y, X_val=None, y_val=None):
        self.model.fit(X, y)
        return [], []

    def predict_helper(self, X):
        return self.model.predict(X)
    
    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = dict()
        return params

    @classmethod
    def get_random_parameters(cls, seed: int):
        params = dict()
        return params
    
    @classmethod
    def default_parameters(cls):
        params = dict()
        return params
    
    def get_classes(self):
        return self.model.classes_

