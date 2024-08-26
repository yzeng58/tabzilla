from mothernet.prediction.tabpfn import TabPFNClassifier
from models.basemodel import BaseModel
import os, pdb
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class TabFastModel(BaseModel):
    def __init__(self, params, args):
        super().__init__(params, args)

        if args.objective == "regression":
            raise NotImplementedError("Does not support")
        elif args.objective in ["classification", 'binary']:
            self.model = TabPFNClassifier(
                device='cuda', 
                model_string = f'ssm_tabpfn_b4_largedatasetTrue_modellinear_attention_nsamples50000_08_01_2024_22_05_50',
                N_ensemble_configurations=3,
                epoch = '110',
            )

        self.max_n_training_samples = args.max_n_training_samples

    def fit(self, X, y, X_val=None, y_val=None):
        #WARNING: When overwrite_warning is true, TabPFN will attempt to run on arbitrarily large datasets! This means if you run TabPFN on a large dataset without sketching/sampling it may crash rather than issuing a warning and refusing to run
        self.model.fit(X[:self.max_n_training_samples,:], y[:self.max_n_training_samples], overwrite_warning=True)
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

