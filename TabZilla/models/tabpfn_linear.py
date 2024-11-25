from mothernet.prediction.tabpfn import TabPFNClassifier
from models.basemodel import BaseModel
import os, pdb
import numpy as np
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
torch.set_num_threads(1)

class TabLinearModel(BaseModel):
    def __init__(self, params, args):
        super().__init__(params, args)

        if args.objective == "regression":
            raise NotImplementedError("Does not support")
        elif args.objective in ["classification", 'binary']:
            model_tabpfn = TabPFNClassifier(
                device='cuda', 
                N_ensemble_configurations=32, 
                model_string = 'prior_diff_real_checkpoint_n_0',
                epoch = '100',
            )

            self.model = TabPFNClassifier(
                device='cuda', 
                model_string = f'tabpfn_modelnaive_linear_attention_validdatanew_11_24_2024_22_30_15',
                N_ensemble_configurations=1,
                epoch = 'on_exit', 
            )

            X = np.random.randn(100, 100)
            y = np.random.randint(0, 10, (100,))
            model_tabpfn.fit(X, y)
            self.model.fit(X, y)

            with torch.no_grad():
                for (name_target, target_param), (name_source, source_param) in zip(self.model.model.named_parameters(), model_tabpfn.model.named_parameters()):
                    try:
                        target_param.copy_(source_param)
                    except:
                        print(f"Failed to copy parameter {name_target} from {name_source}")

        self.max_n_training_samples = args.max_n_training_samples

    def fit(self, X, y, X_val=None, y_val=None):
        #WARNING: When overwrite_warning is true, TabPFN will attempt to run on arbitrarily large datasets! This means if you run TabPFN on a large dataset without sketching/sampling it may crash rather than issuing a warning and refusing to run
        if X.shape[0] > self.max_n_training_samples:
            # select indices to have as balanced a dataset as possible
            classes = np.unique(y)
            selected_indices = []
            for c in classes:
                indices = np.where(y == c)[0]
                selected_indices.extend(np.random.choice(indices, self.max_n_training_samples // len(classes), replace=True))
            self.model.fit(X[selected_indices,:], y[selected_indices], overwrite_warning=True)
        else:
            self.model.fit(X, y, overwrite_warning=True)
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

