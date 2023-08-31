'''
Created on Jul 27, 2023

@author: politti
'''
from cwatm.management_modules.data_handling import cbinding, load_json, load_scikit_model, CWATMFileError
from glob import glob
import os
import pandas as pd

class StatsModels(object):
    '''
    Handles the loading of statistical models from saved files and the creation of the data needed to feed the models
    and create predictions during the cwatm execution.
    '''


    def __init__(self, model):
        '''
        Constructor
        '''
        self.model = model
        self.var = model.var
        
        #TODO: debug add delta to ET_delta
        setattr(model.var, 'delta_ET', None)
    
    def load_stats_models(self):
        '''
        Loads all the scikit statistical models and precomputes correction factors where necessary
        '''
        
        config_file = cbinding('PathStatsModels')
        nameall = glob(os.path.normpath(config_file))
        if not nameall:
            msg = f"Error 215: In StatsModels.load_models, cannot find {config_file} file "
            raise CWATMFileError(config_file, msg, sname='PathStatsModels')
        config = nameall[0]        
        
        ml_configs = load_json(config)
        ml_models = ml_configs['ml_models']
        
        
        for m in ml_models:
            #load the ml model
            model_file = m['ml_file']
            m_name = m['name']
            print(f'Loading {model_file} for {m_name}...')
            m['ml_model'] = load_scikit_model(model_file)
            print(f'{model_file} loaded for {m_name}')
            
            pred_vars = m['predictors']
            pred_vars_dict = {} #create dictionary to store predictors variables data at each iteration
            for p in pred_vars:
                pred_vars_dict[p] = None
                
            m['pred_vars'] = pred_vars_dict
            
            #self.var[m_name] = m
            setattr(self.model, m_name, m)
        
    def scikit_predict(self, scikit_model, data_dict):
        '''
        Applies a scikit prediction model using 2d shaped predictors.
        @param scikit_model, scikit regression model: trained sci kit model (must have function predict).
        @param data_dict, dict: keys are the name of the predictors, values the predictors' data. Predictors
                                names must match the names used as predictor variables when the scikit model
                                was trained.
        @returns numpy array of the predicted variable.
        '''
        shape = list(data_dict.values())[0].shape
        #convert 2d arrays to 1d 
        for pred_var, data in data_dict.items():
            data_dict[pred_var] = data.flatten()
        
        df = pd.DataFrame(data_dict)
        pred = scikit_model.predict(df)
        
        #reconvert to 2d array
        pred = pred.reshape(shape)
        
        return pred
        
    def calculate_x_factor(self, scikit_model, data_dict, cwatm_var):
        '''
        Applies a scikit prediction model using 2d shaped predictors and computes a correction factor to be applied 
        to the cwatm variable
        @param scikit_model, scikit regression model: trained sci kit model (must have function predict).
        @param data_dict, dict: keys are the name of the predictors, values the predictors' data. Predictors
                                names must match the names used as predictor variables when the scikit model
                                was trained.        
        @param cwatm_var, array: variable simulated by cwatm
        @param array: correction factor for cwatm variable.
        
        '''
        pred = self.scikit_predict(scikit_model, data_dict)        
        
        self.var.delta_ET = pred
        x = 1 - (pred/cwatm_var)

        return x
    
        
        
        
        
        
        
        
        
        
        
        
        
                
        
        
        
        
