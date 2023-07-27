'''
Created on Jul 27, 2023

@author: politti
'''
from cwatm.management_modules.data_handling import cbinding, load_json, load_scikit_model, CWATMFileError
from glob import glob
import os

class StatsModels(object):
    '''
    Hanldes the loading of statistical models from saved files and the creation of the data needed to feed the models
    and create predictions during the cwatm execution.
    '''


    def __init__(self, model):
        '''
        Constructor
        '''
        self.model = model
        self.var = model.var
    
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
            m['ml_model'] = load_scikit_model(model_file)
            print(f'{model_file} loaded for {m_name}')
            
            pred_vars = m['predictors']
            pred_vars_dict = {} #create dictionary to store predictors variables data at each iteration
            for p in pred_vars:
                pred_vars_dict[p] = None
                
            m['pred_vars'] = pred_vars_dict
            
            self.var[m_name] = m
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
                
        
        
        
        
