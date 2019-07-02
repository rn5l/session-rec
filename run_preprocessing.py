import time

'''
preprocessing method ["info","org","days_test","slice"]
    info: just load and show info
    org: from gru4rec (last day => test set)
    org_min_date: from gru4rec (last day => test set) but from a minimal date onwards
    days_test: adapted from gru4rec (last N days => test set)
    slice: new (create multiple train-test-combinations with a window approach  
    buys: load buys and safe file to prepared
'''
import sys
from pathlib import Path
import yaml
import importlib
import traceback
import os

def main( conf ): 
    '''
    Execute experiments for the given configuration path
        --------
        conf: string
            Configuration path. Can be a single file or a folder.
        out: string
            Output folder path for endless run listening for new configurations. 
    '''
    print( 'Checking {}'.format( conf ) )
    
    file = Path( conf )
    if file.is_file():
        
        print( 'Loading file' )
        stream = open( str(file) )
        c = yaml.load(stream)
        stream.close()
        print( 'processing config ' + conf )
        
        try:
        
            run_file( c )
            print( 'finished config ' + conf )
            
        except (KeyboardInterrupt, SystemExit):
                        
            print( 'manually aborted config ' + conf )            
            raise
        
        except Exception:
            print( 'error for config ', file )
            traceback.print_exc()
            
        exit()
    
    print( 'File not found: ' + conf )
    
def run_file( conf ):
    
    #include preprocessing
    preprocessor = load_preprocessor( conf )
    
    #load data from raw and transform
    data = preprocessor.load_data( conf['data']['folder'] + conf['data']['prefix'] )
    if type(data) == tuple:
        extra = data[1:]
        data = data[0]
    data = preprocessor.filter_data( data, **conf['filter'] )
    
    ensure_dir( conf['output']['folder'] + conf['data']['prefix'] )
    #call method according to config
    if conf['type'] == 'single':
        preprocessor.split_data( data, conf['output']['folder'] + conf['data']['prefix'], **conf['params']  )
    elif conf['type'] == 'window':
        preprocessor.slice_data( data, conf['output']['folder'] + conf['data']['prefix'], **conf['params']  )
    else:
        if hasattr(preprocessor, conf['type']):
            method_to_call = getattr(preprocessor, conf['type'])
            method_to_call( data, path+file, **conf['params']  )
        else:
            print( 'preprocessing type not supported' )
    

def load_preprocessor( conf ):
    '''
    Load the proprocessing module
        --------
        conf : conf
            Just the last part of the path, e.g., evaluation_last
    '''
    return importlib.import_module( 'preprocessing.preprocess_' + conf['preprocessor'] )

def ensure_dir(file_path):
    '''
    Create all directories in the file_path if non-existent.
        --------
        file_path : string
            Path to the a file
    '''
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == '__main__':
    '''
    Run the preprocessing configured above.
    '''
    
    if len( sys.argv ) == 2: 
        main( sys.argv[1] )
    else:
        print( 'Preprocessing configuration expected.' )
    