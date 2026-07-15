import logging
from datetime import datetime
from pytz import timezone
import os
import sys

"""
def get_logger(log_path, name):
    
    logging.Formatter.converter = lambda *args: datetime.now(tz=timezone('Asia/Taipei')).timetuple()
    # log
    os.makedirs(log_path, exist_ok=True)
    file_handler = logging.FileHandler(filename=f"{log_path}/{name}.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    date = '%(asctime)s %(levelname)s: %(message)s'
    logging.basicConfig(level=logging.INFO, format=date, handlers=handlers)
    logging.info("Start logging " + name)

    return logging
"""
def get_logger(log_path, name):
    logging.Formatter.converter = lambda *args: datetime.now(tz=timezone('Asia/Taipei')).timetuple()

    logger = logging.getLogger(name)
    if logger.hasHandlers():  
        logger.handlers.clear()

    logger.propagate = False

    os.makedirs(log_path, exist_ok=True)
    file_handler = logging.FileHandler(filename=f"{log_path}/{name}.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]

    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(logging.INFO)
    logger.info("Start logging " + name)

    return logger

def get_name(args, modality, model_name=""):
    
    #trainName = f"{args.train_dataset}_{modality}_{args.conv}_train_T{args.train_T}_S{args.model_S}_K{args.numSample}_{model_name}"
    #testName  = f"{args.train_dataset}_to_{args.test_dataset}_{modality}_{args.conv}_test_T{args.test_T}_S{args.model_S}_K{args.numSample}_{model_name}"
    
    trainName = f"{args.train_dataset}_{modality}_{args.conv}_train_T{args.train_T}_S{args.model_S}_{model_name}"
    testName  = f"{args.train_dataset}_to_{args.test_dataset}_{modality}_{args.conv}_test_T{args.test_T}_S{args.model_S}_{model_name}"
    #finetuneName = None
    
    """if finetune:
        
        finetune_dataset = args.finetune_dataset
        test_dataset = args.test_dataset

                    
        finetuneName = f"{args.train_dataset}_finetune_{finetune_dataset}_{args.conv}_train_T{args.train_T}_delta_T_{args.delta_T}_S{args.model_S}_K{args.numSample}_{model_name}"
        testName = f"{args.train_dataset}_finetune_{finetune_dataset}_to_{test_dataset}_{args.conv}_test_T{args.test_T}_delta_T_{args.delta_T}_S{args.model_S}_K{args.numSample}_{model_name}
    """
        
    
    #testName += f"_MB{args.MB_size}_std{args.weight_std}"
    
    return trainName, testName#, finetuneName