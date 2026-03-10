import logging
import os
from datetime import datetime

class LoggerFactory:    
    _loggers = {} 
    
    @staticmethod
    def get_logger(name, log_dir='logs', level=logging.INFO):
        if name in LoggerFactory._loggers:
            return LoggerFactory._loggers[name]
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = os.path.join(log_dir, f"{name}_{timestamp}.log")
        
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        if logger.hasHandlers():
            logger.handlers.clear()
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler = logging.FileHandler(log_filename, mode='a')
        file_handler.setLevel(logging.DEBUG)  
        file_handler.setFormatter(formatter)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        logger.propagate = False
        
        logger.info(f"Logger '{name}' initialized. Log file: {log_filename}")
        
        LoggerFactory._loggers[name] = logger
        
        return logger

def get_preprocessing_logger():
    return LoggerFactory.get_logger('preprocessing')

def get_encoder_training_logger():
    return LoggerFactory.get_logger('encoder_training')

def get_final_training_logger():
    return LoggerFactory.get_logger('final_training')

def get_evaluation_logger():
    return LoggerFactory.get_logger('evaluation')