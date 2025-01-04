import logging
import re
import os
from datetime import datetime

import pandas as pd


class Logger(object):
    """
    A custom logger using logging module
    """
    def __init__(self):
        self._setup_logger()
        self.container = list()
        
    def _setup_logger(self):
        if not os.path.exists('./logs/'):
            os.makedirs('./logs/')
        FORMAT = '%(asctime)s %(message)s'
        FILENAME = f'./logs/{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.log'
        logging.basicConfig(
            level=logging.INFO, filename=FILENAME, filemode='w', format=FORMAT
        )
        
    def add(self, scaler: float, name):
        message = f'{name}: {scaler:.5f}'
        self.container.append(message)
        
    def update(self, desc=None):
        if desc is not None:
            self.container.insert(0, desc)
        message = ', '.join(self.container)
        logging.info(message)
        self.container = list()  # reset
        

def parse_string(string: str) -> dict[str, float]:
    """
    Default format: YYYY-mm-dd HH:MM:SS,fff Epoch [xxx/xxx], key: value, ...
    Default setting: Parse data iff string start with Epoch [xxx/xxx]
    """
    parser = dict()
        
    desc = re.search(r'Epoch \[(\d+)/\d+\]', string)
    if desc:
        parser['epoch'] = int(desc.group(1))
        
        timestamp = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})', string)
        if timestamp:
            parser['timestamp'] = datetime.strptime(timestamp.group(1), "%Y-%m-%d %H:%M:%S,%f")
            
        matches = re.findall(r'(\w+): (\d+\.\d+)', string)
        if matches:
            parser.update({key: float(value) for key, value in matches})
        
    return parser


def parse_log(log_file: str) -> pd.DataFrame:
    """
    Parse log file as a DataFrame table
    """
    with open(log_file, 'r') as file:
        content = file.readlines()
    
    df = pd.DataFrame()
    for line in content:
        parser = parse_string(line)
        if bool(parser):  # check if parser is empty
            df = pd.concat([df, pd.DataFrame([parser])], ignore_index=True)
    
    return df