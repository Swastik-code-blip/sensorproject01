import logging
import os
from datetime import datetime


Log_FILE=f'{datetime.now().strftime('%m_%d_%Y_%H_%M_%s')}.log'

logs_path=os.path.join(os.getcwd(),"logs",Log_FILE)

os.makedirs(logs_path,exist_ok=True)

Log_FILE_PATH=os.path.join(logs_path,Log_FILE)

logging.basicConfig(
    filename=Log_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(leavelname)s - %(message)s",
    leavel=logging.INFO
)