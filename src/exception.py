import sys
from src.logger import logging as lg

def error_message_detail(error, error_detail=sys):
    try:
        if error_detail is sys:
            _, _, exc_tb = sys.exc_info()
        else:
            _, _, exc_tb = error_detail  # Assume tuple from sys.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        error_message = "error occurred python script name [{0}] line number [{1}] error message [{2}]".format(
            file_name, line_number, str(error)
        )
        lg.error(error_message)
        return error_message
    except AttributeError:
        error_message = f"Error processing exception: {str(error)}"
        lg.error(error_message)
        return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail=sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message