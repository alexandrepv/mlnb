import re
import datetime as dt

"""
This script reads my entire logbook and plots my working times throughout the days
"""

logbook_fpath = r"C:\Users\alexandre.vicente\Downloads\logbook.txt"
with open(logbook_fpath, 'r') as file:
    file_content = file.read()
    line_list = re.split('\n', file_content)
    line_list = [item for item in line_list if len(item) > 0]

    g = 0

