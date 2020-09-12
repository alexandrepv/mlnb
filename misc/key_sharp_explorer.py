import pandas as pd
import numpy as np
import xmltodict
import time

"""
Explorer the contents of Key# code saved in a opus file.
https://github.com/martinblech/xmltodict
"""

fpath = "G:\Downloads\BB1_03.03.20_ML 2.5min straight.opus"

with open(fpath, 'r') as file:
    print('[Started parsing XML]')
    opus_dict = xmltodict.parse(file.read(), process_namespaces=True)
    print(' > Done')

print('List of all Score functions')
functions_list = []
for i, function in enumerate(opus_dict['Opus']['Opus_Body']['Score_Function']):
    print(f" > {function['@Function_ID']}")
    functions_list.append(function['@Function_ID'])

functions_df = pd.DataFrame(columns=['function_id'], data=functions_list)
output_fpath = 'G:\Downloads\keysharp_function_list.csv'
functions_df.sort_values(by='function_id', inplace=True)
functions_df.to_csv(output_fpath, header=False, index=False)


