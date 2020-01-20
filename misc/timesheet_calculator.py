import time
import re
import datetime as dt

file = open(r"C:\Users\alexandre.vicente\Downloads\times.txt", 'r', encoding="utf8")

days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

lunch_duration = dt.timedelta(minutes=45)
barda_limit = dt.timedelta(hours=2)

file_content = file.read()
line_list = re.split('\n', file_content)
line_list = [item for item in line_list if len(item) > 0]

assert len(line_list) == len(days), '[ERROR] You need a list of 5 arrival/leaving times on the .txt file'

for line, day in zip(line_list, days):
    line_matches = re.search(r'(.*):(.*)-(.*):(.*)', line)

    # Start time
    start_timedelta = dt.timedelta(hours=int(line_matches.group(1)),
                                   minutes=int(line_matches.group(2)))
    # Leaving time
    leaving_timedelta = dt.timedelta(hours=int(line_matches.group(3)),
                                     minutes=int(line_matches.group(4)))

    total_duration = leaving_timedelta - start_timedelta
    working_duration = total_duration - lunch_duration

    if working_duration > barda_limit:
        print(f'[{day}] Barda Time: {barda_limit}, Extra time: {working_duration - barda_limit}')
    else:
        print(f'[{day}] Barda Time: {barda_limit}')


print("Lunch and breaks already subtracted")