import time
import re
import datetime as dt

file = open("times.txt", 'r')

days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

lunch_duration = dt.timedelta(minutes=45)
workday_duration = dt.timedelta(hours=8)

file_content = file.read()
line_list = re.split('\n', file_content)
line_list = [item for item in line_list if len(item) > 0]

assert len(line_list) == len(days), '[ERROR] You need a list of 5 arrival/leaving times on the .txt file'

for line, day in zip(line_list, days):
    line_matches = re.search(r'(.*):(.*) - (.*):(.*)', line)

    # Start time
    start_timedelta = dt.timedelta(hours=int(line_matches.group(1)),
                                   minutes=int(line_matches.group(2)))
    # Leaving time
    leaving_timedelta = dt.timedelta(hours=int(line_matches.group(3)),
                                     minutes=int(line_matches.group(4)))

    total_duration = leaving_timedelta - start_timedelta
    working_duration = total_duration - lunch_duration

    if working_duration > workday_duration:
        print(f'[{day}] Working time: {workday_duration}, Extra time: {working_duration - workday_duration}')
    else:
        print(f'[{day}] Working time: {working_duration}')


print("Lunch and breaks already subtracted")