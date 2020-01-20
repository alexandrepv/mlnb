import requests
from lxml import etree

cookies = {'ASP.NET_SessionId': 'aqzninock2rjpi2rxy3pgszp'}

r = requests.post('https://dnae.timesheetportal.com/Timesheets/Default.aspx?vid=0&date=04-11-2019', cookies=cookies)

assert r.status_code == 200, f"[ERROR] HTTP request was not successful! Return code {r.status_code}"
html = etree.HTML(r.text)

g = 0