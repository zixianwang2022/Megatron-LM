import requests
import json

url = "http://127.0.0.1:5000/api"
headers = {'Content-Type': 'application/json'}

prompt = f"hi"
data = {"prompts": [prompt], "tokens_to_generate": 10, "temperature": 0.5, "top_p": 0.9}
response = requests.put(url, data=json.dumps(data), headers=headers)

print (response)
print (response.json)
print (dir (response))
