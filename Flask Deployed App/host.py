import requests 
resp = requests.post("http://237.84.2.178:5000/predict", files={"file": open("corn_ healthy.png", "rb")})

print(resp.json())