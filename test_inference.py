import requests

# Define the endpoint and input file
url = "http://127.0.0.1:8080/invocations"
input_file = "./data/input/mnr_data_050825.csv"

# Read the input file
with open(input_file, 'rb') as f:
    response = requests.post(url, headers={"Content-Type": "text/csv"}, data=f)
    print("Response Status Code:", response.status_code)
    print("Response Headers:", response.headers)
    # print("Response Body:", response.text)
