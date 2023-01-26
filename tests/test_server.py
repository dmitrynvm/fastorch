import requests, json, base64


def test_server():
    url = "http://localhost:8000/predict"
    image_path = "data/tench.jpg"
    encoded = str(base64.b64encode(open(image_path, "rb").read()))
    result = requests.post(url, json={"image": encoded}).json()
    assert result['label'] == 'tench'
