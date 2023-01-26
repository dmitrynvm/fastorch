python3 -m venv env
source env/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt
cd data
python3 resnet.py
python3 mobile.py
