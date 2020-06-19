##Environment Setup
###Installing pip
####Windows
```
python get-pip.py
pip --version
python -m pip install --upgrade pip (Upgrade pip version)
```
####Linux
```
sudo apt update
sudo apt install python3-pip (Installing pip for Python 3)
sudo apt install python-pip (Installing pip for Python 2)
pip3 --version
```
###Virtualenv
*If you are using Python 2, replace **venv** with **virtualenv** in Creating a virtual environment.*
####On Windows
```
py -m pip install --user virtualenv (Installing)
py -m venv env (Creating)
.\env\Scripts\activate (Activating)
deactivate (Leaving)
```
####On Linux and macOS
```
python3 -m pip install --user virtualenv (Installing)
python3 -m venv env (Creating)
source env/bin/activate (Activating)
deactivate (Leaving)
```

##Install prerequisite libraries
```
pip install -r requirements.txt
``` 

##Running the web app
```
streamlit run iris_classification.py
```
