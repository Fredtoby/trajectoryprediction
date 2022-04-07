#taken from this StackOverflow answer: https://stackoverflow.com/a/39225039
import requests
import shutil
import subprocess
import os

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    print("Downloading file from google drive...")
    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def download_legacy(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    print("Downloading file from google drive...")
    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination) 


if __name__ == "__main__":
    if(not os.path.isdir("resources")):
        file_id = '1RKcJf6oD3qoVBmZLtIXuLFdAtRKcJDCc'
        destination = 'resources.tar'	
        download_file_from_google_drive(file_id, destination)
        subprocess.call(["tar", "-xvf", "resources.tar"])
        subprocess.call(["rm", "resources.tar"])
        # download_file_from_google_drive(argo_id, "argo.tar")
        # download_file_from_google_drive(lyft_id, "lyft.tar")
        # download_file_from_google_drive(apol_id, "apol.tar")
        # subprocess.call(["tar", "-zxvf", "argo.tar", "-C", "./resources/data/ARGO/"])
        # subprocess.call(["rm", "argo.tar"])
        # subprocess.call(["tar", "-zxvf", "lyft.tar", "-C", "./resources/data/LYFT/"])
        # subprocess.call(["rm", "lyft.tar"])
        # subprocess.call(["tar", "-zxvf", "apol.tar", "-C", "./resources/data/APOL/"])
        # subprocess.call(["rm", "apol.tar"])

        # download_file_from_google_drive(train_id, "forecasting_train.tar.gz")
        # download_file_from_google_drive(val_id, "forecasting_val.tar.gz")
        # download_file_from_google_drive(test_id, "forecasting_test.tar.gz")
        # subprocess.call(["tar", "-zxvf", "forecasting_train.tar.gz", "-C", "./resources/raw_data/ARGO/"])
        # subprocess.call(["rm", "forecasting_train.tar.gz"])
        # subprocess.call(["tar", "-zxvf", "forecasting_val.tar.gz", "-C", "./resources/raw_data/ARGO/"])
        # subprocess.call(["rm", "forecasting_val.tar.gz"])
        # subprocess.call(["tar", "-zxvf", "forecasting_test.tar.gz", "-C", "./resources/raw_data/ARGO/"])
        # subprocess.call(["rm", "forecasting_test.tar.gz"])
    else:
        print("resources folder already exists. Not downloading")



    # print("installing mask r-cnn")
    # subprocess.call(["pip", "install", "-r", "model/Detection/Mask/requirements.txt"])
    # os.chdir("model/Detection/Mask")
    # subprocess.call(["python3", "setup.py", "install"])
