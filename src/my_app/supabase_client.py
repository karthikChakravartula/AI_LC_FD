import os
import requests
from dotenv import load_dotenv 
#from pathlib import Path

# 1. Dynamically locate the folder where THIS file lives
#current_dir = Path(__file__).resolve().parent.parent.parent

def get_token():
    load_dotenv()#dotenv_path=current_dir / ".env")
    return {"API_KEY": os.getenv("API_KEY"), "BASE_URL": os.getenv("DATABASE_URL")}

def fetch_records(table_name, select_fields, top=50) :
    keys = get_token()
    
    url = f"{keys["BASE_URL"]}{table_name}?select={select_fields}&order=name.asc"
    #print(url)
    try:
        start = 0
        pagelength = top
        mainData = []
        while True:
            end = start + pagelength - 1
            #print(f"Start : {start}, End : {end}")
            headers = {
                    "apikey":keys["API_KEY"],
                    "Accept": "application/json",
                    "range":f"{start}-{end}"
                }
            response = requests.get(url,headers = headers)
            response.raise_for_status()
            data = response.json()
            #print(data)
            #print("---------------------------------------------------------------------")
            if not data:
                print("Breaking the Loop")
                break
            mainData.extend(data)
            start += pagelength
        #print(mainData)
        #print(len(mainData))
        return mainData   
    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")

fetch_records("accounts","name,description")
