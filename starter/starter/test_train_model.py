import pandas as pd
import yaml 

with open('config.yaml','rb') as yml:
    config = yaml.safe_load(yml)

df = pd.read_csv(config['main']['data_pth'])

def test_extract_data(df):
    try:
        df.shape[0]>0
        df.shape[1]>0
    except:
        print('Error')