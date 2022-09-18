import pandas as pd
from src import config
from sklearn.preprocessing import LabelEncoder
import joblib

def feature_process(data):
    df = pd.read_csv(data)
    print(df.head(20))

    #Feature Engineering

    #Imputation
    df.society.fillna("No Society", inplace=True)
    df.bhk.fillna(0, inplace=True)
    df.bath.fillna(1, inplace=True)
    df.balcony.fillna(0,inplace=True)

    #Labelencoding categorical variables
    labelencoders = {}

    for c in ['area_type', 'availability', 'location', 'society']:
        labelencoders[c] = LabelEncoder()
        df[c] = labelencoders[c].fit_transform(df[c])

    df.to_csv(config.TRAIN_PROCESSED_DATA,index=False)
    joblib.dump(labelencoders,config.MODELS_PATH+'feature_encoders.pkl')

    df.info()

    print('Constructing Prediction')
    print('Almost There')


feature_process(config.TRAIN_DATA)













