from fastapi import FastAPI,HTTPException, UploadFile, File, Response
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import io
from fastapi.responses import StreamingResponse
import datetime
import pickle
curren_date = datetime.datetime.now()
current_year = curren_date.year

import warnings
warnings.filterwarnings('ignore')

from joblib import dump, load

app = FastAPI()

class Item(BaseModel):
    name: str
    year: int
#    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


def obrabdates(old_df, type_df='new'):
    # работа с дубликатами
    columns_diplicates = ['name', 'year',
                          'km_driven', 'fuel',
                          'seller_type', 'transmission',
                          'owner', 'mileage', 'engine',
                          'max_power', 'torque', 'seats']
    df_train_f = old_df.drop_duplicates(subset=columns_diplicates, keep='first')
    df_train_f = df_train_f.reset_index(drop=True)

    # обработка признаков

    df_train_f['mileage'] = df_train_f['mileage'].str.extract(r'(\d+.\d+|\d+)')
    df_train_f['engine'] = df_train_f['engine'].str.extract(r'(\d+.\d+|\d+)')
    df_train_f['max_power'] = df_train_f['max_power'].str.extract(r'(\d+.\d+|\d+)')

    # type dates
    df_train_f['mileage'] = df_train_f['mileage'].astype(float)

    df_train_f.max_power = df_train_f.max_power.astype(str)
    df_train_f.max_power = df_train_f.max_power.str.replace(r'[^0-9.]+', '', regex=True)
    df_train_f['max_power'] = pd.to_numeric(df_train_f['max_power'], errors='coerce')
    df_train_f['max_power'] = df_train_f['max_power'].astype(float)

    df_train_f['engine'] = df_train_f['engine'].astype(float)

    # заполенение пустых значений медианой
    med_mileage = df_train_f.mileage.median()
    med_engine = df_train_f.engine.median()
    med_max_power = df_train_f.max_power.median()
    med_seats = df_train_f.seats.median()

    # заполнение пропусков в трейн и тест
    df_train_f.mileage = df_train_f.mileage.fillna(med_mileage)
    df_train_f.engine = df_train_f.engine.fillna(med_engine)
    df_train_f.max_power = df_train_f.max_power.fillna(med_max_power)
    df_train_f.seats = df_train_f.seats.fillna(med_seats)

    # del torque
    df_train_f = df_train_f.drop('torque', axis=1)

    # типы данных
    df_train_f['engine'] = df_train_f['engine'].astype(int)
    df_train_f['seats'] = df_train_f['seats'].astype(int)
    df_train_f['km_driven'] = df_train_f['km_driven'].astype(int)

    # доп.фичи
    df_train_f['mean_km'] = df_train_f.groupby('fuel')['km_driven'].transform('mean')

    df_train_f['year_of_car'] = df_train_f['year'].apply(lambda x: current_year - x)

    col_kv = ['year**2', 'km_driven**2', 'mileage**2', 'engine**2', 'max_power**2']
    col_ne_cv = ['year', 'km_driven', 'mileage', 'engine', 'max_power']

    df_train_f[col_kv] = df_train_f[col_ne_cv].apply(lambda x: x ** 2)
    df_train_f[col_kv] = df_train_f[col_kv].apply(lambda x: round(x, 2))

    df_train_f['avg_KM_of_year'] = df_train_f.km_driven / df_train_f.year_of_car
    df_train_f['avg_KM_of_year'] = df_train_f.avg_KM_of_year.apply(lambda x: round(x, 2))

    # поиск целевой переменной

    X_Final_train = df_train_f[['year', 'km_driven', 'fuel', 'seller_type',
                                'transmission', 'owner', 'mileage', 'engine', 'max_power',
                                'seats', 'mean_km', 'year_of_car',
                                'year**2', 'km_driven**2', 'mileage**2', 'engine**2',
                                'max_power**2', 'avg_KM_of_year']]

    return X_Final_train


# Работа с приизнаками категориальными
def onehotencod(X_st):
    X_cat = X_st[['fuel', 'seller_type', 'transmission', 'owner', 'seats']]
    onehotencod_load = load('onegotencod.pkl')
    encod_df_array = onehotencod_load.transform(X_cat).toarray()
    encod_df = pd.DataFrame(encod_df_array,
                            columns=onehotencod_load.get_feature_names_out(X_cat.columns)).reset_index(drop=True)
    X_final = X_st.drop(X_cat, axis=1).reset_index(drop=True)
    X_endcod = pd.concat([X_final, encod_df], axis=1)

    return X_endcod

# Стандартизация
def standard(x):
    scaler_l = load('scale.pkl')
    X_ohe_scale = scaler_l.transform(x)
    X_ohe_scale = pd.DataFrame(X_ohe_scale, columns=x.columns)

    return X_ohe_scale

@app.post("/predict_item")
async def predict_item(item: Item) -> float:
    df_1= pd.DataFrame([item.__dict__])
    X_1 = obrabdates(df_1, type_df='test')
    X_1_ohe = onehotencod(X_1)
    X_1_ohe_scale = standard(X_1_ohe)
    model = load('grid_search_final.pkl')
    y_1 = model.predict(X_1_ohe_scale)
    y_1 = np.exp(y_1)
    return y_1[0][0]

@app.post("/predict_items")
async def predict_items(file: UploadFile):
    df_1 = pd.read_csv(file.file, dtype={
        'name' : str,
        'year' : int,
        'km_driven' : int,
        'fuel' : str,
        'seller_type' : str,
        'transmission' : str,
        'owner' : str,
        'mileage' : str,
        'engine' : str,
        'max_power' : str,
        'torque' : str,
        'seats' : float })
    X_1 = obrabdates(df_1, type_df='test')
    X_1_ohe = onehotencod(X_1)
    X_1_ohe_scale = standard(X_1_ohe)
    model_R = load('grid_search_final.pkl')
    y_1_pred =model_R.predict(X_1_ohe_scale)
    y_1_pred = pd.DataFrame(y_1_pred, columns=['prediction'])
    y_1_pred = np.exp(y_1_pred)
    final_df = pd.concat([df_1, y_1_pred], axis=1)

    stream = io.StringIO()
    final_df.to_csv(stream, index=False)
    resp = StreamingResponse(
        iter([stream.getvalue()]), media_type='text/csv')
    resp.headers["Content-Disposition"] = "attachment; filename=export.csv"


    return resp
