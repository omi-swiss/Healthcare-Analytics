
import pandas as pd

def preprocess_data(df):
    # Drop unnecessary columns
    cols_to_drop = ['HOSPITAL_CODE', 'PATIENTID', 'ADMISSION_DATE', 'DISCHARGE_DATE']
    df.drop(cols_to_drop, axis=1, inplace=True)

    # Set index
    df.set_index('CASE_ID', inplace=True)

    # Convert data types
    num_columns = ['AVAILABLE_EXTRA_ROOMS_IN_HOSPITAL', 'VISITORS_WITH_PATIENT', 'ADMISSION_DEPOSIT', 'LENGTH_OF_STAY']

    for column in df.columns:
        if column in num_columns:
            df[column] = df[column].astype(int)
        else:
            df[column] = df[column].astype('object')

    # One-hot encode categorical columns
    df_after_process = pd.get_dummies(df)

    return df_after_process
