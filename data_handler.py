import requests
import pandas as pd


def pull_data_from_db():
    """
    Pulls data from online UCI databases and collates them
    """
    base_data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/'
    # Pulling processed data as it is in the most usable format but still needs to be cleaned (missing data)
    options = [
        'processed.cleveland.data',
        'processed.hungarian.data',
        'processed.switzerland.data',
        'processed.va.data',
    ]
    columns = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalanch', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'
    ]

    # This chunk parses the online data into float arrays where it can and adds it to all_data together
    all_data = []
    for o in options:
        req = requests.get(base_data_url + o)
        if req.status_code != 200:
            raise requests.exceptions.ConnectionError(f'Data URL {base_data_url + o} invalid. Please check the URL is valid.')
        content = [y.split(',') for y in req.content.decode('utf-8').split('\n') if len(y) > 1]
        for i in range(len(content)):
            for j in range(len(content[i])):
                if content[i][j] == '?':
                    continue
                content[i][j] = float(content[i][j])
        all_data.extend(content)

    # Can use pandas dataframes if we want
    df = pd.DataFrame(all_data, columns=columns)
    return all_data, df


def get_data():
    """
    Primary function. Handles:
    - pulling data from web
    - parsing data into a usable format
    - cleaning data with ML SVM in mind
    - returning a feature: label set
    """
    # Gets data in an array form (all_data) and Pandas DataFrame form (df)
    all_data, df = pull_data_from_db()



if __name__ == '__main__':
    get_data()
