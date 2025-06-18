from data import MNISTDataModule, CIFAR10DataModule, SklearnDataModule, MoonsDataModule
from sklearn.datasets import load_wine, load_iris, load_breast_cancer
from uci_dataset import load_abalone, load_heart_disease, load_credit_approval, \
    load_cylinder_bands, load_dermatology, load_diabetic, load_dishonest_users, load_early_stage_diabetes_risk, \
    load_fertility, load_haberman, load_hayes_roth, load_hcv
import numpy as np
import pandas as pd


def get_dataset(config):
    if config['dataset'] == 'moons':
        data_module = MoonsDataModule(batch_size=config['batch_size'], random_state=config['seed'],
                                      num_workers=config['num_workers'], n_samples=config['n_samples'],
                                      noise=config['noise'])

        data_module.prepare_data()
        data_module.setup()
        num_classes = 1
        in_channels = 2
    elif config['dataset'] == 'mnist':
        data_module = MNISTDataModule(batch_size=config['batch_size'], seed=config['seed'],
                                      num_workers=config['num_workers'])
        in_channels = 1
        num_classes = 10
        data_module.prepare_data()
        data_module.setup()
    elif config['dataset'] == 'cifar10':
        data_module = CIFAR10DataModule(batch_size=config['batch_size'], seed=config['seed'],
                                        num_workers=config['num_workers'])
        num_classes = 10
        data_module.prepare_data()
        data_module.setup()
    elif config['dataset'] == 'wine':
        X, y = load_wine(return_X_y=True)
        data_module = SklearnDataModule(X, y, batch_size=config['batch_size'],
                                        num_workers=config['num_workers'])
        num_classes = 3
        in_channels = X.shape[1]
    elif config['dataset'] == 'iris':
        X, y = load_iris(return_X_y=True)
        data_module = SklearnDataModule(X, y, batch_size=config['batch_size'],
                                        num_workers=config['num_workers'])
        num_classes = 3
        in_channels = X.shape[1]
    elif config['dataset'] == 'credit_approval':
        df = load_credit_approval()
        X = df.drop(columns=['A16'])
        y = df['A16']
        y = (y == '+').astype(int)
        X = pd.get_dummies(X, columns=X.select_dtypes(object).columns).astype(float)
        X, y = X.values, y.values
        data_module = SklearnDataModule(X, y, batch_size=config['batch_size'],
                                        num_workers=config['num_workers'])
        num_classes = 2
        in_channels = X.shape[1]


    elif config['dataset'] == 'breast_cancer':
        X, y = load_breast_cancer(return_X_y=True)
        data_module = SklearnDataModule(X, y, batch_size=config['batch_size'],
                                        num_workers=config['num_workers'])
        num_classes = 2
        in_channels = X.shape[1]

    elif config['dataset'] == 'abalone':
        df = load_abalone()
        df['Sex'] = (df['Sex'] == 'M').astype(int)
        X, y = df.drop(columns=['Rings']).values, df['Rings'].values
        data_module = SklearnDataModule(X, y, batch_size=config['batch_size'],
                                        num_workers=config['num_workers'])
        num_classes = len(np.unique(y))
        in_channels = X.shape[1]

    elif config['dataset'] == 'cylinder_bands':
        df = load_cylinder_bands()

        X = df.drop(columns=['band type'])
        y = df['band type']
        y = (y == 'band').astype(int)
        X = pd.get_dummies(X, columns=X.select_dtypes(object).columns).astype(float).fillna(0)
        X, y = X.values, y.values
        data_module = SklearnDataModule(X, y, batch_size=config['batch_size'],
                                        num_workers=config['num_workers'])
        num_classes = len(np.unique(y))
        in_channels = X.shape[1]
    elif config['dataset'] == 'dermatology':
        df = load_dermatology()

        X = df.drop(columns=['class'])
        classes = {1: 'psoriasis', 2: 'seboreic dermatitis',
                   3: 'lichen planus', 4: 'pityriasis rosea',
                   5: 'cronic dermatitis', 6: 'pityriasis rubra pilaris'}
        classes = {v: k - 1 for k, v in classes.items()}
        y = df['class'].replace(classes)
        X = pd.get_dummies(X, columns=X.select_dtypes(object).columns).astype(float).fillna(0)
        X, y = X.values, y.values
        data_module = SklearnDataModule(X, y, batch_size=config['batch_size'],
                                        num_workers=config['num_workers'])
        num_classes = len(np.unique(y))
        in_channels = X.shape[1]
    elif config['dataset'] == 'heart_disease':
        df = load_heart_disease()
        X, y = df.drop(columns=['target']).values, df['target'].values
        data_module = SklearnDataModule(X, y, batch_size=config['batch_size'],
                                        num_workers=config['num_workers'])
        num_classes = len(np.unique(y))
        in_channels = X.shape[1]

    elif config['dataset'] == 'diabetic':
        df = load_diabetic()

        X = df.drop(columns=['Class'])
        y = df['Class'].astype(int)
        X = pd.get_dummies(X, columns=X.select_dtypes(object).columns).astype(float).fillna(0)
        X, y = X.values, y.values
        data_module = SklearnDataModule(X, y, batch_size=config['batch_size'],
                                        num_workers=config['num_workers'])
        num_classes = len(np.unique(y))
        in_channels = X.shape[1]
    elif config['dataset'] == 'dishonest_users':
        df = load_dishonest_users()

        X = df.drop(columns=['untrustworthy'])
        y = (df['untrustworthy'] == 'untrustworthy').astype(int)
        X = pd.get_dummies(X, columns=X.select_dtypes(object).columns).astype(float).fillna(0)
        X, y = X.values, y.values
        data_module = SklearnDataModule(X, y, batch_size=config['batch_size'],
                                        num_workers=config['num_workers'], val_split=0.21)
        num_classes = len(np.unique(y))
        in_channels = X.shape[1]

    elif config['dataset'] == 'early_stage_diabetes_risk':
        df = load_early_stage_diabetes_risk()

        X = df.drop(columns=['class'])
        y = (df['class'] == 'Positive').astype(int)
        X = pd.get_dummies(X, columns=X.select_dtypes(object).columns).astype(float).fillna(0)
        X, y = X.values, y.values
        data_module = SklearnDataModule(X, y, batch_size=config['batch_size'],
                                        num_workers=config['num_workers'])
        num_classes = len(np.unique(y))
        in_channels = X.shape[1]

    elif config['dataset'] == 'fertility':
        df = load_fertility()

        X = df.drop(columns=['Diagnosis'])
        y = (df['Diagnosis'] == 'O').astype(int)
        X = pd.get_dummies(X, columns=X.select_dtypes(object).columns).astype(float).fillna(0)
        X, y = X.values, y.values
        data_module = SklearnDataModule(X, y, batch_size=config['batch_size'],
                                        num_workers=config['num_workers'])
        num_classes = len(np.unique(y))
        in_channels = X.shape[1]

    elif config['dataset'] == 'haberman':
        df = load_haberman()

        X = df.drop(columns=['survival'])
        y = df['survival'] - 1
        X = pd.get_dummies(X, columns=X.select_dtypes(object).columns).astype(float).fillna(0)
        X, y = X.values, y.values
        data_module = SklearnDataModule(X, y, batch_size=config['batch_size'],
                                        num_workers=config['num_workers'])
        num_classes = len(np.unique(y))
        in_channels = X.shape[1]

    elif config['dataset'] == 'hayes_roth':
        df = load_hayes_roth()

        X = df.drop(columns=['class', 'name'])
        y = (df['class']).astype(int) - 1
        X = pd.get_dummies(X, columns=X.select_dtypes(object).columns).astype(float).fillna(0)
        X, y = X.values, y.values
        data_module = SklearnDataModule(X, y, batch_size=config['batch_size'],
                                        num_workers=config['num_workers'])
        num_classes = len(np.unique(y))
        in_channels = X.shape[1]

    elif config['dataset'] == 'hcv':
        df = load_hcv()

        X = df.drop(columns=['Unnamed: 0', 'Category'])
        y = df['Category'].replace({'0=Blood Donor': 0, '0s=suspect Blood Donor': 1, '1=Hepatitis': 2, '2=Fibrosis': 3,
                                    '3=Cirrhosis': 4})
        X = pd.get_dummies(X, columns=X.select_dtypes(object).columns).astype(float).fillna(0)
        X, y = X.values, y.values
        data_module = SklearnDataModule(X, y, batch_size=config['batch_size'],
                                        num_workers=config['num_workers'])
        num_classes = len(np.unique(y))
        in_channels = X.shape[1]


    else:
        raise ValueError(f'Unknown dataset {config["dataset"]}')

    return data_module, num_classes, in_channels
