import pandas as pd
import psycopg2
import os
from sklearn.model_selection import GroupShuffleSplit
import torch
from monai.transforms import (LoadImaged, EnsureChannelFirstD)

class CAGDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.config = config
        self._construct_loader()
                
    def getDataFromDatabase(self, config):
        connection = psycopg2.connect(
            host=config['host'],
            database=config['database'],
            user=config['username'],
            password=config['password'])
        sql = config['query'].replace(
            "?table_name", "\"" + config['table_name'] + "\"")
        sql = sql.replace(
            "?schema_name", "\"" + config['schema_name'] + "\"")
        sql = sql.replace(
            "??", "\"")
        df = pd.read_sql_query(sql, connection)
        if len(df) == 0:
            print('The requested query does not have any data!')
        connection.close()
        return df

    def set_data_path(self, features):
        for feature in features:
            self.df[feature] = self.df[feature].apply(
                        lambda x: os.path.join(self.config['DataSetPath'], x))

    def get_input_features(self, csv, features='DcmPathFlatten'):
        if features == 'DcmPathFlatten':
            features = [col for col in
                        csv.columns.tolist() if col.startswith(features)]
        else:
            features = features
        return features
    

    def groupEntriesPrPatient(self, df):
        '''Grouping entries pr patients'''
       # X = df.drop(self.config['labels_names'], 1)
       # y = df[self.config['labels_names']]
        if self.config['TestSize'] == 1:
            return None, df
        else:
            gs = GroupShuffleSplit(
                n_splits=2,
                test_size=self.config['TestSize'],
                random_state=0)
            train_ix, val_ix = next(
                gs.split(df, groups=df['PatientID']))
            self.df_train = df.iloc[train_ix]
            self.df_val = df.iloc[val_ix]
          #  self.addPhase(df_train, df_val)
           # return df_train, df_val

    def _construct_loader(self):
        """
        Construct the video loader.
        """

        self.df = self.getDataFromDatabase(self.config)
        self.features = self.get_input_features(self.df)
        self.set_data_path(self.features)

        self._path_to_videos = self.df["DcmPathFlatten"].tolist()
        self._labels = self.df["labels"].tolist()
        self._spatial_temporal_idx = [0] * len(self.df)
        self.groupEntriesPrPatient(self.df)
        
        print('Constructing cag dataset')
        

    def __transforms__(self, frames):
        self.transforms = [
                LoadImaged(keys=self.features),
                EnsureChannelFirstD(keys=self.features),
                self.resampleORresize(),
                self.maybeDeleteMeta(),
                self.getMaybePad(),
                self.getCopy1to3Channels(),
                ScaleIntensityd(keys=self.features),
                self.maybeNormalize(),
                EnsureTyped(keys=self.features, data_type='tensor'),
                self.maybeToGpu(self.features),
                self.maybeCenterCrop(self.features),
                ConcatItemsd(keys=self.features, name='inputs'),
                self.maybeDeleteFeatures(),
                ]

        self.transforms = Compose(self.transforms, log_stats=True)
        self.transforms.set_random_state(seed=0)
        return self.transforms
    def __len__(self):
        return len(self.df_train)
    
    def __getitem__(self, index):
        
        
        return frames, torch.tensor(label_list)
    
    