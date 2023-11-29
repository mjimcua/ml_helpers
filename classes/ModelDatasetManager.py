from abc import ABC, abstractmethod
import pathlib
import pandas as pd
import json
from classes.LoggerManager import LoggerManager


class ModelDatasetManager(ABC):

    def __init__(self, model_id, random_state, log_level='DEBUG'):

        self.model_id = model_id
        self.random_state = random_state
        self.test_size_ratio = 0.3
        self.datasets = []
        self.log_level = log_level

        log_manager = LoggerManager(self.log_level)
        self.logger = log_manager.get_logger_configured()

        assert isinstance(random_state, int), 'Choose a integer random_state variable'

        self.model_root = '_' + model_id
        self.binaries_root = self.model_root+'/binaries/'
        self.pipeline_root = self.model_root+'/pipeline_snapshots/'

        pathlib.Path(self.model_root).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.binaries_root).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.pipeline_root).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_datasets_steps():
        return [
            'raw',
            'raw_with_label',
            'inference',
            'inference_with_label',
            'train',
            'test',
            'train_cleaned',
            'test_cleaned',
            'inference_cleaned'
        ]

    @abstractmethod
    def save_dataset(self, step, df):
        pass

    @abstractmethod
    def load_dataset(self, step):
        pass

    # RAW DATASET MANAGEMENT
    # -----------------------------------------------------------

    @abstractmethod
    def generate_raw_dataset(self, **kwargs):
        """
        This method must be overridden
        :abstract
        """
        pass

    @abstractmethod
    def generate_inference_dataset(self, **kwargs):
        """
        This method must be overridden
        :abstract
        """
        pass

    @abstractmethod
    def set_prediction_label(self, **kwargs):
        """
        This method must be overridden
        :abstract
        """
        pass

    @abstractmethod
    def generate_train_test_datasets(self):
        """
        Cada modelo tiene su propia política de separación de test y train para evitar data leaks o entrenar
        de forma temporalmente correcta
        """
        pass

    # DATA CLEANING: Identifying and correcting mistakes or errors in the data
    # ----------------------------------------------------------------------------------------

    @abstractmethod
    def data_cleaning__delete_columns(self, df):
        """
        This method must be overridden
        :abstract
        """
        pass

    @abstractmethod
    def data_cleaning__fix_datatypes(self, **kwargs):
        """
        This method must be overridden
        :abstract
        """
        pass

    @abstractmethod
    def data_cleaning__null_normalization(self, **kwargs):
        """
        This method must be overridden
        :abstract
        """
        pass

    @abstractmethod
    def data_cleaning__get_insight_from_train_for_cleaning(self, **kwargs):
        """
        This method must be overridden
        :abstract
        """
        pass

    @abstractmethod
    def data_cleaning__missing_values_management__imputation(self, **kwargs):
        """
        This method must be overridden
        :abstract
        """
        pass

    @abstractmethod
    def data_cleaning__missing_values_management__delete_rows(self, **kwargs):
        """
        This method must be overridden
        :abstract
        """
        pass

    @abstractmethod
    def data_cleaning__filter_by_business_domain(self, **kwargs):
        """
        This method must be overridden
        :abstract
        """
        pass

    @abstractmethod
    def generate_cleaned_dataset(self, **kwargs):
        """
        This method must be overridden
        :abstract
        """
        pass

    def generate_cleaned_dataset(self, step):
        """
        Clean dataset
        :param df: pandas dataframe
        :param step : train_cleaned, test_cleaned, production_cleaned
        :return: pandas dataframe with data cleaned
        """

        df = self.load_dataset(step)
        df = self.data_cleaning__fix_datatypes(df)
        df = self.data_cleaning__null_normalization(df)
        df = self.data_cleaning__missing_values_management__imputation(df)
        df = self.data_cleaning__missing_values_management__delete_rows(df)
        df = self.data_cleaning__filter_by_business_domain(df)

        if step == 'inference_with_label':
            step = 'inference'
        self.save_dataset(step+'_cleaned', df)

        setattr(self, 'dataset_'+step+'_cleaned', df)

    # FEATURE SELECTION: Identifying those imput variables that are most relevant to task
    # ----------------------------------------------------------------------------------------

    # DATA TRANSFORMS: Changing the scale or distribution of variables
    # ----------------------------------------------------------------------------------------

    # FEATURE ENGINEERING: Deriving new variables from available data
    # ----------------------------------------------------------------------------------------

    # DIMENSIONALITY REDUCTION: Creating compact projections of the data
    # ----------------------------------------------------------------------------------------

    def check_dataset_reproducibility(self, samples=5):
        self.logger.warning('Cuidado, esto puede consumir muchos recursos')

        # https://www.geeksforgeeks.org/pandas-find-the-difference-between-two-dataframes/
        # https://kanoki.org/2019/07/04/pandas-difference-between-two-dataframes/
        for i in range(0, samples):
            #self.generate_historic_dataset()
            df = self.load_dataset('raw')
            self.set_prediction_label(df)
            self.generate_train_test_datasets()
            self.datasets.append(self.load_dataset('train'))

        # df = pd.concat(self.datasets)
        # df = df.reset_index(drop=True)
        # df_group = df.groupby(['CUSTOMER_ID', 'ORDER_ID', 'date_model_timestamp_order'])
        # idx = [x[0] for x in df_group.groups.values() if len(x) > (len(self.datasets) - 1)]
        # df.reindex(idx)
        # print('Check Method 1: total rows equals between test samples')
        # print('Total test rows'.format(len(self.datasets[0])))
        # print('Total test rows equals between test samples'.format(len(df)))
        empty_df = pd.concat(self.datasets).drop_duplicates(keep=False)
        print('Check Method: dataframe must be empty (0 rows) because all rows are duplicated')
        print(f'{len(empty_df)} rows' )

    @staticmethod
    def get_nulls_ordered(df):
        return df.isna().sum().sort_values(ascending=False).pipe(lambda x: x[x > 0])

    @staticmethod
    def get_raw_dataset_statistics(self):
        self.load_dataset('raw')
        print('Raw dataset statistics:'
              '\n-----------------------------------'
              '\nTotal rows: {}'
              '\nTotal columns: {}'.format(self.dataset_raw.shape[0], self.dataset_raw.shape[1]))

    def get_label_balance(self):
        self.logger.info('Get Prediction Label Balance')

        if self.dataset_raw_with_label is None:

            self.load_dataset('raw_with_label')

        assert isinstance(self.dataset_raw_with_label, pd.DataFrame), 'ERROR: debe cargarse ne memoria un dataframe'

        label_balance = json.loads(self.dataset_raw_with_label.label.value_counts(normalize=True).to_json())

        print('''
            RAW SUMMARY
            --------------------------------------------------------------------
            - Label balance positive class = {}
            - Total Columns = {}
            - Total rows = {}
            '''.format(
            label_balance['1'],
            self.dataset_raw_with_label.shape[1],
            self.dataset_raw_with_label.shape[0],
        )
        )
        return label_balance['1']

    def set_test_size_ratio(self, ratio=0.3):
        self.test_size_ratio = ratio

    def get_test_size_ratio(self):
        return self.test_size_ratio




