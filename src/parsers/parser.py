from abc import ABC, abstractmethod
import pandas as pd


class DatasetParser(ABC):
    """ Interface to define a parser for datasets."""

    @abstractmethod
    def load_med1(self) -> pd.DataFrame:
        """ Load data of 1st medication administration during the 1st admimission of a patient """
        ...

    @abstractmethod
    def load_med2(self) -> pd.DataFrame:
        """ Load data of 2nd medication administration during the 1st admimission of a patient """
        ...

    @abstractmethod
    def load_lab(self, h_med_adm1: pd.DataFrame, h_med_adm2: pd.DataFrame) -> pd.DataFrame:
        """ Load data on all lab tests taken during the 1st admimission of a patient """
        ...

    @abstractmethod
    def parse(self, use_pairs: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """ Load med1, med2 and lab test data """
        ...