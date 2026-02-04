from abc import ABC, abstractmethod


class Pipeline(ABC):
    """
    Abstract class implementing an anonymization pipeline.
    """
    @abstractmethod
    def __init__(self, config: dict, force_compute: bool = False, devices: list = [0]):
        pass

    @abstractmethod
    def run_anonymization_pipeline(self, datasets):
        """
            Runs the anonymization algorithm on the given datasets.

            Args:
                datasets (dict of str -> Path): The datasets on which the
                    anonymization pipeline should be runned on. These dataset
                    will be processed sequentially.
        """
        pass
