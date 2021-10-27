import numpy as np

from sklearn.preprocessing import StandardScaler

from .data_class import PVTrainDataSet, PVTestDataSet


class AbstractPreprocessor:

    def preprocess_for_train(self, train_data: PVTrainDataSet, **kwarg) -> PVTrainDataSet:
        raise NotImplementedError

    def preprocess_for_test_input(self, test_data: PVTestDataSet) -> PVTestDataSet:
        raise NotImplementedError

    def postprocess_for_prediction(self, predict: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class IdentityPreprocessor(AbstractPreprocessor):

    def __init__(self):
        super(IdentityPreprocessor, self).__init__()

    def preprocess_for_train(self, train_data: PVTrainDataSet, **kwarg) -> PVTrainDataSet:
        return train_data

    def preprocess_for_test_input(self, test_data: PVTestDataSet) -> PVTestDataSet:
        return test_data

    def postprocess_for_prediction(self, predict: np.ndarray) -> np.ndarray:
        return predict


class ScaleAllPreprocessor(AbstractPreprocessor):
    treatment_scaler: StandardScaler
    outcome_scaler: StandardScaler

    def __init__(self):
        super(ScaleAllPreprocessor, self).__init__()

    def preprocess_for_train(self, train_data: PVTrainDataSet, **kwarg) -> PVTrainDataSet:
        treatment_proxy_scaler = StandardScaler()
        treatment_proxy_s = treatment_proxy_scaler.fit_transform(train_data.treatment_proxy)

        self.treatment_scaler = StandardScaler()
        treatment_s = self.treatment_scaler.fit_transform(train_data.treatment)

        self.outcome_scaler = StandardScaler()
        outcome_s = self.outcome_scaler.fit_transform(train_data.outcome)

        outcome_proxy_scaler = StandardScaler()
        outcome_proxy_s = outcome_proxy_scaler.fit_transform(train_data.outcome_proxy)

        backdoor_s = None
        if train_data.backdoor is not None:
            backdoor_scaler = StandardScaler()
            backdoor_s = backdoor_scaler.fit_transform(train_data.backdoor)

        return PVTrainDataSet(treatment=treatment_s,
                              treatment_proxy=treatment_proxy_s,
                              outcome_proxy=outcome_proxy_s,
                              outcome=outcome_s,
                              backdoor=backdoor_s)

    def preprocess_for_test_input(self, test_data: PVTestDataSet) -> PVTestDataSet:
        return PVTestDataSet(treatment=self.treatment_scaler.transform(test_data.treatment),
                             structural=test_data.structural)

    def postprocess_for_prediction(self, predict: np.ndarray) -> np.ndarray:
        return self.outcome_scaler.inverse_transform(predict)


def get_preprocessor_ate(id: str) -> AbstractPreprocessor:

    if id == "ScaleAll":
        return ScaleAllPreprocessor()
    elif id == "Identity":
        return IdentityPreprocessor()
    else:
        raise KeyError(f"{id} is invalid name for preprocessing")
