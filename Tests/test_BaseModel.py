import pickle

import pytest

from savageml.models import BaseModel

EXPECTED_PARAMS = {"test_param": False}


def test_set_params_valid_params():
    model = BaseModel()
    model.set_params(**EXPECTED_PARAMS)
    for param, value in EXPECTED_PARAMS.items():
        assert getattr(model, param) == value


def test_set_params_invalid_params():
    invalid_params = {"test_param_bool": True}
    model = BaseModel()
    with pytest.raises(ValueError) as errorInfo:
        model.set_params(**invalid_params)
        assert "Invalid parameter" in str(errorInfo.value)


def test_get_params_expected_params_present():
    model = BaseModel()
    retrieved_params = model.get_params()
    for param in EXPECTED_PARAMS:
        assert param in retrieved_params


def test_get_params_retrieved_params_expected():
    model = BaseModel()
    retrieved_params = model.get_params()
    for retrieved_param in retrieved_params:
        assert retrieved_param in EXPECTED_PARAMS


def test_clone_has_all_original_params():
    model = BaseModel()
    model.set_params(**EXPECTED_PARAMS)
    clonedModel = model.clone()
    modelParams = model.get_params()
    cloneParams = clonedModel.get_params()
    for param, value in modelParams.items():
        assert param in cloneParams
        assert cloneParams[param] == value


def test_clone_only_has_original_params():
    model = BaseModel()
    model.set_params(**EXPECTED_PARAMS)
    clonedModel = model.clone()
    modelParams = model.get_params()
    cloneParams = clonedModel.get_params()
    for param, value in cloneParams.items():
        assert param in modelParams
        assert modelParams[param] == value


def test_clone_is_not_reference():
    model = BaseModel()
    model.set_params(**EXPECTED_PARAMS)
    clonedModel = model.clone()
    assert clonedModel is not model


def test_model_can_be_pickled():
    model = BaseModel(**EXPECTED_PARAMS)
    data = pickle.dumps(model)
    new_model = pickle.loads(data)
    assert new_model.test_param is False
