import os
import json
import dill
import joblib


def save_obj(obj, location, file_name, serialize_type="dill", **kwargs):
    """
    serialize object state
    """
    if not os.path.exists(location):
        os.makedirs(location)

    if serialize_type == "joblib":
        with open(os.path.join(location, file_name + ".joblib"), "wb") as save_file:
            joblib.dump(obj, save_file)
    elif serialize_type == "dill":
        with open(os.path.join(location, file_name + ".pickle"), "wb") as save_file:
            dill.dump(obj, save_file)
    elif serialize_type == "json":
        with open(os.path.join(location, file_name + ".json"), "w") as save_file:
            json_encoder = kwargs["cls"] if "cls" in kwargs else None
            json.dump(obj, save_file, cls=json_encoder)


def load_obj(location, file_name, serialize_type="dill", **kwargs):
    """
    deserialize object state
    """

    if serialize_type == "joblib":
        with open(os.path.join(location, file_name + ".joblib"), "rb") as load_file:
            return joblib.load(load_file)
    elif serialize_type == "dill":
        with open(os.path.join(location, file_name + ".pickle"), "rb") as load_file:
            return dill.load(load_file)
    elif serialize_type == "json":
        with open(os.path.join(location, file_name + ".json"), "r") as save_file:
            json_encoder = kwargs["cls"] if "cls" in kwargs else None
            return json.load(save_file, cls=json_encoder)
