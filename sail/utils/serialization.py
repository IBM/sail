import os
import dill
import joblib


def save_obj(obj, location, file_name, serialize_type="dill"):
    """
    serialize object state
    """
    if not os.path.exists(location):
        os.makedirs(location)

    if serialize_type == "joblib":
        with open(os.path.join(location, file_name + ".joblib"), "wb") as save_file:
            joblib.dump(obj, save_file)
    else:
        with open(os.path.join(location, file_name + ".pickle"), "wb") as save_file:
            dill.dump(obj, save_file)


def load_obj(location, file_name, serialize_type="dill"):
    """
    deserialize object state
    """

    if serialize_type == "joblib":
        with open(os.path.join(location, file_name + ".joblib"), "rb") as load_file:
            return joblib.load(load_file)
    else:
        with open(os.path.join(location, file_name + ".pickle"), "rb") as load_file:
            return dill.load(load_file)
