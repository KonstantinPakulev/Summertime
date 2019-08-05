import torch


def print_dict(d, indent=0):
    for key, value in d.items():
        if isinstance(value, dict):
            print("\t" * indent + f"{key}")
            print_dict(value, indent + 1)
        else:
            print("\t" * indent + f"{key:>18} : {value}")
