import time


def gct(f="l"):
    """
    get current time
    :param f: "l" for log, "f" for file name
    :return: formatted time
    """
    if f == "l":
        return time.strftime("%m-%d %H:%M:%S", time.localtime(time.time()))
    elif f == "f":
        return f'{time.strftime("%m_%d_%H_%M", time.localtime(time.time()))}'


def pretty_dict(d, indent=0):
    for key, value in d.items():
        if isinstance(value, dict):
            print("\t" * indent + f"{key}")
            pretty_dict(value, indent + 1)
        else:
            print("\t" * indent + f"{key:>18} : {value}")