def calculate_difference_matrix(t1, t2):
    """
    :param t1: B x N1
    :param t2: B x N2
    """
    t1 = t1.unsqueeze(2).float()
    t2 = t2.unsqueeze(1).float()

    diff = t1 - t2

    return diff