# import torch
#
#
# def distance_matrix(t1, t2):
#     """
#     :param t1: B x C x N
#     :param t2: B x C x N
#     :return d_matrix: B x N x N
#     """
#
#     d_matrix = 2 - 2 * torch.bmm(t1.transpose(1, 2), t2)  # [0, 4]
#     d_matrix = d_matrix.clamp(min=1e-8, max=4.0)
#     d_matrix = torch.sqrt(d_matrix)  # [0, 2]
#
#     return d_matrix


# print("Pred min distance index:", pmid)
# print("True min distance index:", tmid)
# print("Pred min distance:", dot_dest[0, index, pmid])
# print("True min distance:", dot_dest[0, index, tmid])
# print("All other distances in row", v)
# print("Their indicies:", ind[:20])
# print("In a short:", s[0, index, pmid])

# print("\n")
# print("EVAL")
#
# count = 0
#
# t1 = []
# t2 = []
# t3 = []
#
# for index in range(0, 1200):
#     pmid = dot_dest[0, index].argmax()
#     tmid = s[0, index].nonzero()
#
#     v, ind = torch.sort(dot_dest[0, index], descending=True)
#
#     if tmid in ind[:9]:
#         t1.append(index)
#         t2.append(pmid)
#
#         t3.append((pmid, tmid))
#         count += 1
#
# print("In a short:", count)
