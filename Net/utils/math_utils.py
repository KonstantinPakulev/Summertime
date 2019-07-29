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


# dot = torch.zeros((1, 1, 240, 320))
# dot[0, 0, 120, 160] = 1
# d_dot = dilate_mask(dot)
# neg = d_dot - dot

# ones = torch.ones((1, 1, 240, 320))
# warped = warp_image(ones, homo).gt(0).float()
# w_eroded = erode_mask(warped)

# all = torch.cat((neg, warped, w_eroded, warped - w_eroded), dim=0)

# print(neg.nonzero().shape)

# writer.add_image("image", make_grid(all))

# mask1 = create_bordering_mask(im2, homo)
# mask2 = create_bordering_mask(im1, homo_inv)
#
# _, des1 = model(im1)
# _, des2 = model(im2)
#
# loss1, s1, dot_des1, r_mask1 = criterion(des1, des2, homo, mask1)
# loss2, s2, dot_des2, r_mask2 = criterion(des2, des1, homo_inv, mask2)
#
# loss = (loss1 + loss2) / 2
# loss.backward()


# return {
#     'loss': loss,
#
#     'des1': des1,
#     's1': s1,
#     'dot_des1': dot_des1,
#     'r_mask1': r_mask1,
#
#     'des2': des2,
#     's2': s2,
#     'dot_des2': dot_des2,
#     'r_mask2': r_mask2
# }
# \tValidation loss is {tester.state.metrics['loss']: .4f}
#                 \tNN match score is: {tester.state.metrics['nn_match']: .4f}
