import os

import torch
from torchvision.utils import make_grid

import Net.source.datasets.dataset_utils as du
import Net.source.nn.net.utils.endpoint_utils as eu
import Net.source.nn.net.utils.criterion_utils as cu
import Net.source.utils.metric_utils as meu

from Net.source.core import experiment as exp

from Net.source.utils.vis_utils import draw_cv_matches
from Net.source.utils.metric_utils import repeatability_score, match_score


"""
Tensorboard logging functions
"""


def plot_losses_tensorboard(writer, data_engine, state_engine, tag):
    for key, value in data_engine.state.metrics.items():
        if cu.LOSS in key and value is not None:
            writer.add_scalar(f"{tag}/{key}", value, state_engine.state.iteration)


def plot_metrics_tensorboard(writer, data_engine, state_engine, tag):
    for key, value in data_engine.state.metrics.items():
        if cu.LOSS not in key and value is not None:
            writer.add_scalar(f"{tag}/{key}", value, state_engine.state.iteration)


def plot_scores(writer, state_engine, data_engine, keys, normalize=False):
    batch, endpoint = data_engine.state.output

    image1_name, image2_name = batch.get(du.IMAGE1_NAME), batch.get(du.IMAGE2_NAME)

    s1, s2 = endpoint[keys[0]], endpoint[keys[1]]

    if normalize:
        s1 = s1 / s1.max()
        s2 = s2 / s2.max()

    image1_name = image1_name[0]
    image2_name = image2_name[0]

    s = make_grid(torch.cat((s1, s2), dim=0))

    writer.add_image(f"{keys[0]} and {keys[1]} of {image1_name} and {image2_name}", s, state_engine.state.epoch)


def plot_kp_matches(writer, state_engine, data_engine, px_thresh):
    batch, endpoint = data_engine.state.output

    image1, image2 = batch.get(du.IMAGE1), batch.get(du.IMAGE2)
    image1_name, image2_name = batch.get(du.IMAGE1_NAME), batch.get(du.IMAGE2_NAME)
    kp1, kp2 = endpoint[eu.KP1], endpoint[eu.KP2]
    w_kp1, w_kp2 = endpoint[eu.W_KP1], endpoint[eu.W_KP2]
    w_vis_kp1_mask, w_vis_kp2_mask = endpoint[eu.W_VIS_KP1_MASK], endpoint[eu.W_VIS_KP2_MASK]

    image1_name = image1_name[0]
    image2_name = image2_name[0]

    _, _, _, nn_kp_ids, match_mask = repeatability_score(kp1, kp2, w_kp1, w_kp2, w_vis_kp1_mask, w_vis_kp2_mask,
                                                         px_thresh, True)

    cv_keypoints_matches = draw_cv_matches(image1, image2, kp1, kp2, nn_kp_ids, match_mask[0])

    writer.add_image(f"{image1_name} and {image2_name} keypoints matches", cv_keypoints_matches,
                     state_engine.state.epoch, dataformats='HWC')


def plot_desc_matches(writer, state_engine, data_engine, px_thresh, dd_measure):
    batch, endpoint = data_engine.state.output

    image1, image2 = batch.get(du.IMAGE1), batch.get(du.IMAGE2)
    image1_name, image2_name = batch.get(du.IMAGE1_NAME), batch.get(du.IMAGE2_NAME)
    kp1, kp2 = endpoint[eu.KP1], endpoint[eu.KP2]
    w_kp1, w_kp2 = endpoint[eu.W_KP1], endpoint[eu.W_KP2]
    w_vis_kp1_mask, w_vis_kp2_mask = endpoint[eu.W_VIS_KP1_MASK], endpoint[eu.W_VIS_KP2_MASK]
    kp1_desc, kp2_desc = endpoint[eu.KP1_DESC], endpoint[eu.KP2_DESC]

    image1_name = image1_name[0]
    image2_name = image2_name[0]

    _, _, _, nn_desc_ids, match_mask = match_score(kp1, kp2, w_kp1, w_kp2, w_vis_kp1_mask, w_vis_kp2_mask,
                                                   kp1_desc, kp2_desc, px_thresh, dd_measure, detailed=True)

    cv_desc_matches = draw_cv_matches(image1, image2, kp1, kp2, nn_desc_ids, match_mask[0])

    writer.add_image(f"{image1_name} and {image2_name} descriptor matches", cv_desc_matches, state_engine.state.epoch,
                     dataformats='HWC')


"""
Terminal and csv logging
"""


def join_logs(data_engine):
    log_names = list(data_engine.state.metrics.keys())
    log = data_engine.state.metrics[log_names[0]]

    on = [du.SCENE_NAME, du.IMAGE1_NAME, du.IMAGE2_NAME, du.ID1, du.ID2]

    for i in range(1, len(log_names)):
        next_log = data_engine.state.metrics[log_names[i]]

        for j in range(len(log)):
            log[j] = log[j].merge(next_log[j], on=on)

    return log


def print_summary(logs, metric_config):
    print("Evaluation summary")
    print("-" * 18)

    px_thresh = metric_config[exp.PX_THRESH]
    r_err_thresh, t_err_thresh = metric_config[exp.R_ERR_THRESH], metric_config[exp.T_ERR_THRESH]

    for thresh, l in zip(px_thresh, logs):
        print("\t" + f"Thresholds: {thresh} px, {r_err_thresh} deg. orientation, {t_err_thresh} deg. translation")
        print("\t" + "-" * 70)

        intent = "\t" * 3

        if meu.R_ERR in l.columns:
            r_correct = l[meu.R_ERR].le(r_err_thresh).sum()
            t_correct = l[meu.T_ERR].le(t_err_thresh).sum()

            r_correct_ratio = float(r_correct) / l.shape[0]
            t_correct_ratio = float(t_correct) / l.shape[0]

            print(intent + f"Correctly estimated orientation in {r_correct}/{l.shape[0]} ({r_correct_ratio:.4})")
            print(intent + f"Correctly estimated translation in {t_correct}/{l.shape[0]} ({t_correct_ratio:.4})")

        if meu.REP in l.columns:
            rep_score = l[meu.REP].mean()

            rep_num_matches = l[meu.REP_NUM_MATCHES].mean()
            rep_num_vis_matches = l[meu.REP_NUM_VIS_GT_MATCHES].mean()

            print(intent + f"Repeatability: {rep_score:.4f} ({rep_num_matches:.2f}/{rep_num_vis_matches:.2f})")

        if meu.MS in l.columns:
            m_score = l[meu.MS].mean()

            ms_num_matches = l[meu.MS_NUM_MATCHES].mean()
            ms_num_vis_matches = l[meu.MS_NUM_VIS_GT_MATCHES].mean()

            print(intent + f"Match score: {m_score:.4f} ({ms_num_matches:.2f}/{ms_num_vis_matches:.2f})")

        if meu.MMA in l.columns:
            mma = l[meu.MMA].mean()

            mma_num_matches = l[meu.MMA_NUM_MATCHES].mean()
            mma_num_vis_matches = l[meu.MMA_NUM_VIS_GT_MATCHES].mean()

            print(intent + f"MMA: {mma:.4f} ({mma_num_matches:.2f}/{mma_num_vis_matches:.2f})")

        if meu.EMS in l.columns:
            ems = l[meu.EMS].mean()

            ems_num_matches = l[meu.EMS_NUM_MATCHES].mean()
            ems_num_vis_gt_matches = l[meu.EMS_NUM_VIS_GT_MATCHES].mean()

            print(intent + f"EMS: {ems:.4f} ({ems_num_matches:.2f}/{ems_num_vis_gt_matches:.2f})")


def test_log_to_csv(log_dir, log, metric_config, models_config, datasets):
    px_thresh = metric_config[exp.PX_THRESH]
    checkpoint_name = models_config[exp.CHECKPOINT_NAME]

    dataset_str = "_".join(datasets)

    for p, l in zip(px_thresh, log):
        l.to_csv(os.path.join(log_dir, f"{checkpoint_name}_{p}_{dataset_str}_px_log.csv"))


def print_metric(metric_name, image1_name, image2_name, metric, num_matches, num_vis_matches, px_thresh):
    for b in range(len(image1_name)):
        pair_name = image1_name[b] + " and " + image2_name[b]
        print(f"Pair: {pair_name}")
        print("-" * 66)

        for i, thresh in enumerate(px_thresh):
            print("\t" + f"Threshold {thresh} px")
            print("\t" + "-" * 18)
            if num_matches is None:
                print("\t" * 2 + f"{metric_name}: {metric[i][b]:.4}")
            else:
                print(
                    "\t" * 2 + f"{metric_name}: {metric[i][b]:.4} ({num_matches[i][b]}/{num_vis_matches[b]})")

            print()

        print()


# Legacy code


# def save_analysis_log(log_dir, log, models_config):
#     checkpoint_name = models_config[exp.CHECKPOINT_NAME]
#     log[0].to_csv(os.path.join(log_dir, f"{checkpoint_name[0]}_analysis_log.csv"))


# def save_aachen_inference(dataset_root, output):
#     batch, endpoint = output
#
#     scene_name, image1_name = batch.get(d.SCENE_NAME)[0], batch.get(d.IMAGE1_NAME)[0]
#     kp1, kp1_desc = revert_data_transform(endpoint[eu.KP1], batch.get(d.SHIFT_SCALE1)), endpoint[eu.KP1_DESC]
#
#     kp1 = kp1.cpu().squeeze().numpy()
#     kp1_desc = kp1_desc.cpu().squeeze().numpy()
#
#     method_name = "NetVGG"
#     file_path = os.path.join(dataset_root, scene_name, f"{image1_name}.{method_name}.npz")
#     r_file_path = os.path.join(dataset_root, scene_name, f"{image1_name}.{method_name}")
#
#     np.savez(file_path, keypoints=kp1, descriptors=kp1_desc)
#     os.rename(file_path, r_file_path)


# """
# Evaluation saving/loading functions
# """

#
# def save(data, log_dir, checkpoint_name):
#     file_path = os.path.join(log_dir, f"{checkpoint_name}.pkl")
#
#     with open(file_path, 'wb') as file:
#         pickle.dump(data, file)
#
#
# def save_eval_results(data_engine, dataset_config, log_dir, checkpoint_name):
#     metrics_dict = {
#         REP: data_engine.state.metrics[REP].cpu().numpy(),
#         MS: data_engine.state.metrics[MS].cpu().numpy(),
#         MMA: data_engine.state.metrics[MMA].cpu().numpy()
#     }
#
#     dataset_name = list(dataset_config.keys())[0]
#     file_path = os.path.join(log_dir, f"{dataset_name}_{EVAL_RESULTS_FILE}_{checkpoint_name}.pkl")
#
#     with open(file_path, 'wb') as file:
#         pickle.dump(metrics_dict, file)


# def save_visualization(engine, dataset_config, log_dir, checkpoint_name, kp_dist_threshold):
#     """"
#     :param engine: Ignite engine to get collected data from
#     :param dataset_config: dict
#     :param log_dir: str
#     :param checkpoint_name: str
#     :param kp_dist_threshold: torch tensor
#     """
#     dataset_name = list(dataset_config.keys())[0]
#     save_dir = os.path.join(log_dir,
#                             f"{dataset_name}_{EVAL_SAMPLES_DIR}_{checkpoint_name}_{kp_dist_threshold[0].item()}px")
#
#     if os.path.exists(save_dir):
#         shutil.rmtree(save_dir)
#
#     os.mkdir(save_dir)
#
#     for (image1_name, image2_name, s_image1, s_image2, kp1, w_kp1, kp2, wv_kp1_mask, wv_kp2_mask, kp1_desc, kp2_desc) in \
#             engine.state.metrics[VISUALIZATION_DATA]:
#         cv_s_image1 = torch2cv(s_image1[0])
#         cv_s_image2 = torch2cv(s_image2[0])
#
#         _, _, _, nn_kp_ids, kp_matches = \
#             repeatability_score(w_kp1, kp2, wv_kp1_mask, wv_kp2_mask, kp_dist_threshold, True)
#
#         _, _, _, nn_desc_ids, desc_matches = \
#             match_score(w_kp1, kp2, wv_kp1_mask, wv_kp2_mask, kp_dist_threshold, kp1_desc, kp2_desc, detailed=True)
#
#         cv_keypoints_matches = cv2.cvtColor(
#             draw_cv_matches(cv_s_image1, cv_s_image2, kp1[0], kp2[0], nn_kp_ids[0], kp_matches[0]), cv2.COLOR_RGB2BGR)
#         cv_desc_matches = cv2.cvtColor(
#             draw_cv_matches(cv_s_image1, cv_s_image2, kp1[0], kp2[0], nn_desc_ids[0], desc_matches[0]),
#             cv2.COLOR_RGB2BGR)
#
#         cv2.imwrite(os.path.join(save_dir, f'{image1_name}_{image2_name}_kp_matches.png'), cv_keypoints_matches)
#         cv2.imwrite(os.path.join(save_dir, f'{image1_name}_{image2_name}_desc_matches.png'), cv_desc_matches)


# def load_eval_results(test_config_path, log_dirs):
#     eval_results = []
#
#     with open(test_config_path, 'r') as stream:
#         test_config = yaml.safe_load(stream)
#
#     for dataset_dict in test_config[exp.DATASET]:
#         for dataset_name in dataset_dict.keys():
#             eval_summary = {exp.PX_THRESH: test_config[exp.METRIC][exp.PX_THRESH]}
#             # KP_DIST_INTERVAL: test_config[METRIC][KP_DIST_INTERVAL]}
#
#             for model_configs, log_dir in zip(test_config[exp.MODEL], log_dirs):
#                 for model_name, model_config in model_configs.items():
#                     path = os.path.join(log_dir, model_config[exp.EXP_NAME],
#                                         f"{dataset_name}_{EVAL_RESULTS_FILE}_{model_config[exp.CHECKPOINT_NAME]}.pkl")
#                     with open(path, 'rb') as stream:
#                         eval_summary[model_name] = pickle.load(stream)
#
#             eval_results.append(tuple([dataset_name, eval_summary]))
#
#     return eval_results


# def plot_nms_scores(writer, state_engine, data_engine):
#     batch, endpoint = data_engine.state.output
#
#     image1_name = batch.get(d.IMAGE1_NAME)
#     # image2_name, batch.get(d.IMAGE2_NAME)
#
#     nms_score1 = endpoint[eu.MODEL_INFO1][mu.NMS_MS_SCORE]
#
#     b, c = nms_score1.shape[:2]
#
#     nms_score1 = nms_score1 / nms_score1.view(b, c, -1).max(dim=-1)[0].view(b, c, 1, 1)
#     # nms_score2# endpoint[eu.MODEL_INFO2][mu.MULTI_SCALE_NMS]
#
#     image1_name = image1_name[0]
#     # image2_name = image2_name[0]
#
#     nms_score1 = nms_score1.permute(1, 0, 2, 3)
#     # nms_score2 = nms_score2.permute(1, 0, 2, 3)
#
#     nms_scores = make_grid(nms_score1, nrow=2)
#
#     writer.add_image(f"{image1_name} nms scoress", nms_scores, state_engine.state.epoch)

# def display_metrics_comparison(eval_results):
#     rows_data = []
#     column_keys = [REP, MS, MMA, NN_MAP]
#
#     dataset_index = []
#     threshold_index = []
#     model_index = []
#
#     column_names = [METRIC_MAPPING[REP],
#                     METRIC_MAPPING[MS],
#                     METRIC_MAPPING[MMA]
#
#     for dataset_name, eval_summary in eval_results:
#         kp_dist_thresh = eval_summary[exp.PX_THRESH]
#         # kp_dist_interval = eval_summary[KP_DIST_INTERVAL]
#
#         for i, thresh in enumerate(kp_dist_thresh):
#
#             for key, value in eval_summary.items():
#                 if key not in [exp.PX_THRESH]:
#                     dataset_index.append(dataset_name)
#                     threshold_index.append(thresh)
#                     model_index.append(key)
#
#                     row_data = []
#
#                     for col in column_keys:
#                         row_data.append(value[col][i])
#                         # if col == MMA:
#
#                         # row_data.append(value[col][int(kp_dist_thresh[i] - kp_dist_interval[0])])
#                         # else:
#                         #     row_data.append(value[col][i])
#
#                     rows_data.append(tuple(row_data))
#
#     multi_index = pd.MultiIndex.from_tuples(list(zip(dataset_index, threshold_index, model_index)),
#                                             names=['Dataset', "Threshold [px]", "Model"])
#
#     table = pd.DataFrame(rows_data, columns=column_names, index=multi_index)
#
#     index_style = dict(selector=f"th:nth-child(-n+{3})", props=[('text-align', 'center'),
#                                                                 ('width', '5em'),
#                                                                 ('border-style', 'solid'),
#                                                                 ('border-width', '2px'),
#                                                                 ('border-color', '#6aab7b')])
#     column_style = dict(selector=f"th:nth-last-child(-n+{5})", props=[('text-align', 'center'),
#                                                                       ('width', '10em'),
#                                                                       ('border-style', 'solid'),
#                                                                       ('border-width', '2px'),
#                                                                       ('border-color', '#6aab7b')])
#
#     def highlight_max(data):
#         styles = [''] * data.size
#
#         num_thresholds = len(data.index.levels[1].values)
#         num_models = len(data.index.levels[2].values)
#
#         for i, dataset_name in enumerate(data.index.levels[0].values):
#             for j, thresh in enumerate(data.index.levels[1].values):
#                 subset = data.loc[(dataset_name, thresh)]
#                 index = i * num_thresholds * num_models + j * num_models + subset.values.argmax()
#                 styles[index] = 'color: #c2617c; font-weight: bold; '
#
#         return styles
#
#     return table.style. \
#         format({METRIC_MAPPING[REP]: '{:.6f}',
#                 METRIC_MAPPING[MS]: '{:.6f}',
#                 METRIC_MAPPING[MMA]: '{:.6f}',
#                 METRIC_MAPPING[NN_MAP]: '{:.6f}'}). \
#         set_properties(subset=column_names, **{'width': '10em',
#                                                'text-align': 'center',
#                                                'border-style': 'solid',
#                                                'border-width': '1px',
#                                                'border-color': 'black'}). \
#         set_table_styles([index_style, column_style]). \
#         apply(highlight_max, axis=0)

# def plot_keypoints(writer, state_engine, data_engine):
#     batch, endpoint = data_engine.state.output
#
#     s_image1, s_image2 = batch.get(d.S_IMAGE1), batch.get(d.S_IMAGE1)
#     image1_name, image2_name = batch.get(d.IMAGE1_NAME), batch.get(d.IMAGE2_NAME)
#     kp1, kp2 = endpoint[eu.KP1], endpoint[eu.KP2]
#
#     cv_s_image1 = torch2cv(s_image1[0])
#     cv_s_image2 = torch2cv(s_image2[0])
#
#     image1_name = image1_name[0]
#     image2_name = image2_name[0]
#
#     s_image1_kp = cv2torch(draw_cv_keypoints(cv_s_image1, kp1[0], (0, 255, 0))).unsqueeze(0)
#     s_image2_kp = cv2torch(draw_cv_keypoints(cv_s_image2, kp2[0], (0, 255, 0))).unsqueeze(0)
#
#     writer.add_image(f"{image1_name} and {image2_name} keypoints",
#                      make_grid(torch.cat((s_image1_kp, s_image2_kp), dim=0)), state_engine.state.epoch)
# """
# Variables for saving the results into files
# """
# EVAL_RESULTS_FILE = 'eval_results'
# EVAL_SAMPLES_DIR = 'eval_samples'