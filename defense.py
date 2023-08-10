import torch
from utils import *

from geometric_median import geometric_median
import sklearn.metrics.pairwise as smp
import numpy as np


def vectorize_net(net):
    return torch.cat([p.view(-1) for p in net.parameters()])


def load_model_weight(net, weight):
    index_bias = 0
    for p_index, p in enumerate(net.parameters()):
        p.data = weight[index_bias:index_bias + p.numel()].view(p.size())
        index_bias += p.numel()


def load_model_weight_diff(net, weight_diff, global_weight):
    """
    load rule: w_t + clipped(w^{local}_t - w_t)
    """
    listed_global_weight = list(global_weight.parameters())
    index_bias = 0
    for p_index, p in enumerate(net.parameters()):
        p.data = weight_diff[index_bias:index_bias + p.numel()].view(p.size()) + listed_global_weight[p_index]
        index_bias += p.numel()


class Defense:
    def __init__(self, *args, **kwargs):
        self.hyper_params = None

    def exec(self, client_model, *args, **kwargs):
        raise NotImplementedError()


class ClippingDefense(Defense):
    """
    Deprecated, do not use this method
    """

    def __init__(self, norm_bound, *args, **kwargs):
        self.norm_bound = norm_bound

    def exec(self, client_model, *args, **kwargs):
        vectorized_net = vectorize_net(client_model)
        weight_norm = torch.norm(vectorized_net).item()
        clipped_weight = vectorized_net / max(1, weight_norm / self.norm_bound)

        logger.info("Norm Clipped Mode {}".format(
            torch.norm(clipped_weight).item()))
        load_model_weight(client_model, clipped_weight)
        # index_bias = 0
        # for p_index, p in enumerate(client_model.parameters()):
        #     p.data =  clipped_weight[index_bias:index_bias+p.numel()].view(p.size())
        #     index_bias += p.numel()
        ##weight_norm = torch.sqrt(sum([torch.norm(p)**2 for p in client_model.parameters()]))
        # for p_index, p in enumerate(client_model.parameters()):
        #    p.data /= max(1, weight_norm/self.norm_bound)
        return None


class WeightDiffClippingDefense(Defense):
    def __init__(self, norm_bound, *args, **kwargs):
        self.norm_bound = norm_bound

    def exec(self, client_model, global_model, *args, **kwargs):
        """
        global_model: the global model at iteration T, bcast from the PS
        client_model: starting from `global_model`, the model on the clients after local retraining
        """
        vectorized_client_net = vectorize_net(client_model)
        vectorized_global_net = vectorize_net(global_model)
        vectorize_diff = vectorized_client_net - vectorized_global_net

        weight_diff_norm = torch.norm(vectorize_diff).item()
        clipped_weight_diff = vectorize_diff / max(1, weight_diff_norm / self.norm_bound)

        logger.info("Norm Weight Diff: {}, Norm Clipped Weight Diff {}".format(weight_diff_norm,
                                                                               torch.norm(clipped_weight_diff).item()))
        load_model_weight_diff(client_model, clipped_weight_diff, global_model)
        return None


class WeakDPDefense(Defense):
    """
        deprecated: don't use!
        according to literature, DPDefense should be applied
        to the aggregated model, not invidual models
        """

    def __init__(self, norm_bound, *args, **kwargs):
        self.norm_bound = norm_bound

    def exec(self, client_model, device, *args, **kwargs):
        self.device = device
        vectorized_net = vectorize_net(client_model)
        weight_norm = torch.norm(vectorized_net).item()
        clipped_weight = vectorized_net / max(1, weight_norm / self.norm_bound)
        dp_weight = clipped_weight + torch.randn(
            vectorized_net.size(), device=self.device) * self.stddev

        load_model_weight(client_model, clipped_weight)
        return None


class AddNoise(Defense):
    def __init__(self, stddev, *args, **kwargs):
        self.stddev = stddev

    def exec(self, client_model, device, *args, **kwargs):
        self.device = device
        vectorized_net = vectorize_net(client_model)
        gaussian_noise = torch.randn(vectorized_net.size(),
                                     device=self.device) * self.stddev
        dp_weight = vectorized_net + gaussian_noise
        load_model_weight(client_model, dp_weight)
        logger.info("Weak DP Defense: added noise of norm: {}".format(torch.norm(gaussian_noise)))

        return None

class Multi_metrics(Defense):   #Our defense

    def __init__(self, num_workers, num_adv, *args, **kwargs):
        self.num_workers = num_workers
        self.s = num_adv

    def exec(self, client_models, num_dps, g_user_indices, device, attack_round=False, *args, **kwargs):

        vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in client_models]

        cos_dis = [0.0] * len(vectorize_nets)
        length_dis = [0.0] * len(vectorize_nets)
        manhattan_dis = [0.0] * len(vectorize_nets)
        for i, g_i in enumerate(vectorize_nets):
            for j in range(len(vectorize_nets)):
                if i != j:
                    g_j = vectorize_nets[j]

                    cosine_distance = float(
                        (1 - np.dot(g_i, g_j) / (np.linalg.norm(g_i) * np.linalg.norm(g_j))) ** 2)   #Compute the different value of cosine distance
                    manhattan_distance = float(np.linalg.norm(g_i - g_j, ord=1))    #Compute the different value of Manhattan distance
                    length_distance = np.abs(float(np.linalg.norm(g_i) - np.linalg.norm(g_j)))    #Compute the different value of Euclidean distance

                    cos_dis[i] += cosine_distance
                    length_dis[i] += length_distance
                    manhattan_dis[i] += manhattan_distance

        tri_distance = np.vstack([cos_dis, manhattan_dis, length_dis]).T

        cov_matrix = np.cov(tri_distance.T)
        inv_matrix = np.linalg.inv(cov_matrix)

        ma_distances = []
        for i, g_i in enumerate(vectorize_nets):
            t = tri_distance[i]
            ma_dis = np.dot(np.dot(t, inv_matrix), t.T)
            ma_distances.append(ma_dis)

        scores = ma_distances
        print(scores)

        p = 0.3
        p_num = p*len(scores)
        topk_ind = np.argpartition(scores, int(p_num))[:int(p_num)]   #sort

        selected_num_dps = np.array(num_dps)[topk_ind]
        reconstructed_freq = [snd / sum(selected_num_dps) for snd in selected_num_dps]

        logger.info("Num data points: {}".format(num_dps))
        logger.info("Num selected data points: {}".format(selected_num_dps))
        logger.info("The chosen ones are users: {}, which are global users: {}".format(topk_ind,
                                                                                       [g_user_indices[ti] for ti in
                                                                                        topk_ind]))
        aggregated_grad = np.average(np.array(vectorize_nets)[topk_ind, :], weights=reconstructed_freq,
                                     axis=0).astype(np.float32)

        aggregated_model = client_models[0]  # slicing which doesn't really matter
        load_model_weight(aggregated_model, torch.from_numpy(aggregated_grad).to(device))
        neo_net_list = [aggregated_model]
        # logger.info("Norm of Aggregated Model: {}".format(torch.norm(torch.nn.utils.parameters_to_vector(aggregated_model.parameters())).item()))
        neo_net_freq = [1.0]
        return neo_net_list, neo_net_freq

class Krum(Defense):
    """
    we implement the robust aggregator at: https://papers.nips.cc/paper/6617-machine-learning-with-adversaries-byzantine-tolerant-gradient-descent.pdf
    and we integrate both krum and multi-krum in this single class
    """

    def __init__(self, mode, num_workers, num_adv, *args, **kwargs):
        assert (mode in ("krum", "multi-krum"))
        self._mode = mode
        self.num_workers = num_workers
        self.s = num_adv

    def exec(self, client_models, num_dps, g_user_indices, device, attack_round=False, *args, **kwargs):


        vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in client_models]
        neighbor_distances = []
        for i, g_i in enumerate(vectorize_nets):
            distance = []
            for j in range(i + 1, len(vectorize_nets)):
                if i != j:
                    g_j = vectorize_nets[j]
                    distance.append(float(np.linalg.norm(g_i - g_j) ** 2))  # let's change this to pytorch version  Euler
            neighbor_distances.append(distance)

        # compute scores
        nb_in_score = self.num_workers - self.s - 2
        scores = []
        for i, g_i in enumerate(vectorize_nets):
            dists = []
            for j, g_j in enumerate(vectorize_nets):
                if j == i:
                    continue
                if j < i:
                    dists.append(neighbor_distances[j][i - j - 1])
                else:
                    dists.append(neighbor_distances[i][j - i - 1])
            # alternative to topk in pytorch and tensorflow
            topk_ind = np.argpartition(dists, nb_in_score)[:nb_in_score]
            scores.append(sum(np.take(dists, topk_ind)))
        if self._mode == "krum":
            i_star = scores.index(min(scores))
            logger.info("@@@@ The chosen one is user: {}, which is global user: {}".format(scores.index(min(scores)),
                                                                                           g_user_indices[scores.index(
                                                                                               min(scores))]))
            aggregated_model = client_models[0]  # slicing which doesn't really matter
            load_model_weight(aggregated_model, torch.from_numpy(vectorize_nets[i_star]).to(device))
            neo_net_list = [aggregated_model]
            logger.info("Norm of Aggregated Model: {}".format(
                torch.norm(torch.nn.utils.parameters_to_vector(aggregated_model.parameters())).item()))
            neo_net_freq = [1.0]
            return neo_net_list, neo_net_freq, int(scores.index(min(scores))), int(
                g_user_indices[scores.index(min(scores))])
        elif self._mode == "multi-krum":
            nb_in_score = 7
            topk_ind = np.argpartition(scores, nb_in_score + 2)[:nb_in_score + 2]
            selected_num_dps = np.array(num_dps)[topk_ind]
            reconstructed_freq = [snd / sum(selected_num_dps) for snd in selected_num_dps]

            logger.info("Num data points: {}".format(num_dps))
            logger.info("Num selected data points: {}".format(selected_num_dps))
            logger.info("The chosen ones are users: {}, which are global users: {}".format(topk_ind,
                                                                                           [g_user_indices[ti] for ti in
                                                                                            topk_ind]))
            aggregated_grad = np.average(np.array(vectorize_nets)[topk_ind, :], weights=reconstructed_freq,
                                         axis=0).astype(np.float32)

            aggregated_model = client_models[0]  # slicing which doesn't really matter
            load_model_weight(aggregated_model, torch.from_numpy(aggregated_grad).to(device))
            neo_net_list = [aggregated_model]
            # logger.info("Norm of Aggregated Model: {}".format(torch.norm(torch.nn.utils.parameters_to_vector(aggregated_model.parameters())).item()))
            neo_net_freq = [1.0]
            return neo_net_list, neo_net_freq


class RFA(Defense):
    """
    we implement the robust aggregator at: 
    https://arxiv.org/pdf/1912.13445.pdf
    the code is translated from the TensorFlow implementation: 
    https://github.com/krishnap25/RFA/blob/01ec26e65f13f46caf1391082aa76efcdb69a7a8/models/model.py#L264-L298
    """

    def __init__(self, *args, **kwargs):
        pass

    def exec(self, client_models, net_freq,
             maxiter=4, eps=1e-5,
             ftol=1e-6, device=torch.device("cuda"),
             *args, **kwargs):
        """Computes geometric median of atoms with weights alphas using Weiszfeld's Algorithm
        """
        # so alphas will be the same as the net freq in our code
        alphas = np.asarray(net_freq, dtype=np.float32)
        vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in client_models]
        # print("client_grads size",vectorize_nets[0].parameters())
        median = self.weighted_average_oracle(vectorize_nets, alphas)

        num_oracle_calls = 1

        # logging
        obj_val = self.geometric_median_objective(median=median, points=vectorize_nets, alphas=alphas)

        logs = []
        log_entry = [0, obj_val, 0, 0]
        logs.append("Tracking log entry: {}".format(log_entry))
        logger.info('Starting Weiszfeld algorithm')
        logger.info(log_entry)

        # start
        for i in range(maxiter):
            prev_median, prev_obj_val = median, obj_val
            weights = np.asarray([alpha / max(eps, self.l2dist(median, p)) for alpha, p in zip(alphas, vectorize_nets)],
                                 dtype=alphas.dtype)
            weights = weights / weights.sum()
            median = self.weighted_average_oracle(vectorize_nets, weights)
            num_oracle_calls += 1
            obj_val = self.geometric_median_objective(median, vectorize_nets, alphas)
            log_entry = [i + 1, obj_val,
                         (prev_obj_val - obj_val) / obj_val,
                         self.l2dist(median, prev_median)]
            logs.append(log_entry)
            logs.append("Tracking log entry: {}".format(log_entry))
            logger.info("#### Oracle Cals: {}, Objective Val: {}".format(num_oracle_calls, obj_val))
            if abs(prev_obj_val - obj_val) < ftol * obj_val:
                break
        # logger.info("Num Oracale Calls: {}, Logs: {}".format(num_oracle_calls, logs))

        aggregated_model = client_models[0]  # slicing which doesn't really matter
        load_model_weight(aggregated_model, torch.from_numpy(median.astype(np.float32)).to(device))
        neo_net_list = [aggregated_model]
        neo_net_freq = [1.0]
        return neo_net_list, neo_net_freq

    def weighted_average_oracle(self, points, weights):
        """Computes weighted average of atoms with specified weights
        Args:
            points: list, whose weighted average we wish to calculate
                Each element is a list_of_np.ndarray
            weights: list of weights of the same length as atoms
        """
        ### original implementation in TFF
        # tot_weights = np.sum(weights)
        # weighted_updates = [np.zeros_like(v) for v in points[0]]
        # for w, p in zip(weights, points):
        #    for j, weighted_val in enumerate(weighted_updates):
        #        weighted_val += (w / tot_weights) * p[j]
        # return weighted_updates
        ####
        tot_weights = np.sum(weights)
        weighted_updates = np.zeros(points[0].shape)
        for w, p in zip(weights, points):
            weighted_updates += (w * p / tot_weights)
        return weighted_updates

    def l2dist(self, p1, p2):
        """L2 distance between p1, p2, each of which is a list of nd-arrays"""
        # this is a helper function
        return np.linalg.norm(p1 - p2)

    def geometric_median_objective(self, median, points, alphas):
        """Compute geometric median objective."""
        return sum([alpha * self.l2dist(median, p) for alpha, p in zip(alphas, points)])


class GeoMedian(Defense):
    """
    we implement the robust aggregator of Geometric Median (GM)
    """

    def __init__(self, *args, **kwargs):
        pass

    def exec(self, client_models, net_freq,
             maxiter=4, eps=1e-5,
             ftol=1e-6, device=torch.device("cuda"),
             *args, **kwargs):
        """Computes geometric median of atoms with weights alphas using Weiszfeld's Algorithm
        """
        # so alphas will be the same as the net freq in our code
        alphas = np.asarray(net_freq, dtype=np.float32)
        vectorize_nets = np.array([vectorize_net(cm).detach().cpu().numpy() for cm in client_models]).astype(np.float32)
        median = geometric_median(vectorize_nets)

        aggregated_model = client_models[0]  # slicing which doesn't really matter
        load_model_weight(aggregated_model, torch.from_numpy(median.astype(np.float32)).to(device))
        neo_net_list = [aggregated_model]
        neo_net_freq = [1.0]
        return neo_net_list, neo_net_freq


class FoolsGold(Defense):
    def __init__(self, use_memory=False, *args, **kwargs):
        self.memory = None
        self.memory_dict = dict()
        self.wv_history = []
        self.use_memory = use_memory
        self.num_clients = 10

    def exec(self, client_models, names, device=torch.device("cuda"), *args, **kwargs):
        cur_time = time.time()
        # num_clients = len(client_grads)

        # client_grads = [vectorize_net(cm).detach().cpu().numpy() for cm in client_models]
        # client_grads = np.array([vectorize_net(cm).detach().cpu().numpy() for cm in client_models]).astype(np.float32)
        client_grads = [vectorize_net(cm).detach().cpu().numpy() for cm in client_models]
        grad_len = np.array(client_grads[0].shape).prod()
        print("client_grads size", client_models[0].parameters())
        # grad_len = len(client_grads)
        # if self.memory is None:
        #     self.memory = np.zeros((self.num_clients, grad_len))
        if len(names) < len(client_grads):
            names = np.append([-1], names)  # put in adv
        self.memory = np.zeros((self.num_clients, grad_len))
        grads = np.zeros((self.num_clients, grad_len))
        for i in range(len(client_grads)):
            # grads[i] = np.reshape(client_grads[i][-2].cpu().data.numpy(), (grad_len))
            grads[i] = np.reshape(client_grads[i], (grad_len))
            if names[i] in self.memory_dict.keys():
                self.memory_dict[names[i]] += grads[i]
            else:
                self.memory_dict[names[i]] = copy.deepcopy(grads[i])
            self.memory[i] = self.memory_dict[names[i]]
        # self.memory += grads

        if self.use_memory:
            wv, alpha = self.foolsgold(self.memory)  # Use FG
        else:
            wv, alpha = self.foolsgold(grads)  # Use FG
        logger.info(f'[foolsgold agg] wv: {wv}')
        self.wv_history.append(wv)

        agg_grads = []
        # Iterate through each layer
        for i in range(len(client_grads[0])):
            assert len(wv) == len(client_grads), 'len of wv {} is not consistent with len of client_grads {}'.format(
                len(wv), len(client_grads))
            temp = wv[0] * client_grads[0][i]
            # Aggregate gradients for a layer
            for c, client_grad in enumerate(client_grads):
                if c == 0:
                    continue
                temp += wv[c] * client_grad[i]
            temp = temp / len(client_grads)
            agg_grads.append(temp)
        # agg_grads = torch.as_tensor(agg_grads)
        aggregated_grad = np.average(np.array(client_grads), weights=wv, axis=0).astype(np.float32)
        agg_grads = np.array(agg_grads)
        aggregated_model = client_models[0]  # slicing which doesn't really matter
        # median = self.weighted_average_oracle(client_grads, wv)
        load_model_weight(aggregated_model, torch.from_numpy(aggregated_grad).to(device))
        # load_model_weight(aggregated_model, torch.from_numpy(median.astype(np.float32)).to(device))
        net_list2 = [aggregated_model]
        net_freq = [1.0]
        # net_freq = [float(item) for item in net_freq]
        print('model aggregation took {}s'.format(time.time() - cur_time))
        print('aggregated model.parameters', net_list2[0].parameters())
        print('agg_grads', agg_grads[0])
        return net_list2, net_freq

    def foolsgold(self, grads):
        """
        :param grads:
        :return: compute similatiry and return weightings
        """
        n_clients = grads.shape[0]
        cs = smp.cosine_similarity(grads) - np.eye(n_clients)

        maxcs = np.max(cs, axis=1)
        # pardoning
        for i in range(n_clients):
            for j in range(n_clients):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
        wv = 1 - (np.max(cs, axis=1))

        wv[wv > 1] = 1
        wv[wv < 0] = 0

        alpha = np.max(cs, axis=1)

        # Rescale so that max value is wv
        wv = wv / np.max(wv)
        wv[(wv == 1)] = .99

        # Logit function
        wv = (np.log(wv / (1 - wv)) + 0.5)
        wv[(np.isinf(wv) + wv > 1)] = 1
        wv[(wv < 0)] = 0

        # wv is the weight
        return wv, alpha

    def weighted_average_oracle(self, points, weights):

        tot_weights = np.sum(weights)
        weighted_updates = np.zeros(points[0].shape)
        for w, p in zip(weights, points):
            weighted_updates += (w * p / tot_weights)
        return weighted_updates


if __name__ == "__main__":
    # some tests here
    import copy
    import torch.nn as nn
    import torch.nn.functional as F


    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout2d(0.25)
            self.dropout2 = nn.Dropout2d(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)
            return output


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # check 1, this should recover the global model
    sim_global_model = Net().to(device)
    sim_local_model1 = copy.deepcopy(sim_global_model)
    # sim_local_model = Net().to(device)
    defender = WeightDiffClippingDefense(norm_bound=5)
    defender.exec(client_model=sim_local_model1, global_model=sim_global_model)

    vec_global_sim_net = vectorize_net(sim_global_model)
    vec_local_sim_net1 = vectorize_net(sim_local_model1)

    # Norm Weight Diff: 0.0, Norm Clipped Weight Diff 0.0
    # Norm Global model: 8.843663215637207, Norm Clipped local model1: 8.843663215637207    
    print("Norm Global model: {}, Norm Clipped local model1: {}".format(torch.norm(vec_global_sim_net).item(),
                                                                        torch.norm(vec_local_sim_net1).item()))

    # check 2, adding some large perturbation
    sim_local_model2 = copy.deepcopy(sim_global_model)
    scaling_facor = 2
    for p_index, p in enumerate(sim_local_model2.parameters()):
        p.data = p.data + torch.randn(p.size()) * scaling_facor
    defender.exec(client_model=sim_local_model2, global_model=sim_global_model)
    vec_local_sim_net2 = vectorize_net(sim_local_model2)

    # Norm Weight Diff: 2191.04345703125, Norm Clipped Weight Diff 4.999983787536621
    # Norm Global model: 8.843663215637207, Norm Clipped local model1: 10.155366897583008    
    print("Norm Global model: {}, Norm Clipped local model1: {}".format(torch.norm(vec_global_sim_net).item(),
                                                                        torch.norm(vec_local_sim_net2).item()))
