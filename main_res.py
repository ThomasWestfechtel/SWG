import argparse
import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network_res as network
import loss
import pre_process as prep
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_twice
import math
import scipy
from scipy.optimize import fsolve

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

class MyDataset(ImageList):
    def __init__(self, cfg, transform):
        self.btw_data = ImageList(cfg, transform=transform)
        self.imgs = self.btw_data.imgs

    def __getitem__(self, index):
        data, target = self.btw_data[index]
        return data, target, index

    def __len__(self):
        return len(self.btw_data)


class data_batch:
    def __init__(self, gt_data, batch_size: int, drop_last: bool, randomize_flag: bool, num_class: int, num_batch: int) -> None:
        # if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
        #         batch_size <= 0:
        #     raise ValueError("batch_size should be a positive integer value, "
        #                      "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))

        gt_data = gt_data.astype(dtype=int)

        self.class_num = num_class
        self.batch_num = num_batch

        self.norm_mode = False
        self.all_data = np.arange(len(gt_data))

        self.data_len = len(gt_data)

        self.norm_mode_len= math.floor(self.data_len/ self.batch_num)


        self.i_range = len(gt_data)
        self.s_list = []
        self.s_len = np.zeros(num_class, dtype=int)
        if randomize_flag == True:
            self.norm_mode = True
            self.set_length(self.norm_mode_len)
            self.i_range = self.norm_mode_len
        else:
            for c_iter in range(self.class_num):
                cur_data = np.where(gt_data == c_iter)[0]
                self.s_list.append(cur_data)
                self.s_len[c_iter] = len(cur_data)
                if(len(cur_data) < 5):
                    print("Low samples found for: ")
                    print(c_iter)
                    print(len(cur_data))

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.idx = 0
        self.c_iter = 0
        self.drop_class = set()
        self.min_rounds = math.floor((self.data_len / num_class) / 4.0)

    def shuffle_list(self):
        for c_iter in range(self.class_num):
            np.random.shuffle(self.s_list[c_iter])

    def set_length(self, length: int):
        self.data_len = length

    def __iter__(self):
        batch = []
        bs = self.batch_num
        if(self.norm_mode):
            while(True):
                np.random.shuffle(self.all_data)
                for idx in range(self.i_range):
                    for b_iter in range(bs):
                        batch.append(self.all_data[idx*bs+b_iter])
                    yield batch
                    batch = []
        else:
            cur_ctr = 0
            pick_item = np.arange(self.class_num)
            rounds = np.min(self.s_len)
            if rounds < self.min_rounds:
                rounds = self.min_rounds
            while(True):
                self.shuffle_list()
                for l_iter in range(rounds):
                    np.random.shuffle(pick_item)
                    for c_iter in range(self.class_num):
                        c_iter_l = pick_item[c_iter]
                        if(self.s_len[c_iter_l] > l_iter):
                            batch.append(self.s_list[c_iter_l][l_iter])
                            cur_ctr += 1
                            if(cur_ctr % bs == 0):
                                yield batch
                                batch = []
                                cur_ctr = 0


    def __len__(self):
        return self.data_len

    def get_range(self):
        return self.i_range



def image_classification(loader, model):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader["test"])
        for i in range(len(loader['test'])):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            labels = labels
            _, outputs = model(inputs)
            outputs = nn.Softmax(dim=1)(outputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy, all_output


def image_classification_test(loader, model):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader["test"])
        for i in range(len(loader['test'])):
        #for i in range(1):
            # data = iter_test.next()
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            labels = labels
            _, outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy


def train(config):
    ## set pre-process
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["source"] = prep.image_train(**config["prep"]['params'])
    prep_dict["source_fm"] = prep.image_fm(**config["prep"]['params'])
    prep_dict["target"] = prep.image_train(**config["prep"]['params'])
    prep_dict["target_fm"] = prep.image_fm(**config["prep"]['params'])
    prep_dict["test"] = prep.image_test(**config["prep"]['params'])

    ## prepare data
    dsets = {}
    dset_loaders = {}
    dset_loaders_bn = {}
    dset_loaders_bn_t = {}
    data_config = config["data"]
    train_bs = data_config["source"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]

    dsets["source"] = ImageList_twice(open(data_config["source"]["list_path"]).readlines(), \
                                transform=[prep_dict["source"], prep_dict["source_fm"]])
    dsets["target"] = ImageList_twice(open(data_config["target"]["list_path"]).readlines(),
                                      transform=[prep_dict["target"], prep_dict["target_fm"]])

    dsets["stest"] = ImageList(open(data_config["source"]["list_path"]).readlines(), \
                            transform=prep_dict["test"])
    dset_loaders_bn["test"] = DataLoader(dsets["stest"], batch_size=32, \
                            shuffle=True, num_workers=32)

    dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                              transform=prep_dict["test"])
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, \
                                      shuffle=False, num_workers=32)

    dset_loaders_bn_t["test"] = DataLoader(dsets["test"], batch_size=32, \
                            shuffle=True, num_workers=32)

    class_num = config["network"]["params"]["class_num"]

    if(args.dset == 'office-home'):
        if(args.pretext == 0):
            preds_source = np.load("Predictions/OH/ResNet/" + args.s + "_3_1.npy")
            preds_target = np.load("Predictions/OH/ResNet/" + args.t + "_3_1.npy")
        elif(args.pretext == 1):
            preds_source = np.load("Predictions/DAPL/OH/ResNet/" + args.s.lower()[0] + "2" + args.t.lower()[0] + "_source_" + str(args.seed) + ".npy")
            preds_target = np.load("Predictions/DAPL/OH/ResNet/" + args.s.lower()[0] + "2" + args.t.lower()[0] + "_target_" + str(args.seed) + ".npy")
    elif(args.dset == 'visda'):
        if(args.pretext == 0):
            preds_source = np.load("Predictions/VS/ResNet/" + args.s + "_3_1.npy")
            preds_target = np.load("Predictions/VS/ResNet/" + args.t + "_3_1.npy")
        elif(args.pretext == 1):
            preds_source = np.load("Predictions/DAPL/VS/ResNet/" + args.s + "_"+str(args.seed)+".npy")
            preds_target = np.load("Predictions/DAPL/VS/ResNet/" + args.t + "_"+str(args.seed)+".npy")
    elif (args.dset == 'domain-net'):
        if(args.pretext == 0):
            preds_source = np.load("Predictions/DN/ResNet/" + args.s + "_3_1.npy")
            preds_target = np.load("Predictions/DN/ResNet/" + args.t + "_3_1.npy")

    func = lambda tau: args.tau_p - 0.5 * (np.mean(np.max(scipy.special.softmax(preds_source*tau, axis=1), axis=1)) + np.mean(np.max(scipy.special.softmax(preds_target*tau, axis=1), axis=1))) + 100 * (1-np.sign(tau))

    #tau_initial_guess = 0
    tau_initial_guess = 16
    tau_solution = fsolve(func, tau_initial_guess)
    if(np.abs(func(tau_solution) > 0.001) or tau_solution < 0):
        print("Solution failed!!!" + str(np.abs(func(tau_solution))) +" "+str(tau_solution))
        config["out_file"].write("Estimation failed!!!" + "\n")
        config["out_file"].flush()
        return
    else:
        print("Found Solution: " + str(tau_solution))
        config["out_file"].write("Found solution: " + str(tau_solution) + "\n")
        config["out_file"].flush()

    tau_solution = float(tau_solution)

    preds_source = scipy.special.softmax(preds_source*tau_solution, axis=1)
    preds_target = scipy.special.softmax(preds_target*tau_solution, axis=1)

    len_train_source = math.floor(preds_source.shape[0] / args.bs)
    len_train_target = math.floor(preds_target.shape[0] / args.bs)

    if (args.perc > 0.005):
        if ( args.run_id > 0 ):
            loadFile = open(config["load_stem"] + str(args.run_id - 1) + ".npy", 'rb')
            tar_pseu_load = np.load(loadFile)
            add_fac = args.kd_gam
            for l_id in range(args.run_id -1):
                add_fac *= args.kd_gam

            if(args.gsde_flac == 1):
                tar_pseu_load += add_fac * preds_target
        else:
            tar_pseu_load = preds_target


        tar_win_row = np.max(tar_pseu_load, axis=1)
        move_iter = int(round(tar_pseu_load.shape[0] * args.perc))
        tts_ind = np.argpartition(tar_win_row, -move_iter)[-move_iter:]
        tts_classes = np.argmax(tar_pseu_load[tts_ind], axis=1)

        t_source = np.array(dsets["source"].imgs)
        t_target = np.array(dsets["target"].imgs)

        tts_items = t_target[tts_ind]
        tts_items[:, 1] = tts_classes

        t_source = np.append(t_source, tts_items, axis=0)
        preds_source = np.append(preds_source, preds_target[tts_ind], axis=0)

        s_gt = t_source[:, 1]
        t_gt = t_target[:, 1]

        t_source = list(map(tuple, t_source))
        t_target = list(map(tuple, t_target))

        dsets["source"].imgs = t_source
        dsets["target"].imgs = t_target
    else:
        if (args.run_id != 0):
            loadFile = open(config["load_stem"] + str(args.run_id - 1) + ".npy", 'rb')
            tar_pseu_load = np.load(loadFile)
        s_gt = np.array(dsets["source"].imgs)[:, 1]
        t_gt = np.array(dsets["target"].imgs)[:, 1]

    data_batch_source = data_batch(s_gt, batch_size=train_bs, drop_last=False, randomize_flag=False, num_class=class_num, num_batch=train_bs)
    data_batch_target = data_batch(t_gt, batch_size=train_bs, drop_last=False, randomize_flag=True, num_class=class_num, num_batch=train_bs)

    dataloader_source = torch.utils.data.DataLoader(
        dataset=dsets["source"],
        batch_sampler=data_batch_source,
        shuffle=False,
        num_workers=32,
        drop_last=False)

    dataloader_target = torch.utils.data.DataLoader(
        dataset=dsets["target"],
        batch_sampler=data_batch_target,
        shuffle=False,
        num_workers=32,
        drop_last=False)

    ## set base network
    net_config = config["network"]
    base_network = network.basenet(net_config["name"],**net_config["params"])
    base_network = base_network.cuda()

    for name, module in base_network.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            name = name.replace(".", " ")
            indeces = [int(s) for s in name.split() if s.isdigit()]
            exchange = nn.Sequential(network.get_stats(), module)
            if(len(indeces) == 1):
                base_network.feature_layers[indeces[0]] = exchange
            if(len(indeces) == 2):
                if(name[-1] == "1"):
                    base_network.feature_layers[indeces[0]][indeces[1]].bn1 = exchange
                if (name[-1] == "2"):
                    base_network.feature_layers[indeces[0]][indeces[1]].bn2 = exchange
                if (name[-1] == "3"):
                    base_network.feature_layers[indeces[0]][indeces[1]].bn3 = exchange
            if(len(indeces) == 3):
                base_network.feature_layers[indeces[0]][indeces[1]].downsample[2] = exchange


    base_network.train(False)
    temp_acc = image_classification_test(dset_loaders_bn, base_network)
    print("Source Acc 0: "+str(temp_acc))

    all_mean = []
    all_var = []

    for name, module in base_network.named_modules():
        if isinstance(module, network.get_stats):
            mean, var = module.stat_out()
            all_mean.append(mean)
            all_var.append(var)
            module.reset()

    temp_acc = image_classification_test(dset_loaders_bn_t, base_network)
    print("Tgt Acc 0: "+str(temp_acc))

    bn_iter = 0

    for name, module in base_network.named_modules():
        if isinstance(module, network.nn.BatchNorm2d):
            name = name.replace(".", " ")
            indeces = [int(s) for s in name.split() if s.isdigit()]
            indeces[-1] -=1
            if(len(indeces) == 2):
                stats = base_network.feature_layers[indeces[0]][indeces[1]]
            if(len(indeces) == 3):
                if(name[-3] == "1"):
                    stats = base_network.feature_layers[indeces[0]][indeces[1]].bn1[indeces[2]]
                if (name[-3] == "2"):
                    stats = base_network.feature_layers[indeces[0]][indeces[1]].bn2[indeces[2]]
                if (name[-3] == "3"):
                    stats = base_network.feature_layers[indeces[0]][indeces[1]].bn3[indeces[2]]
            if(len(indeces) == 4):
                stats = base_network.feature_layers[indeces[0]][indeces[1]].downsample[2][indeces[3]]

            mean, var = stats.stat_out()

            s_bn = nn.BatchNorm2d(module.num_features)
            s_bn.bias.data = (module.bias.data - (module.running_mean - all_mean[bn_iter]) * module.weight.data / torch.sqrt(module.running_var))
            s_bn.weight.data = (module.weight.data * torch.sqrt(all_var[bn_iter]) / torch.sqrt(module.running_var))
            s_bn.running_mean = all_mean[bn_iter]
            s_bn.running_var = all_var[bn_iter]

            t_bn = nn.BatchNorm2d(module.num_features)
            t_bn.bias.data = (module.bias.data - (module.running_mean - mean) * module.weight.data / torch.sqrt(module.running_var))
            t_bn.weight.data = (module.weight.data * torch.sqrt(var) / torch.sqrt(module.running_var))
            t_bn.running_mean = mean
            t_bn.running_var = var

            par_bn = network.par_bn(s_bn, t_bn)

            if (len(indeces) == 2):
                base_network.feature_layers[indeces[0]] = par_bn
            if (len(indeces) == 3):
                if (name[-3] == "1"):
                    base_network.feature_layers[indeces[0]][indeces[1]].bn1 = par_bn
                if (name[-3] == "2"):
                    base_network.feature_layers[indeces[0]][indeces[1]].bn2 = par_bn
                if (name[-3] == "3"):
                    base_network.feature_layers[indeces[0]][indeces[1]].bn3 = par_bn
            if (len(indeces) == 4):
                base_network.feature_layers[indeces[0]][indeces[1]].downsample[2] = par_bn

            bn_iter +=1


    ## add additional network for some methods
    if (args.meth == "BP"):
        ad_net = network.AdversarialNetwork(256, 1024)
    if (args.meth == "CDAN" or args.meth == "CDANE"):
        ad_net = network.AdversarialNetwork(256 * class_num, 1024)

    ad_net = ad_net.cuda()
    parameter_list = base_network.get_parameters() + ad_net.get_parameters()

    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, \
                                         **(optimizer_config["optim_params"]))

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * param_group['lr_mult']

    ## train
    best_acc = 0.0
    print("Start Training!")
    iter_source = iter(dataloader_source)
    iter_target = iter(dataloader_target)

    preds_source = torch.from_numpy(preds_source)
    preds_source = preds_source.cuda()
    preds_source = preds_source.type(torch.float32)

    preds_target = torch.from_numpy(preds_target)
    preds_target = preds_target.cuda()
    preds_target = preds_target.type(torch.float32)

    config["num_iterations"] = int(args.epochs * len_train_target)
    config["test_interval"] = int(math.floor(config["num_iterations"] / 10))

    print("Training for " + str(config["num_iterations"]) + " iterations")

    for i in range(config["num_iterations"]+1):
        i_ov = i
        if i_ov % config["test_interval"] == 0 and i_ov != 0:
            t_cur = int(i_ov / config["test_interval"])
            base_network.train(False)
            base_network.change_par_bn_state(1)
            temp_acc, tar_prec_vec = image_classification(dset_loaders, \
                                                 base_network)
            if temp_acc > best_acc:
                best_acc = temp_acc
            log_str = "Tgt: iter: {:05d}, precision: {:.5f}".format(t_cur, temp_acc)
            config["out_file"].write(log_str + "\n")
            config["out_file"].flush()
            print(log_str)
            saveFile = open(config["save_labels"], 'wb')
            np.save(saveFile, tar_prec_vec)

        loss_params = config["loss"]
        base_network.train(True)
        ad_net.train(True)
        optimizer.zero_grad()
        inputs_source_c, labels_source, idx_source = iter_source.next()
        inputs_target_c, labels_target, idx_target = iter_target.next()
        inputs_target = inputs_target_c[0]
        inputs_target_2 = inputs_target_c[1]
        inputs_source = inputs_source_c[0]
        inputs_source_2 = inputs_source_c[1]
        inputs_source = inputs_source.cuda()
        inputs_target = inputs_target.cuda()
        labels_source = torch.from_numpy(np.array(labels_source).astype(int))
        labels_source = labels_source.cuda()
        inputs_target_2 = inputs_target_2.cuda()
        inputs_source_2 = inputs_source_2.cuda()

        tp_s = preds_source[idx_source]
        tp_t = preds_target[idx_target]

        base_network.zero_grad()

        base_network.change_par_bn_state(0)
        features_source, outputs_source = base_network(inputs_source)
        base_network.change_par_bn_state(1)
        features_target, outputs_target = base_network(inputs_target)
        features = torch.cat((features_source, features_target), dim=0)
        outputs = torch.cat((outputs_source, outputs_target), dim=0)
        softmax_out = nn.Softmax(dim=1)(outputs)
        transfer_loss = 0

        if (args.kd_flac == 1):
            KD_loss_s = nn.KLDivLoss()(nn.functional.log_softmax(outputs_source, dim=1), tp_s)
            KD_loss_t = nn.KLDivLoss()(nn.functional.log_softmax(outputs_target, dim=1), tp_t)
            KD_loss = KD_loss_t + KD_loss_s
            transfer_loss += KD_loss
        if(args.meth == "BP"):
            outputs_target_sm = nn.Softmax(dim=1)(outputs_target)
            c_, pseu_labels_target = torch.max(outputs_target_sm, 1)

            batch_size = outputs_target.shape[0]
            mask_target = torch.ones(batch_size, 256)
            mask_target = mask_target.cuda()
            for b_cur in range(batch_size):
                mask_target[b_cur] = base_network.fc.weight.data[pseu_labels_target[b_cur]]
            mask_target = mask_target.detach()

            batch_size = outputs_source.shape[0]
            mask_source = torch.ones(batch_size, 256)
            mask_source = mask_source.cuda()
            for b_cur in range(batch_size):
                mask_source[b_cur] = base_network.fc.weight.data[labels_source[b_cur]]
            mask_source = mask_source.detach()

            mask = torch.cat((mask_source, mask_target), dim=0)
            mask = torch.mul(features, mask)
            transfer_loss = loss.DANN(mask, ad_net)
        if(args.meth == "CDAN"):
            transfer_loss = loss.CDAN([features, softmax_out], ad_net, None, None, None)
        if(args.meth == "CDANE" and args.cdan_flac == 1):
            entropy = loss.Entropy(softmax_out)
            transfer_loss += loss.CDAN([features, softmax_out], ad_net, entropy, 1.0, None)


        if(args.aug_flac == 1):
            base_network.change_par_bn_state(0)
            features_u_sr, outputs_u_sr = base_network(inputs_source_2)
            base_network.change_par_bn_state(1)
            features_u_s, outputs_u_s = base_network(inputs_target_2)
            if (args.kd_flac == 1):
                KD_loss_ss = nn.KLDivLoss()(nn.functional.log_softmax(outputs_u_sr, dim=1), tp_s)
                KD_loss_ts = nn.KLDivLoss()(nn.functional.log_softmax(outputs_u_s, dim=1), tp_t)
                KD_loss_da = KD_loss_ss + KD_loss_ts
                transfer_loss += KD_loss_da

            features_u = torch.cat((features_u_sr, features_u_s), dim=0)
            outputs_u = torch.cat((outputs_u_sr, outputs_u_s), dim=0)
            softmax_out_u = nn.Softmax(dim=1)(outputs_u)

            if(args.meth == "CDANE" and args.cdan_flac == 1):
                entropy_u = loss.Entropy(softmax_out_u)
                transfer_loss += loss.CDAN([features_u, softmax_out_u], ad_net, entropy_u, 1.0, None)

        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
        if (args.aug_flac == 1):
            classifier_loss += nn.CrossEntropyLoss()(outputs_u_sr, labels_source)
        total_loss = classifier_loss + loss_params["trade_off"] * transfer_loss
        total_loss.backward()
        optimizer.step()

    return best_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
    parser.add_argument('--meth', type=str, default='BP', choices=["CDAN", "BP", "CDANE"])
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--dset', type=str, default='office-home', choices=['visda', 'office-home', 'domain-net'], help="The dataset or source dataset used")
    parser.add_argument('--s_dset_path', type=str, default='../OfficeHome/Art/label_c.txt', help="The source dataset path list")
    parser.add_argument('--t_dset_path', type=str, default='../OfficeHome/Clipart/label_c.txt', help="The target dataset path list")
    parser.add_argument('--output_dir', type=str, default='san', help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--tau_p', type=float, default=0.95, help="For distillation source")
    parser.add_argument('--s', type=str, default="Art", help="For distillation source")
    parser.add_argument('--t', type=str, default="Clipart", help="for Distillation target")
    parser.add_argument('--bs', type=int, default=64, help="Batch size")
    parser.add_argument('--epochs', type=int, default=50, help="Training epochs")
    parser.add_argument('--pretext', type=int, default=0, help="Training epochs")
    parser.add_argument('--run_id', type=int, default=0, help="Iterative training - run ID")
    parser.add_argument('--perc', type=float, default=0.1, help="Target to source percentage")
    parser.add_argument('--kd_gam', type=float, default=0.5, help="Factor for Zero-shot for GSDE")

    parser.add_argument('--aug_flac', type=int, default=1, help="Target to source percentage")
    parser.add_argument('--cdan_flac', type=int, default=1, help="Target to source percentage")
    parser.add_argument('--kd_flac', type=int, default=1, help="Target to source percentage")
    parser.add_argument('--gsde_flac', type=int, default=1, help="Target to source percentage")

    args = parser.parse_args()
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

    # Set random number seed.
    np.random.seed(args.seed + 100 * args.run_id)
    torch.manual_seed(args.seed + 100 * args.run_id)

    # train config
    config = {}
    config["gpu"] = args.gpu_id
    config["output_path"] = "snapshot/" + args.output_dir
    if not osp.exists(config["output_path"]):
        os.system('mkdir -p '+config["output_path"])
    config["out_file"] = open(osp.join(config["output_path"], "log-" + str(args.run_id) + ".txt"), "w")
    config["save_labels"] = osp.join(config["output_path"], "logits-" + str(args.run_id) + ".npy")
    config["load_stem"] = osp.join(config["output_path"], "logits-")
    if not osp.exists(config["output_path"]):
        os.mkdir(config["output_path"])

    config["prep"] = {"test_10crop":False, 'params':{"resize_size":256, "crop_size":224, 'alexnet':False}}
    config["loss"] = {"trade_off":1.0}

    config["optimizer"] = {"type":optim.AdamW, "optim_params":{'lr':args.lr, \
                           "weight_decay":0.05}, "lr_type":"inv", \
                           "lr_param":{"lr":args.lr, "gamma":0.001, "power":0.75 } }

    config["dataset"] = args.dset
    config["data"] = {"source":{"list_path":args.s_dset_path, "batch_size":args.bs}, \
                      "target":{"list_path":args.t_dset_path, "batch_size":args.bs}, \
                      "test":{"list_path":args.t_dset_path, "batch_size":args.bs}}

    config["network"] ={"name":"RN50", "params": {"use_bottleneck": True, "bottleneck_dim": 256, "new_cls": True}}

    if config["dataset"] == "visda":
        config["network"]["params"]["class_num"] = 12
        config["network"]["name"] = "RN101"
    elif config["dataset"] == "office-home":
        config["network"]["params"]["class_num"] = 65
    elif config["dataset"] == "domain-net":
        config["network"]["params"]["class_num"] = 345
        config["network"]["name"] = "RN101"
    else:
        raise ValueError('Dataset cannot be recognized. Please define your own dataset here.')
    config["out_file"].write(str(config)+"\n")
    config["out_file"].flush()
    train(config)
