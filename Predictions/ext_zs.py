import argparse
import os
import os.path as osp

import numpy as np
import torch
import pre_process_zs as prep
from torch.utils.data import DataLoader
from data_list import ImageList

def image_classification(loader, model):
    count = 0
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader["test"])
        for i in range(len(loader['test'])):
            if(count%100 == 0):
                print(count)
            count = count + 1
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            labels = labels
            feats, outputs = model(inputs)
            # print(labels)
            _, predict_loc = torch.max(outputs, 1)
            # print(outputs.shape)
            # print(predict_loc)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                all_feats = feats.float().cpu()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                all_feats = torch.cat((all_feats, feats.float().cpu()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy, all_output, all_label.cpu()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    # parser.add_argument('--net', type=str, default='ResNet50', choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "VGG11", "VGG13", "VGG16", "VGG19", "VGG11BN", "VGG13BN", "VGG16BN", "VGG19BN", "AlexNet"])
    parser.add_argument('--dset', type=str, default='office-home', choices=['office', 'visda', 'office-home', 'domain-net'], help="The dataset or source dataset used")
    parser.add_argument('--dset_path', type=str, default='../OfficeHome/Art/label_c.txt', help="The source dataset path list")
    parser.add_argument('--dom_name', type=str, default='Art', help="Domain Name")
    parser.add_argument('--template_id', type=int, default='3', help="The text template")
    parser.add_argument('--interpolate', type=int, default='1', help="Interpolate positional embeddings")
    parser.add_argument('--network', type=str, default='ViT', choices=['ResNet', 'ViT'], help="Backbone network")

    args = parser.parse_args()

    if(args.network == "ResNet"):
        import network_zs_res as network
        print("ResNet")
    else:
        import network_zs as network

    dataset = args.dset
    template_id = args.template_id
    interpolate = args.interpolate

    dom_name = args.dom_name
    if(dataset == 'office-home'):
        d_name = "OH"
    if (dataset == 'office'):
        d_name = "O31"
    if (dataset == 'visda'):
        d_name = "VS"
    if (dataset == 'domain-net'):
        d_name = "DN"

    # train config
    config = {}
    # config["gpu"] = args.gpu_id
    config["output_path"] = d_name + "/" + args.network + "/"
    if not osp.exists(config["output_path"]):
        os.system('mkdir -p '+config["output_path"])
    out_file = open(osp.join(config["output_path"], dom_name + "_" + str(template_id) + "_" + str(interpolate) + "-log.txt"), "w")
    if not osp.exists(config["output_path"]):
        os.mkdir(config["output_path"])

    if dataset == "office":
        class_num = 31
    elif dataset == "visda":
        class_num = 12
    elif dataset == "office-home":
        class_num = 65
    elif dataset == "domain-net":
        class_num = 345
    else:
        raise ValueError('Dataset cannot be recognized. Please define your own dataset here.')
    out_file.write(str(config)+"\n")
    out_file.flush()
    ## set pre-process
    prep_dict = {}

    bs = 1

    if(args.interpolate == 0):
        bs = 64
        prep_dict["test"] = prep.image_test()
    elif(args.network == 'ViT'):
        prep_dict["test"] = prep.image_test_interViT()
    elif(args.network == 'ResNet'):
        prep_dict["test"] = prep.image_test_interRes()
    else:
        print("Not found:" + args.network)
    ## prepare data
    dsets = {}
    dset_loaders = {}
    data_path = args.dset_path


    dsets["test"] = ImageList(open(data_path).readlines(), \
                              transform=prep_dict["test"])
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=bs, \
                                      shuffle=False, num_workers=32)

    base_network = network.ResNetFc(class_num=class_num, template_id=template_id, inter=interpolate, dset=dataset)

    base_network = base_network.cuda()


    base_network.train(False)
    temp_acc, feats_src, lab_src = image_classification(dset_loaders, \
                                         base_network)
    log_str = "Src: precision: {:.5f}".format(temp_acc)
    out_file.write(log_str + "\n")
    out_file.flush()
    print(log_str)
    feats_src = feats_src.numpy()
    saveFile = osp.join(config["output_path"], dom_name + "_" + str(template_id) + "_" + str(interpolate) + ".npy")
    np.save(saveFile, feats_src)
    # np.save("Predictions_ViT_dn/real_ImageNetSm.npy", feats_src)
