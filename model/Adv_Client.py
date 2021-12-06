import copy

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from .Sketch import Sketch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR

from model.Network import NN, MLP_SketchLinear, CNNCifar, CNNMnist, CNNCifar_Sketch, CNNMnist_Sketch

from conf import Args

args = Args()
class DatasetSplit(Dataset):
    def __init__(self, dataset, idx):
        self.dataset = dataset
        self.idx = idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, item):
        image, label = self.dataset[self.idx[item]]
        return image, label


class AdvClient:
    def __init__(self, data, idx, args):
        self.idx = idx
        self.args = args
        self.loss_func = torch.nn.NLLLoss()

        # the adversory client do not the training dataset
        self.ldr_train = DataLoader(DatasetSplit(data, idx), batch_size=self.args.local_batch_size, shuffle=True)

        # model = NN , MLP_SketchLinear
        if self.args.model_type == 'NN':
            self.model = NN(self.args.dim_in, self.args.dim_out).to(self.args.device)
        elif self.args.model_type == 'MLP_SketchLinear':
            self.model = MLP_SketchLinear(self.args.dim_in, self.args.dim_out, self.args.p).to(self.args.device)

        # model == CNN
        elif self.args.model_type == 'CNN' and self.args.datatype == 'mnist':
            self.model = CNNMnist().to(self.args.device)
        elif self.args.model_type == 'CNN' and self.args.datatype == 'cifar':
            self.model = CNNCifar().to(self.args.device)
        elif self.args.model_type == 'CNN' and self.args.datatype == 'LFW':
            self.model = CNNCifar(num_classes = 2).to(self.args.device)

        # model == CNN_Sketch
        elif self.args.model_type == 'CNN_sketch' and self.args.datatype == 'mnist':
            self.model = CNNMnist_Sketch(self.args.p).to(self.args.device)
            self.dummy_model = CNNMnist().to(self.args.device)
        elif self.args.model_type == 'CNN_sketch' and self.args.datatype == 'cifar':
            self.model = CNNCifar_Sketch(self.args.p).to(self.args.device)
            self.dummy_model = CNNCifar().to(self.args.device)
        elif self.args.model_type == 'CNN_sketch' and self.args.datatype == 'LFW':
            self.model = CNNCifar_Sketch(num_classes = 2).to(self.args.device)
            self.dummy_model = CNNCifar(num_classes = 2).to(self.args.device)

        self.weight_t_1 = copy.deepcopy(self.model.state_dict())
        self.trained_weight_t_1 = copy.deepcopy(self.model.state_dict())
        self.prev_paras = copy.deepcopy(self.model.state_dict())

    # get gradients and sketch matrix S from server
    # for gaussian sketch, one needs to pass in sketch_matrices
    # for count sketch, one only needs to pass in hash indices and rnadom signs of the count sketch matrix
    def get_paras(self, paras, hash_idxs, rand_sgns, sketch_matrices=None):
        if self.args.model_type == 'MLP_SketchLinear' or self.args.model_type == 'CNN_sketch':
            if self.args.sketchtype == "gaussian":
                self.sketch_matrices = sketch_matrices
            else:
                self.hash_idxs = hash_idxs
                self.rand_sgns = rand_sgns
            self.weight_t_1 = self.prev_paras
            self.prev_paras = paras  # W_0

            # print("Parameters of model", self.args.model_type)
            # for k in paras.keys():
            #     print(k, paras[k].shape)

            self.model.load_state_dict(paras)
            # !!!!!!!!! the shape of weights is not the same
            dummy_dict = dict()
            dummy_state_dict = self.dummy_model.state_dict()
            for k in dummy_state_dict:
                dummy_dict[k] = torch.reshape(paras[k], dummy_state_dict[k].shape)
            self.dummy_model.load_state_dict(dummy_dict)
        else:            
            self.weight_t_1 = self.prev_paras
            self.prev_paras = paras
            self.model.load_state_dict(paras)

            print("Parameters of model", self.args.model_type)
            for k in paras.keys():
                print(k, paras[k].shape)
            
    
    # # get average updates from server avg_up = (up1 + up2)/2
    # def get_avg_updates(self, avg_update):
    #     self.broadcasted_update = avg_update  # avg_updates from server

    def adjust_learning_rate(self, optimizer, epoch, step):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr_ad = self.args.learningrate_client * (0.1 ** (epoch // step))
        if lr_ad <= 1e-5:
            lr = 1e-5
        else:
            lr = lr_ad
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # every client sends updates to server after training on local data for several epochs
    # actually updates
    def send_grads(self):
        current_grad = dict()
        current_paras = self.model.state_dict()
        for k in current_paras.keys():
            current_grad[k] = current_paras[k] - self.prev_paras[k]
        return current_grad

    def send_paras(self):
        return copy.deepcopy(self.model.state_dict())

    def size(self):
        return len(self.idx)

    def label_to_onehot(self, target, num_classes):
        target = torch.unsqueeze(target, 1)
        onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
        onehot_target.scatter_(1, target, 1)
        return onehot_target

    def criterion(self, pred, target):
        return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))

    # get the gradient of victim client: lambda_W
    def get_lambda_W(self):
        lambda_W = dict()
        i = 0
        for k_idx, k in enumerate(self.prev_paras.keys()):
            # self.model_type = 'CNN'
            if not "sketch" in self.args.model_type :
                lambda_W[k] = -(2*self.prev_paras[k] - self.trained_weight_t_1[k] - \
                    self.weight_t_1[k]) / self.args.learningrate_client/ self.args.local_epochs
            # sketeched model CNN_sketch

            # !!!!!! what if the len(layer weights shape) is greater than 2, like [32, 3, 5, 5]

            else: 
                w_vic = 2*self.prev_paras[k] - self.trained_weight_t_1[k]
                w_avg = self.weight_t_1[k]
                # print(k, "w_vic.shape", w_vic.shape)
                # print(k, "w_avg.shape", w_avg.shape)
                # the first layer weight shape is conv1.weight w_avg.shape torch.Size([32, 3, 5, 5])
                if not len(w_vic.shape) == 2:
                    lambda_W[k] = -(w_vic - w_avg)/ self.args.learningrate_client/ self.args.local_epochs
                # !!!!!!!!!!!!!! only sketch when the weight shape is [_,_]
                elif len(w_vic.shape) == 2:
                    if self.args.sketchtype == 'count':
                        w_vic_sketch = Sketch.countsketch(w_vic, self.hash_idxs[i], self.rand_sgns[i]) # @ S_new
                        w_vic_sketch = Sketch.transpose_countsketch(w_vic_sketch, self.hash_idxs[i], self.rand_sgns[i]) # @ S_new.T

                        w_avg_sketch = Sketch.countsketch(w_avg, self.hash_idxs_old[i], self.rand_sgns_old[i]) # @ S_old
                        w_avg_sketch = Sketch.transpose_countsketch(w_avg_sketch, self.hash_idxs_old[i], self.rand_sgns_old[i]) # @ S_old.T
                        
                    else:
                        # gaussian sketch
                        w_vic_sketch = Sketch.gaussiansketch(w_vic, self.sketch_matrices[i]) # @ S_new
                        w_vic_sketch = Sketch.transpose_gaussiansketch(w_vic_sketch, self.sketch_matrices[i]) # @ S_new.T

                        w_avg_sketch = Sketch.gaussiansketch(w_avg, self.sketch_matrices_old[i]) # @ S_old
                        w_avg_sketch = Sketch.transpose_gaussiansketch(w_avg_sketch, self.sketch_matrices_old[i]) # @ S_old.T
                    i += 1
                    lambda_W[k] = -(w_vic_sketch - w_avg_sketch)/ self.args.learningrate_client/ self.args.local_epochs
                else:
                    continue
        return lambda_W

    # local training for each client
    # optimizer: Adam
    def train(self, current_round):
        self.model.train()
        # train and update

        epoch_losses = list()
        epoch_acces = list()

        if args.attack == 1:
            optimizer_1 = torch.optim.SGD(self.model.parameters(), momentum=0, lr=self.args.learningrate_client)
        else:
            optimizer_1 = torch.optim.Adam(self.model.parameters(), lr=self.args.learningrate_client)
        scheduler_1 = CosineAnnealingLR(optimizer_1, T_max=10, eta_min=1e-5)
        # scheduler = ExponentialLR(optimizer, 0.9, last_epoch=-1)

        # define label_size for generate randint for dummy label
        label_size_dict = {"mnist": 10, "cifar": 10, "LFW": 2}
        label_size = label_size_dict[self.args.datatype]
        
        print("")
        print("======================")
        for iter in range(self.args.local_epochs):
            print('Adversory client local epoch', iter)
            l_sum = 0.0
            correct = 0.0
            # batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                if args.attack==1 and batch_idx > 0: 
                    break
                
                optimizer_1.zero_grad()
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                
                # TODO dealing with the sketched model
                # Predict
                # ========================================
                # Predict sketch
                if self.args.model_type == 'MLP_SketchLinear' or self.args.model_type == 'CNN_sketch':
                    if self.args.sketchtype == 'gaussian':
                        log_probs = self.model(images, sketchmats=self.sketch_matrices)
                    else:
                        log_probs = self.model(images, self.hash_idxs, self.rand_sgns)
                # Predict non-sketch
                else:
                    log_probs = self.model(images)
                # ========================================
                # attacks start
                if current_round == 1:
                    # generate dummy data and label
                    restore_idx = 0  # only extract one image to recover
                    dummy_image = torch.randn(images[restore_idx].size())
                    
                    gt_label = labels[restore_idx] # labels
                    gt_label = gt_label.view(1, )
                    gt_onehot_label = self.label_to_onehot(gt_label, label_size)  # 10
                    dummy_label = torch.randn(gt_onehot_label.size()) #  torch.Size([1, 10])

                    # make up the batch of dummy images
                    dummy_images = dummy_image.repeat(self.args.local_batch_size, 1, 1, 1).to(self.args.device).requires_grad_(True)
                    dummy_labels = dummy_label.repeat(self.args.local_batch_size, 1).to(self.args.device).requires_grad_(True)


                    lambda_W = self.get_lambda_W()                      

                    history = []
                    loss_list = []
                    step_size = 30
                    total_iter = 300

                    optimizer = torch.optim.LBFGS([dummy_images, dummy_labels])
                    # =============================================
                    # adversory attack start
                    for iters in range(total_iter):
                        def closure():
                            optimizer.zero_grad()

                            # compute gradient from dummy data and label
                            # nonsketch model
                            if not "sketch" in self.args.model_type:
                                dummy_preds = self.model(dummy_images)   # F(x_dum, Wt) torch.Size([50, 10]) float
                            else: # sketeched model
                                # weights reshape, then load to 
                                dummy_preds = self.dummy_model(dummy_images)  # Using CNN model 
                                # Using Sketch model
                                # # !!!!!!!!! 
                                # if self.args.sketchtype == 'gaussian':
                                #     dummy_preds = self.model(dummy_images, sketchmats=self.sketch_matrices)
                                # else:
                                #     dummy_preds = self.model(dummy_images, self.hash_idxs, self.rand_sgns)
                            
                            dummy_loss = self.criterion(dummy_preds, dummy_labels)  # loss
                            if 'sketch' in self.args.model_type:
                                dummy_dy_dx = torch.autograd.grad(dummy_loss, self.dummy_model.parameters(), create_graph=True) # \par Loss / \par W
                            else:
                                dummy_dy_dx = torch.autograd.grad(dummy_loss, self.model.parameters(), create_graph=True) # \par Loss / \par W
                            grad_diff = 0
                            for idx, k in enumerate(lambda_W.keys()):
                                gx = dummy_dy_dx[idx]
                                gy = lambda_W[k]

                                # print("gx.shape, gy.shape", gx.shape, gy.shape)

                                grad_diff += ((torch.flatten(gx) - torch.flatten(gy)) ** 2).sum()
                            grad_diff.backward()
                            return grad_diff
                        
                        optimizer.step(closure)

                        # save results to history per step_size operationn
                        if iters % step_size == 0: 
                            current_loss = closure()
                            print("During batch idx:", batch_idx,", iteration number:", iters, ", current loss: %.4f" % current_loss.item())         # !!!!!!!!!!!!!! loss is too high
                            # Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead
                            Adv_result = dummy_images[0].cpu().permute(1, 2, 0).detach().numpy()
                            history.append(Adv_result) # tt = transforms.ToPILImage()
                            loss_list.append(current_loss.item())
                    # adversory attack end

                    # save attack result
                    plt.figure(figsize=(12, 8))   
                    rows = 2
                    total_slices = total_iter // step_size
                    for i in range(total_slices):
                        plt.subplot(rows, total_slices // rows, i + 1)
                        plt.imshow(history[i])
                        plt.title("batch id = %d, iter=%d" % (batch_idx, i * step_size))
                        plt.xlabel("Current Loss = %f"%( loss_list[i]))
                        # plt.axis('off')    
                    plt.suptitle('Attack status: %d , Model type: %s, data type: %s' % (self.args.attack, self.args.model_type, self.args.datatype))
                    if not "sketch" in self.args.model_type:
                        save_path = './data/adv_attack_res/%s_%s_attacking_%d_batch%d.jpg' \
                            % (self.args.model_type, self.args.datatype, self.args.attack, batch_idx)
                    else:
                        save_path = './data/adv_attack_res/%s_%s_%s_attacking_%d_batch%d.jpg' \
                            % (self.args.model_type, self.args.sketchtype, self.args.datatype, self.args.attack, batch_idx)
                    print("Successfully attacked, saved image in ", save_path)
                    plt.savefig(save_path)
                    # =============================================

                    
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer_1.step()
                scheduler_1.step()
                self.trained_weight_t_1 = copy.deepcopy(self.model.state_dict())  # CNN_Sketch

                # save the history for the next iteration
                if self.args.model_type == 'MLP_SketchLinear' or self.args.model_type == 'CNN_sketch':
                    if self.args.sketchtype == "gaussian":
                        self.sketch_matrices_old = self.sketch_matrices
                    else:
                        self.hash_idxs_old = self.hash_idxs
                        self.rand_sgns_old = self.rand_sgns
                else:
                    continue

                l_sum += loss.item()
                pred = log_probs.data.max(1, keepdim=True)[1]
                correct += pred.eq(labels.view_as(pred)).sum()

            n = float(len(self.idx))
            epoch_acc = 100.0 * float(correct) / n
            epoch_loss = l_sum / (batch_idx + 1)
            epoch_losses.append(epoch_loss)
            epoch_acces.append(epoch_acc)
        return sum(epoch_losses) / len(epoch_losses), sum(epoch_acces) / len(epoch_acces)



