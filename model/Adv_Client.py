import copy

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

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
        elif self.args.model_type == 'CNN_sketch' and self.args.datatype == 'cifar':
            self.model = CNNCifar_Sketch(self.args.p).to(self.args.device)
        
        elif self.args.model_type == 'CNN_sketch' and self.args.datatype == 'LFW':
            self.model = CNNCifar_Sketch(num_classes = 2).to(self.args.device)

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
            self.model.load_state_dict(paras)
        else:            
            self.weight_t_1 = self.prev_paras
            self.prev_paras = paras
            self.model.load_state_dict(paras)
    
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
                if args.attack==1 and batch_idx >0: 
                    break
                
                optimizer_1.zero_grad()
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                
                # TODO dealing with the sketched model
                # Predict
                # ========================================
                if self.args.model_type == 'MLP_SketchLinear' or self.args.model_type == 'CNN_sketch':
                    if self.args.sketchtype == 'gaussian':
                        log_probs = self.model(images, sketchmats=self.sketch_matrices)
                    else:
                        log_probs = self.model(images, self.hash_idxs, self.rand_sgns)

                # FIRST DEAL WITH NON SKETCHED
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

                    lambda_W = dict()
                    for k in self.prev_paras.keys():
                        lambda_W[k] = -(2*self.prev_paras[k] - self.trained_weight_t_1[k] - self.weight_t_1[k]) / self.args.learningrate_client/ self.args.local_epochs

                    history = []
                    step_size = 30
                    total_iter = 300

                    optimizer = torch.optim.LBFGS([dummy_images, dummy_labels])
                    # =============================================
                    # adversory attack start
                    for iters in range(total_iter):
                        def closure():
                            optimizer.zero_grad()

                            # compute gradient from dummy data and label
                            # model = CNN_sketch or CNN
                            dummy_preds = self.model(dummy_images)   # F(x_dum, Wt) torch.Size([50, 10]) float
                            dummy_loss = self.criterion(dummy_preds, dummy_labels)  # loss
                            dummy_dy_dx = torch.autograd.grad(dummy_loss, self.model.parameters(), create_graph=True) # \par Loss / \par W
                            grad_diff = 0
                            for idx, k in enumerate(lambda_W.keys()):
                                gx = dummy_dy_dx[idx]
                                gy = lambda_W[k]
                                grad_diff += ((gx - gy) ** 2).sum()
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
                    # adversory attack end

                    # save attack result
                    plt.figure(figsize=(12, 8))   
                    rows = 2
                    total_slices = total_iter // step_size
                    for i in range(total_slices):
                        plt.subplot(rows, total_slices // rows, i + 1)
                        plt.imshow(history[i])
                        plt.title("batch id = %d, iter=%d" % (batch_idx, i * step_size))
                        plt.axis('off')    
                    plt.suptitle('Attack status' + self.args.attack + 'Model type' + self.args.model_type)
                    save_path = './data/adv_attack_res/attacking_%d.jpg' % (batch_idx)
                    print("Successfully attacked, saved image in ", save_path)
                    plt.savefig(save_path)
                    # =============================================

                    
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer_1.step()
                scheduler_1.step()
                self.trained_weight_t_1 = copy.deepcopy(self.model.state_dict())
                l_sum += loss.item()
                pred = log_probs.data.max(1, keepdim=True)[1]
                correct += pred.eq(labels.view_as(pred)).sum()

            n = float(len(self.idx))
            epoch_acc = 100.0 * float(correct) / n
            epoch_loss = l_sum / (batch_idx + 1)
            epoch_losses.append(epoch_loss)
            epoch_acces.append(epoch_acc)
        return sum(epoch_losses) / len(epoch_losses), sum(epoch_acces) / len(epoch_acces)



