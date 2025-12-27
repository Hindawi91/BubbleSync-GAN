from model import Generator
from model import Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
from sklearn.cluster import KMeans
from get_blobs_properties import get_blobs_properties
import torch.nn as nn


class Solver(object):
    """Solver for training and testing"""

    def __init__(self, data_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.data_loader = data_loader

        # Model configurations.
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp
        self.lambda_id = config.lambda_id

        ########################################### Blobs Settings #############################

        self.add_blob_count_loss = config.add_blob_count_loss 
        self.add_blob_mean_area_loss = config.add_blob_mean_area_loss
        self.add_blob_std_area_loss = config.add_blob_std_area_loss

        self.lambda_count = config.lambda_count
        self.lambda_mean = config.lambda_mean
        self.lambda_std= config.lambda_std

        self.source_domain = config.source_domain
        self.target_domain = config.target_domain

        ########################################################################################

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""
        self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
        self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num) 
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
            
        self.G.to(self.device)
        self.D.to(self.device)

    def recreate_image(self, codebook, labels, w, h):
        """Recreate the (compressed) image from the code book & labels"""
        d = codebook.shape[1]
        image = np.zeros((w, h, d))
        label_idx = 0
        for i in range(w):
            for j in range(h):
                image[i][j] = codebook[labels[label_idx]]
                label_idx += 1
        return image

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim=1, dataset='Boiling'):
        """Generate target domain labels for debugging and testing."""

        c_trg_list = []
        for i in range(c_dim):
            c_trg = c_org.clone()
            c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def classification_loss(self, logit, target, dataset='CelebA'):
        """Compute binary or softmax cross entropy loss."""
        return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)


    def blob_count_loss(self, blob_counts_real, blob_counts_fake):
        blob_count_loss = nn.MSELoss()(blob_counts_real, blob_counts_fake)
        return blob_count_loss

    def blob_mean_size_loss(self, blob_mean_areas_real, blob_mean_areas_fake):
        blob_mean_size_loss = nn.MSELoss()(blob_mean_areas_real, blob_mean_areas_fake)
        return blob_mean_size_loss

    def blob_std_size_loss(self, blob_std_areas_real, blob_std_areas_fake):
        blob_std_size_loss = nn.MSELoss()(blob_std_areas_real, blob_std_areas_fake)
        return blob_std_size_loss

    def all_blobs_losses(self,image1,image1_label,image2,image2_label):

        blob_counts_real,blob_mean_areas_real,blob_std_areas_real = get_blobs_properties(images=image1,labels=image1_label, device = self.device,
                                    source_domain = self.source_domain, target_domain= self.target_domain)

        blob_counts_fake,blob_mean_areas_fake,blob_std_areas_fake = get_blobs_properties(images=image2,labels=image2_label, device = self.device,
                                    source_domain = self.source_domain, target_domain= self.target_domain)

        blob_count_loss = self.blob_count_loss(blob_counts_real, blob_counts_fake)
        blob_mean_size_loss = self.blob_mean_size_loss(blob_mean_areas_real, blob_mean_areas_fake)
        blob_std_size_loss = self.blob_std_size_loss(blob_std_areas_real, blob_std_areas_fake)

        return blob_count_loss,blob_mean_size_loss,blob_std_size_loss



    def train(self):
        """Train BubbleSync-GAN within a single dataset."""
        # Set data loader.
        data_loader = self.data_loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        x_fixed, c_org = next(data_iter)
        x_fixed = x_fixed.to(self.device)
        c_fixed_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_real, label_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, label_org = next(data_iter)

            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]


            c_org = label_org.clone()
            c_trg = label_trg.clone()


            x_real = x_real.to(self.device)           # Input images.
            c_org = c_org.to(self.device)             # Original domain labels.
            c_trg = c_trg.to(self.device)             # Target domain labels.
            label_org = label_org.to(self.device)     # Labels for computing classification loss.
            label_trg = label_trg.to(self.device)     # Labels for computing classification loss.

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            out_src, out_cls = self.D(x_real)
            d_loss_real = - torch.mean(out_src)
            d_loss_cls = self.classification_loss(out_cls, label_org, self.dataset)

            # Compute loss with fake images.
            delta = self.G(x_real, c_trg)
            x_fake = torch.tanh(x_real + delta)
            out_src, out_cls = self.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_gp'] = d_loss_gp.item()
            
            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #
            
            if (i+1) % self.n_critic == 0:
                # Original-to-target domain.
                delta = self.G(x_real, c_trg)
                x_fake = torch.tanh(x_real + delta)
                out_src, out_cls = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls = self.classification_loss(out_cls, label_trg, self.dataset)

                ##################################################### BLOB LOSSES #################################################################
                blob_count_loss,blob_mean_size_loss,blob_std_size_loss = self.all_blobs_losses(x_real,label_org,x_fake,label_trg)
                ###################################################################################################################################

                # blob_counts_real,blob_mean_areas_real,blob_std_areas_real = get_blobs_properties(images=x_real,labels=label_org, device = self.device,
                #                     source_domain = self.source_domain, target_domain= self.target_domain)

                # blob_counts_fake,blob_mean_areas_fake,blob_std_areas_fake = get_blobs_properties(images=x_fake,labels=label_trg, device = self.device, 
                #                     source_domain = self.source_domain, target_domain= self.target_domain)


                # blob_count_loss = self.blob_count_loss(blob_counts_real, blob_counts_fake)
                # blob_mean_size_loss = self.blob_mean_size_loss(blob_mean_areas_real, blob_mean_areas_fake)
                # blob_std_size_loss = self.blob_std_size_loss(blob_std_areas_real, blob_std_areas_fake)

                ##################################################### BLOB LOSSES #################################################################


                # Original-to-original domain.
                delta_id = self.G(x_real, c_org)
                x_fake_id = torch.tanh(x_real + delta_id)
                out_src_id, out_cls_id = self.D(x_fake_id)
                g_loss_fake_id = - torch.mean(out_src_id)
                g_loss_cls_id = self.classification_loss(out_cls_id, label_org, self.dataset)
                g_loss_id = torch.mean(torch.abs(x_real - torch.tanh(delta_id + x_real)))

                ##################################################### BLOB LOSSES #################################################################
                blob_count_loss_id,blob_mean_size_loss_id,blob_std_size_loss_id = self.all_blobs_losses(x_real,label_org,x_fake_id,label_org)
                ###################################################################################################################################

                # Target-to-original domain.
                delta_reconst = self.G(x_fake, c_org)
                x_reconst = torch.tanh(x_fake + delta_reconst)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                ##################################################### BLOB LOSSES #################################################################
                blob_count_loss_reconst,blob_mean_size_loss_reconst,blob_std_size_loss_reconst = self.all_blobs_losses(x_fake,label_trg,x_reconst,label_org)
                ###################################################################################################################################

                # Original-to-original domain.
                delta_reconst_id = self.G(x_fake_id, c_org)
                x_reconst_id = torch.tanh(x_fake_id + delta_reconst_id)
                g_loss_rec_id = torch.mean(torch.abs(x_real - x_reconst_id))

                ##################################################### BLOB LOSSES #################################################################
                blob_count_loss_reconst_id,blob_mean_size_loss_reconst_id,blob_std_size_loss_reconst_id = self.all_blobs_losses(x_fake_id,label_org,x_reconst_id,label_org)
                ###################################################################################################################################

                # Backward and optimize.

                  

                blob_count_losses = self.lambda_count * (blob_count_loss + blob_count_loss_id + blob_count_loss_reconst + blob_count_loss_reconst_id)
                blob_mean_area_losses = self.lambda_mean * (blob_mean_size_loss + blob_mean_size_loss_id + blob_mean_size_loss_reconst + blob_mean_size_loss_reconst_id)
                blob_std_area_losses = self.lambda_std * (blob_std_size_loss + blob_std_size_loss_id + blob_std_size_loss_reconst + blob_std_size_loss_reconst_id)

                # ONLY INCLUDE SPECIFIED LOSSES based on flags configurations
                blobs_losses = self.add_blob_count_loss*blob_count_losses + self.add_blob_mean_area_loss*blob_mean_area_losses + self.add_blob_std_area_loss*blob_std_area_losses


                g_loss_same = g_loss_fake_id + self.lambda_rec * g_loss_rec_id + self.lambda_cls * g_loss_cls_id + self.lambda_id * g_loss_id
                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls + g_loss_same + blobs_losses

                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_cls.item()
                loss['G/loss_fake_id'] = g_loss_fake_id.item()
                loss['G/loss_rec_id'] = g_loss_rec_id.item()
                loss['G/loss_cls_id'] = g_loss_cls_id.item()
                loss['G/loss_id'] = g_loss_id.item()

                if self.add_blob_count_loss:
                    loss['G/blob_count_losses'] = blob_count_losses.item()

                    # loss['G/blob_count_loss'] = blob_count_loss.item()
                    # loss['G/blob_count_loss_id'] = blob_count_loss_id.item()
                    # loss['G/blob_count_loss_reconst'] = blob_count_loss_reconst.item()
                    # loss['G/blob_count_loss_reconst_id'] = blob_count_loss_reconst_id.item()

                if self.add_blob_mean_area_loss:

                    loss['G/blob_mean_area_losses'] = blob_mean_area_losses.item()

                    # loss['G/blob_mean_size_loss'] = blob_mean_size_loss.item()
                    # loss['G/blob_mean_size_loss_id'] = blob_mean_size_loss_id.item()
                    # loss['G/blob_mean_size_loss_reconst'] = blob_mean_size_loss_reconst.item()
                    # loss['G/blob_mean_size_loss_reconst_id'] = blob_mean_size_loss_reconst_id.item()

                if self.add_blob_std_area_loss:

                    loss['G/blob_std_area_losses'] = blob_std_area_losses.item()

                    # loss['G/blob_std_size_loss'] = blob_std_size_loss.item()
                    # loss['G/blob_std_size_loss_id'] = blob_std_size_loss_id.item()
                    # loss['G/blob_std_size_loss_reconst'] = blob_std_size_loss_reconst.item()
                    # loss['G/blob_std_size_loss_reconst_id'] = blob_std_size_loss_reconst_id.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    for c_fixed in c_fixed_list:
                        delta = self.G(x_fixed, c_fixed)
                        x_fake_list.append(torch.tanh(delta + x_fixed))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))


    def test(self):
        """Translate images using BubbleSync-GAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        
        # Set data loader.
        if self.dataset in ['Boiling']:
            data_loader = self.data_loader
        
        with torch.no_grad():
            for i, (x_real, c_org, filename) in enumerate(data_loader):
                x_real = x_real.to(self.device)

                c_trg = c_org.clone()
                c_trg[:, 0] = 0 #0 # always to healthy      (FIRAS: CHANGE BACK TO 0 TO CANCEL REVERSE TRANSLATION)       
                c_trg_list = [c_trg.to(self.device)]

                # Translate images.
                #x_fake_list = [x_real]
                x_fake_list = []
                for c_trg in c_trg_list:
                    delta = self.G(x_real, c_trg)
                    delta_org = torch.abs(torch.tanh(delta + x_real) - x_real) - 1.0
                    delta_gray = np.mean(delta_org.data.cpu().numpy(), axis=1)
                    delta_gray_norm = []

                    loc = []
                    cls_mul = []

                    for indx in range(delta_gray.shape[0]):
                        temp = delta_gray[indx, :, :] + 1.0  
                        tempimg_th = np.percentile(temp, 99)
                        tempimg = np.float32(temp >= tempimg_th)
                        temploc = np.reshape(tempimg, (self.image_size*self.image_size, 1))

                        kmeans = KMeans(n_clusters=2, random_state=0).fit(temploc)
                        labels = kmeans.predict(temploc)

                        recreated_loc = self.recreate_image(kmeans.cluster_centers_, labels, self.image_size, self.image_size)
                        recreated_loc = ((recreated_loc - np.min(recreated_loc)) / (np.max(recreated_loc) - np.min(recreated_loc)))

                        loc.append(recreated_loc)
                        delta_gray_norm.append( tempimg )


                    loc = np.array(loc, dtype=np.float32)[:, :, :, 0]
                    delta_gray_norm = np.array(delta_gray_norm)

                    loc = (loc * 2.0) - 1.0
                    delta_gray_norm = (delta_gray_norm * 2.0) - 1.0

                    #x_fake_list.append( torch.from_numpy(np.repeat(delta_gray[:, np.newaxis, :, :], 3, axis=1)).to(self.device) ) # difference map
                    #x_fake_list.append( torch.from_numpy(np.repeat(delta_gray_norm[:, np.newaxis, :, :], 3, axis=1)).to(self.device) ) # localization thershold
                    #x_fake_list.append( torch.from_numpy(np.repeat(loc[:, np.newaxis, :, :], 3, axis=1)).to(self.device) ) # localization kmeans
                    x_fake_list.append( torch.tanh(delta + x_real) ) # generated image
                    
                    # generated_image = torch.tanh(delta + x_real)
                    
                    # for s,image in enumerate(generated_image):
                    #     result_path = os.path.join(self.result_dir, f'seperated_image-{s},{i}-{indx}-imagesNEW.jpg')
                    #     save_image(self.denorm(image.data.cpu()), result_path, nrow=1, padding=0)

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                # result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                result_path = os.path.join(self.result_dir, '{}'.format(filename[0].split('/')[-1]))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))
