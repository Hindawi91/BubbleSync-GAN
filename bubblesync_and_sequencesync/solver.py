"""
SequenceSync-GAN solver, merging:
  - SequenceSync-GAN's novel contribution: Temporal_Discriminator, triplet
    (3-frame) data handling, TD (temporal-order) loss (lambda_TD).
  - BubbleSync-GAN's fixes, carried over:
    * Differentiable blob losses (get_blobs_properties_differentiable.py),
      fixing the original zero-gradient bug where blob stats were computed
      via non-differentiable skimage/numpy round-trips.
    * Per-iteration JSONL loss logging (iteration_losses.jsonl).
    * TensorBoard made optional (use_tensorboard=False avoids needing
      TensorFlow at all).

Blob losses now operate on 3-channel triplets instead of single images, so
they're computed PER-FRAME (split each triplet into 3 separate 1-channel
frames, run blob-loss on each, average across the 3) -- see
compute_blob_losses_per_frame(). Both images passed to it use the
differentiable path uniformly, since either can be generator output
depending on the call site (same reasoning as BubbleSync-GAN).

Uses all 4 real/fake comparisons for blob losses, matching BubbleSync-GAN's
original design exactly: (real, fake), (real, fake_id), (fake, reconst),
(fake_id, reconst_id). The 4th comparison requires x_reconst_id, which the
SequenceSync-GAN solver.py I was given had commented out (along with
lambda_rec*g_loss_rec_id in g_loss_same) -- both restored here.

SequenceSync-GAN's test() already correctly selects a single domain loader
based on --direction (no "both domains mixed" bug like BubbleSync-GAN
originally had) -- preserved as-is, not "fixed".
"""
from model import Generator
from model import Temporal_Discriminator
from torchvision.utils import save_image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
import json
from get_blobs_properties_differentiable import get_blobs_properties_differentiable


class Solver(object):
    """Solver for training and testing SequenceSync-GAN."""

    def __init__(self, data_loader_A, data_loader_B, config):
        """Initialize configurations."""

        self.data_loader_A = data_loader_A
        self.data_loader_B = data_loader_B
        self.direction = config.direction

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
        self.lambda_TD = config.lambda_TD

        # Blob loss configuration (carried over from BubbleSync-GAN).
        self.add_blob_count_loss = config.add_blob_count_loss
        self.add_blob_mean_area_loss = config.add_blob_mean_area_loss
        self.add_blob_std_area_loss = config.add_blob_std_area_loss
        self.lambda_count = config.lambda_count
        self.lambda_mean = config.lambda_mean
        self.lambda_std = config.lambda_std
        self.source_domain = config.source_domain
        self.target_domain = config.target_domain

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.td_lr = config.td_lr
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
        self.D = Temporal_Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])

        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')

        self.G.to(self.device)
        self.D.to(self.device)

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
        out = torch.zeros(batch_size, dim, device=labels.device)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org):
        """Generate target domain labels for debugging and testing."""
        c_trg_list = []
        c_trg = c_org.clone()
        c_trg[:, 0] = (c_trg[:, 0] == 0)  # Reverse attribute value.
        c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def classification_loss(self, logit, target):
        """Compute binary or softmax cross entropy loss."""
        return F.binary_cross_entropy_with_logits(logit, target, reduction='sum') / logit.size(0)

    def compute_blob_losses_per_frame(self, image1_triplet, image1_label, image2_triplet, image2_label):
        """Split each 3-channel triplet into 3 separate 1-channel frames,
        compute blob-property losses per-frame via the differentiable path
        (both images, since either can be generator output depending on the
        call site), average across the 3 frames.

        image1_triplet/image2_triplet: (B, 3, H, W) tensors.
        image1_label/image2_label: (B, 1) tensors -- same label for all 3
        frames within a triplet, since all 3 frames come from the same
        domain (see data_loader.py's __getitem__ comment).
        """
        count_loss_total = 0.0
        mean_loss_total = 0.0
        std_loss_total = 0.0

        for frame_idx in range(3):
            frame1 = image1_triplet[:, frame_idx:frame_idx+1, :, :]
            frame2 = image2_triplet[:, frame_idx:frame_idx+1, :, :]

            blob_counts_1, blob_mean_1, blob_std_1 = get_blobs_properties_differentiable(
                images=frame1, labels=image1_label, device=self.device,
                source_domain=self.source_domain, target_domain=self.target_domain)
            blob_counts_2, blob_mean_2, blob_std_2 = get_blobs_properties_differentiable(
                images=frame2, labels=image2_label, device=self.device,
                source_domain=self.source_domain, target_domain=self.target_domain)

            count_loss_total = count_loss_total + nn.MSELoss()(blob_counts_1, blob_counts_2)
            mean_loss_total = mean_loss_total + nn.MSELoss()(blob_mean_1, blob_mean_2)
            std_loss_total = std_loss_total + nn.MSELoss()(blob_std_1, blob_std_2)

        return count_loss_total / 3, mean_loss_total / 3, std_loss_total / 3

    def train(self):
        """Train SequenceSync-GAN."""

        data_loader_A = self.data_loader_A
        data_loader_B = self.data_loader_B

        data_iter_A = iter(data_loader_A)
        data_iter_B = iter(data_loader_B)

        x_fixed_A, c_org_A, seq_label_A, file_names_A = next(data_iter_A)
        x_fixed_B, c_org_B, seq_label_B, file_names_B = next(data_iter_B)
        x_fixed = torch.cat((x_fixed_A, x_fixed_B), dim=0)
        x_fixed = x_fixed.to(self.device)
        c_org = torch.cat((c_org_A, c_org_B), dim=0)
        c_fixed_list = self.create_labels(c_org)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Per-iteration loss log (JSON Lines: one object per line, appended
        # not rewritten -- avoids the O(n^2) I/O cost of rewriting a growing
        # JSON array every iteration, and is crash-safe).
        self.iteration_losses_path = os.path.join(self.log_dir, 'iteration_losses.jsonl')
        self._iteration_loss_file = open(self.iteration_losses_path, 'a' if self.resume_iters else 'w')

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

            try:
                x_real_A, label_org_A, seq_label_A, file_names_A = next(data_iter_A)
            except StopIteration:
                data_iter_A = iter(data_loader_A)
                x_real_A, label_org_A, seq_label_A, file_names_A = next(data_iter_A)

            try:
                x_real_B, label_org_B, seq_label_B, file_names_B = next(data_iter_B)
            except StopIteration:
                data_iter_B = iter(data_loader_B)
                x_real_B, label_org_B, seq_label_B, file_names_B = next(data_iter_B)

            x_real = torch.cat((x_real_A, x_real_B), dim=0)
            label_org = torch.cat((label_org_A, label_org_B), dim=0)
            seq_label = torch.cat((seq_label_A, seq_label_B), dim=0)

            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]

            c_org = label_org.clone()
            c_trg = label_trg.clone()

            x_real = x_real.to(self.device)
            c_org = c_org.to(self.device)
            c_trg = c_trg.to(self.device)
            label_org = label_org.to(self.device)
            label_trg = label_trg.to(self.device)

            seq_label = self.label2onehot(seq_label, dim=2)
            seq_label = seq_label.to(self.device)

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            out_src, out_cls, out_cls_TD = self.D(x_real)
            d_loss_real = - torch.mean(out_src)
            d_loss_cls = self.classification_loss(out_cls, label_org)
            d_td_loss = self.classification_loss(out_cls_TD, seq_label)

            delta = self.G(x_real, c_trg)
            x_fake = torch.tanh(x_real + delta)
            out_src, out_cls, _ = self.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src)

            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            d_loss = d_td_loss + d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_gp'] = d_loss_gp.item()
            loss['D/td_loss'] = d_td_loss.item()

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            if (i+1) % self.n_critic == 0:
                # Original-to-target domain.
                delta = self.G(x_real, c_trg)
                x_fake = torch.tanh(x_real + delta)
                out_src, out_cls, out_cls_TD = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls = self.classification_loss(out_cls, label_trg)
                g_loss_td_cls = self.classification_loss(out_cls_TD, seq_label)

                # Original-to-original domain.
                delta_id = self.G(x_real, c_org)
                x_fake_id = torch.tanh(x_real + delta_id)
                out_src_id, out_cls_id, _ = self.D(x_fake_id)
                g_loss_fake_id = - torch.mean(out_src_id)
                g_loss_cls_id = self.classification_loss(out_cls_id, label_org)
                g_loss_id = torch.mean(torch.abs(x_real - torch.tanh(delta_id + x_real)))

                # Target-to-original domain.
                delta_reconst = self.G(x_fake, c_org)
                x_reconst = torch.tanh(x_fake + delta_reconst)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                # Original-to-original domain (reconstruction of the identity-mapped triplet).
                # Restored: this was commented out in the SequenceSync-GAN version I was given,
                # which meant g_loss_rec_id was silently missing from g_loss_same, and the blob
                # losses below were only using 3 of the original 4 real/fake comparisons.
                delta_reconst_id = self.G(x_fake_id, c_org)
                x_reconst_id = torch.tanh(x_fake_id + delta_reconst_id)
                g_loss_rec_id = torch.mean(torch.abs(x_real - x_reconst_id))

                g_loss_same = g_loss_fake_id + self.lambda_rec * g_loss_rec_id + self.lambda_cls * g_loss_cls_id + self.lambda_id * g_loss_id
                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls + g_loss_same + self.lambda_TD * g_loss_td_cls

                # --- Blob losses (per-frame, differentiable) ------------------------------------
                blobs_losses = 0.0
                if self.add_blob_count_loss or self.add_blob_mean_area_loss or self.add_blob_std_area_loss:
                    # (real, fake): does the translated triplet preserve blob stats vs input?
                    bc1, bm1, bs1 = self.compute_blob_losses_per_frame(x_real, label_org, x_fake, label_trg)
                    # (real, fake_id): identity-mapped triplet should also preserve blob stats.
                    bc2, bm2, bs2 = self.compute_blob_losses_per_frame(x_real, label_org, x_fake_id, label_org)
                    # (fake, reconst): reconstruction should preserve blob stats vs the fake it came from.
                    bc3, bm3, bs3 = self.compute_blob_losses_per_frame(x_fake, label_trg, x_reconst, label_org)
                    # (fake_id, reconst_id): identity reconstruction should also preserve blob stats.
                    bc4, bm4, bs4 = self.compute_blob_losses_per_frame(x_fake_id, label_org, x_reconst_id, label_org)

                    blob_count_losses = self.lambda_count * (bc1 + bc2 + bc3 + bc4)
                    blob_mean_area_losses = self.lambda_mean * (bm1 + bm2 + bm3 + bm4)
                    blob_std_area_losses = self.lambda_std * (bs1 + bs2 + bs3 + bs4)

                    if self.add_blob_count_loss:
                        blobs_losses = blobs_losses + blob_count_losses
                        loss['G/blob_count_losses'] = blob_count_losses.item()
                        loss['G/blob_count_loss_raw'] = (bc1 + bc2 + bc3 + bc4).item()
                    if self.add_blob_mean_area_loss:
                        blobs_losses = blobs_losses + blob_mean_area_losses
                        loss['G/blob_mean_area_losses'] = blob_mean_area_losses.item()
                        loss['G/blob_mean_area_loss_raw'] = (bm1 + bm2 + bm3 + bm4).item()
                    if self.add_blob_std_area_loss:
                        blobs_losses = blobs_losses + blob_std_area_losses
                        loss['G/blob_std_area_losses'] = blob_std_area_losses.item()
                        loss['G/blob_std_area_loss_raw'] = (bs1 + bs2 + bs3 + bs4).item()

                    g_loss = g_loss + blobs_losses
                # ----------------------------------------------------------------------------------

                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_cls.item()
                loss['G/loss_fake_id'] = g_loss_fake_id.item()
                loss['G/loss_rec_id'] = g_loss_rec_id.item()
                loss['G/loss_cls_id'] = g_loss_cls_id.item()
                loss['G/loss_id'] = g_loss_id.item()
                loss['G/loss_td_cls'] = g_loss_td_cls.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            iter_record = {"iteration": i + 1, **loss}
            self._iteration_loss_file.write(json.dumps(iter_record) + "\n")
            self._iteration_loss_file.flush()

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
                print('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

        self._iteration_loss_file.close()

    def test(self):
        """Translate images using SequenceSync-GAN. Already correctly
        selects a SINGLE domain loader based on --direction (no "both
        domains mixed" issue) -- preserved from the original design."""
        self.restore_model(self.test_iters)
        data_loader = self.data_loader_B if self.direction == "B2A" else self.data_loader_A

        with torch.no_grad():
            for i, (x_real, label_org, seq_label, file_names) in enumerate(data_loader):

                x_real = x_real.to(self.device)
                c_trg = label_org.clone()
                c_trg[:, 0] = 0 if self.direction == "B2A" else 1
                c_trg_list = [c_trg.to(self.device)]

                x_fake_list = []
                for c_trg in c_trg_list:
                    delta = self.G(x_real, c_trg)
                    x_fake_list.append(torch.tanh(delta + x_real))

                x_concat = torch.cat(x_fake_list, dim=3)

                for i in range(x_concat.size(0)):
                    seq_image = x_concat[i]
                    for j in range(seq_image.size(0)):
                        image = seq_image[j]
                        image_name = file_names[j][i]
                        result_path = os.path.join(self.result_dir, '{}'.format(image_name.split('/')[-1]))
                        save_image(self.denorm(image.data.cpu()), result_path, nrow=1, padding=0)
                        print('Saved real and fake images into {}...'.format(result_path))
