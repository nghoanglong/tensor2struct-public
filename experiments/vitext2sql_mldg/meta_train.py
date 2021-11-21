import argparse
import collections
import datetime
import json
import os
import logging
import itertools

import _jsonnet
import attr
import numpy as np
import torch
import wandb

from tensor2struct.utils import registry, random_state, vocab
from tensor2struct.utils import saver as saver_mod
from tensor2struct.commands import train, meta_train
from tensor2struct.training import maml


@attr.s
class MetaTrainConfig(meta_train.MetaTrainConfig):
    use_bert_training = attr.ib(kw_only=True)


class MetaTrainer(meta_train.MetaTrainer):
    def load_train_config(self):
        self.train_config = registry.instantiate(
            MetaTrainConfig, self.config["meta_train"]
        )

        if self.train_config.num_batch_accumulated > 1:
            self.logger.warn("Batch accumulation is used only at MAML-step level")

        if self.train_config.use_bert_training:
            if self.train_config.clip_grad is None:
                self.logger.info("Gradient clipping is recommended for BERT training")

    def load_optimizer(self, config):
        with self.init_random:
            # 0. create inner_optimizer
            # inner_parameters = list(self.model.get_trainable_parameters())
            inner_parameters = list(self.model.get_non_bert_parameters())
            inner_optimizer = registry.construct(
                "optimizer", self.train_config.inner_opt, params=inner_parameters
            )
            self.logger.info(f"{len(inner_parameters)} parameters for inner update")

            # 1. MAML trainer, might add new parameters to the optimizer, e.g., step size
            maml_trainer = maml.MAML(
                model=self.model, inner_opt=inner_optimizer, device=self.device,
            )
            maml_trainer.to(self.device)

            opt_params = maml_trainer.get_inner_opt_params()
            self.logger.info(f"{len(opt_params)} opt meta parameters")

            # 2. Outer optimizer
            # if config["optimizer"].get("name", None) in ["bertAdamw", "torchAdamw"]:
            if self.train_config.use_bert_training:
                phobert_params = list(self.model.encoder.phobert_model.parameters())
                assert len(phobert_params) > 0
                non_phobert_params = []
                for name, _param in self.model.named_parameters():
                    if "phobert" not in name:
                        non_phobert_params.append(_param)
                assert len(non_phobert_params) + len(phobert_params) == len(list(self.model.parameters()))
                self.logger.info(
                    f"{len(phobert_params)} PhoBERT parameters and {len(non_phobert_params)} non-PhoBERT parameters"
                )

                optimizer = registry.construct(
                    "optimizer",
                    config["optimizer"],
                    non_phobert_params=non_phobert_params,
                    phobert_params=phobert_params,
                )
                lr_scheduler = registry.construct(
                    "lr_scheduler",
                    config.get("lr_scheduler", {"name": "noop"}),
                    param_groups=[
                        optimizer.non_phobert_param_group,
                        optimizer.phobert_param_group,
                    ],
                )
            else:
                optimizer = registry.construct(
                    "optimizer",
                    config["optimizer"],
                    params=self.model.get_trainable_parameters(),
                )
                lr_scheduler = registry.construct(
                    "lr_scheduler",
                    config.get("lr_scheduler", {"name": "noop"}),
                    param_groups=optimizer.param_groups,
                )

            lr_scheduler = registry.construct(
                "lr_scheduler",
                config.get("lr_scheduler", {"name": "noop"}),
                param_groups=optimizer.param_groups,
            )
            return inner_optimizer, maml_trainer, optimizer, lr_scheduler

    def step(
        self,
        config,
        train_data_scheduler,
        maml_trainer,
        optimizer,
        lr_scheduler,
        last_step,
    ):
        with self.model_random:
            for _i in range(self.train_config.num_batch_accumulated):
                task = train_data_scheduler.get_batch(last_step)
                inner_batch, outer_batches = task
                ret_dic = maml_trainer.meta_train(
                    self.model, inner_batch, outer_batches
                )
                loss = ret_dic["loss"]

            # clip bert grad
            if self.train_config.clip_grad and self.train_config.use_bert_training:
                for param_group in optimizer.param_groups:
                    torch.nn.utils.clip_grad_norm_(
                        param_group["params"], self.train_config.clip_grad,
                    )


            optimizer.step()
            optimizer.zero_grad()

            # log lr for each step
            outer_lr = lr_scheduler.update_lr(last_step)
            if outer_lr is None:
                outer_lr = [param["lr"] for param in optimizer.param_groups]
            inner_lr = [param["lr"] for param in maml_trainer.inner_opt.param_groups]

        # Report metrics and lr
        if last_step % self.train_config.report_every_n == 0:
            self.logger.info("Step {}: loss={:.4f}".format(last_step, loss))
            self.logger.info(f"Step {last_step}, lr={inner_lr, outer_lr}")
            wandb.log({"train_loss": loss}, step=last_step)
            for idx, lr in enumerate(inner_lr):
                wandb.log({f"inner_lr_{idx}": lr}, step=last_step)
            for idx, lr in enumerate(outer_lr):
                wandb.log({f"outer_lr_{idx}": lr}, step=last_step)


def main(args):
    # setup logger etc
    config, logger = train.setup(args)

    # Construct trainer and do training
    trainer = MetaTrainer(logger, config)
    trainer.train(config, modeldir=args.logdir)


if __name__ == "__main__":
    args = train.add_parser()
    main(args)
