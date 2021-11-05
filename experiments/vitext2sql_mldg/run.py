#!/usr/bin/env python

import os
import _jsonnet
import json
import argparse
import attr
import wandb

from experiments.spider_dg import (
    train,
    meta_train,
)
from tensor2struct.commands import preprocess

@attr.s
class PreprocessConfig:
    config = attr.ib()
    config_args = attr.ib()
    
@attr.s
class TrainConfig:
    config = attr.ib()
    config_args = attr.ib()
    logdir = attr.ib()


@attr.s
class MetaTrainConfig:
    config = attr.ib()
    config_args = attr.ib()
    logdir = attr.ib()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=["preprocess", "train", "meta_train", "eval"], help="train/meta_train/dist_train",
    )
    parser.add_argument("exp_config_file", help="jsonnet file for experiments")
    parser.add_argument("--logdir", help="optional override for logdir")
    args = parser.parse_args()

    exp_config = json.loads(_jsonnet.evaluate_file(args.exp_config_file))
    model_config_file = exp_config["model_config"]
    if "model_config_args" in exp_config:
        model_config_args = exp_config["model_config_args"]
        if args.model_config_args is not None:
            model_config_args_json = _jsonnet.evaluate_snippet("", args.model_config_args)
            model_config_args.update(json.loads(model_config_args_json))
        model_config_args = json.dumps(model_config_args)
    elif args.model_config_args is not None:
        model_config_args = _jsonnet.evaluate_snippet("", args.model_config_args)
    else:
        model_config_args = None

    logdir = args.logdir or exp_config["logdir"]

    # # wandb init
    # expname = exp_config["logdir"].split("/")[-1]
    # project = exp_config["project"]

    # # dist train need to start a wandb session in each process, not a global one
    # if args.mode in ["train", "meta_train"]:
    #     wandb.init(project=project, group=expname, job_type=args.mode)

    if args.mode == "preprocess":
        preprocess_config = PreprocessConfig(model_config_file, model_config_args)
        preprocess.main(preprocess_config)

    elif args.mode == "train":
        train_config = TrainConfig(model_config_file, model_config_args, logdir)
        train.main(train_config)
    elif args.mode == "meta_train":
        train_config = MetaTrainConfig(model_config_file, model_config_args, logdir)
        meta_train.main(train_config)


if __name__ == "__main__":
    main()