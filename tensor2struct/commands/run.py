#!/usr/bin/env python
import os
import _jsonnet
import json
import argparse
import attr
import collections
import torch

import tensor2struct
import experiments

from tensor2struct.commands import preprocess, train, infer, batched_infer, eval, meta_train
from tensor2struct.utils import registry


@attr.s
class SpiderItem:
    text = attr.ib()
    code = attr.ib()
    schema = attr.ib()
    orig = attr.ib()
    orig_schema = attr.ib()

@attr.s
class PreprocessConfig:
    config = attr.ib()
    config_args = attr.ib()


@attr.s
class TrainConfig:
    exp_config = attr.ib()
    config = attr.ib()
    config_args = attr.ib()
    logdir = attr.ib()


@attr.s
class MetaTrainConfig:
    exp_config = attr.ib()
    config = attr.ib()
    config_args = attr.ib()
    logdir = attr.ib()


@attr.s
class MetaTestConfig:
    config = attr.ib()
    config_args = attr.ib()
    logdir = attr.ib()


@attr.s
class InferConfig:
    config = attr.ib()
    config_args = attr.ib()
    logdir = attr.ib()
    section = attr.ib()
    beam_size = attr.ib()
    output = attr.ib()
    step = attr.ib()
    debug = attr.ib(default=False)
    method = attr.ib(default="beam_search")
    mode = attr.ib(default="infer")
    limit = attr.ib(default=None)
    output_history = attr.ib(default=False)


@attr.s
class EvalConfig:
    config = attr.ib()
    config_args = attr.ib()
    logdir = attr.ib()
    section = attr.ib()
    inferred = attr.ib()
    output = attr.ib()
    etype = attr.ib(default="match")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=["preprocess", "train", "eval", "meta_train", "eval_only", "batched_eval", "predict"],
        help="preprocess/train/eval/meta_train/eval_only/batched_eval",
    )
    parser.add_argument("exp_config_file", help="jsonnet file for experiments")
    parser.add_argument("--config_args", help="exp configs")
    args = parser.parse_args()
    return args


def load_args(args):
    # 1. exp config
    if args.config_args:
        exp_config = json.loads(
            _jsonnet.evaluate_file(
                args.exp_config_file, tla_codes={"args": args.config_args}
            )
        )
    else:
        # empty args make it compatible with non-function exp file
        exp_config = json.loads(_jsonnet.evaluate_file(args.exp_config_file))

    # pretend that model_config_args is parsed through command line
    model_config_args = exp_config["model_config_args"]
    model_config_args = json.dumps(model_config_args)

    return exp_config, model_config_args


def eval_and_report(args, exp_config, model_config_args, logdir, infer_mod):
    model_config_file = exp_config["model_config"]

    summary = collections.defaultdict(float)
    for step in exp_config["eval_steps"]:
        infer_output_path = "{}/{}-step{}.infer".format(
            exp_config["eval_output"], exp_config["eval_name"], step
        )

        # eval_only will not run the infer
        if args.mode != "eval_only":
            infer_config = InferConfig(
                model_config_file,
                model_config_args,
                logdir,
                exp_config["eval_section"],
                exp_config["eval_beam_size"],
                infer_output_path,
                step,
                debug=exp_config["eval_debug"],
                method=exp_config["eval_method"],
            )

            try:
                infer_mod.main(infer_config)
            except infer.CheckpointNotFoundError as e:
                print(f"Infer error {str(e)}")
                continue

        eval_output_path = "{}/{}-step{}.eval".format(
            exp_config["eval_output"], exp_config["eval_name"], step
        )
        eval_config = EvalConfig(
            model_config_file,
            model_config_args,
            logdir,
            exp_config["eval_section"],
            infer_output_path,
            eval_output_path,
        )

        # etype
        if "eval_type" in exp_config:
            eval_config.etype = exp_config["eval_type"]
        else:
            assert eval_config.etype == "match"

        try:
            metrics = eval.main(eval_config)
        except infer.CheckpointNotFoundError as e:
            print(f"Eval error {str(e)}")
            continue

        # commit with step
        eval_section = exp_config["eval_section"]
        if "all" in metrics["total_scores"]:  # spider
            exact_match = metrics["total_scores"]["all"]["exact"]
            exec_match = metrics["total_scores"]["all"]["exec"]
            print(
                "Step: ",
                step,
                "\texact score:",
                exact_match,
                "\texec score:",
                exec_match
            )

            if exact_match > summary[f"{eval_section}-best-exact_match"]:
                summary[f"{eval_section}-best-exact_match"] = exact_match
                summary[f"{eval_section}-best_exact_match_step"] = step
            if exec_match > summary[f"{eval_section}-best-exec_match"]:
                summary[f"{eval_section}-best-exec_match"] = exec_match
                summary[f"{eval_section}-best_exec_match_step"] = step
        else:  # wikisql, etc
            lf_accuracy = metrics["total_scores"]["lf_accuracy"]
            exe_accuracy = metrics["total_scores"]["exe_accuracy"]
            # wandb.log(
            #     {f"{eval_section}-lf-accuracy": lf_accuracy}, step=step,
            # )
            # wandb.log(
            #     {f"{eval_section}-exe-accuracy": exe_accuracy}, step=step,
            # )
            print(step, metrics["total_scores"])

            if lf_accuracy > summary[f"{eval_section}-best-lf-accuracy"]:
                summary[f"{eval_section}-best-lf-accuracy"] = lf_accuracy
                summary[f"{eval_section}-best_lf_accuracy_step"] = step
            if exe_accuracy > summary[f"{eval_section}-best-exe-accuracy"]:
                summary[f"{eval_section}-best-exe-accuracy"] = exe_accuracy
                summary[f"{eval_section}-best_exe_accuracy_step"] = step

def predict(exp_config, model_config_args, logdir, input_nl, db_id):
    model_config_file = exp_config["model_config"]
    infer_output_path = "{}/{}-step{}.infer".format(
            exp_config["eval_output"], exp_config["eval_name"], exp_config["eval_steps"][0]
    )
    infer_config = InferConfig(
        model_config_file,
        model_config_args,
        logdir,
        exp_config["eval_section"],
        exp_config["eval_beam_size"],
        infer_output_path,
        exp_config["eval_steps"][0],
        debug=exp_config["eval_debug"],
        method=exp_config["eval_method"],
    )
    setup_infer, output_path = infer.setup(infer_config)
    inferer = infer.Inferer(setup_infer)
    pretrained_model = inferer.load_model(logdir, exp_config["eval_steps"][0])
    dataset = registry.construct('dataset', inferer.config['data']['val'])

    for _, schema in dataset.schemas.items():
        pretrained_model.preproc.enc_preproc.preprocess_schema(schema)

    def question(q, db_id):
        spider_schema = dataset.schemas[db_id]
        data_item = SpiderItem(
            text=None,
            code=None,
            schema=spider_schema,
            orig_schema=spider_schema.orig,
            orig={"question": q}
        )
        pretrained_model.preproc.clear_items()
        enc_input = pretrained_model.preproc.enc_preproc.preprocess_item(data_item)
        preproc_data = enc_input, None
        with torch.no_grad():
            return inferer._infer_one(pretrained_model, data_item, preproc_data, beam_size=1)
    
    return question(input_nl, db_id)

def main():
    args = parse_args()
    exp_config, model_config_args = load_args(args)

    # model config file
    model_config_file = exp_config["model_config"]

    # cluster base dir
    log_base_dir = os.environ.get("LOG_BASE_DIR", None)
    if log_base_dir is None:
        # print(f"Using default log base dir {os.getcwd()}")
        logdir = exp_config["logdir"]
    else:
        logdir = os.path.join(log_base_dir, exp_config["logdir"])

    # execute command
    if args.mode == "preprocess":
        preprocess_config = PreprocessConfig(model_config_file, model_config_args)
        preprocess.main(preprocess_config)
    elif args.mode == "train":
        train_config = TrainConfig(model_config_file, model_config_args, logdir)
        train.main(train_config)
    elif args.mode == "meta_train":
        train_config = MetaTrainConfig(exp_config, model_config_file, model_config_args, logdir)
        meta_train.main(train_config)
    elif args.mode in ["eval", "eval_only"]:
        eval_and_report(args, exp_config, model_config_args, logdir, infer_mod=infer)
    elif args.mode == "batched_eval":
        eval_and_report(args, exp_config, model_config_args, logdir, infer_mod=batched_infer)
    elif args.mode == "predict":
        db_id = input('enter database name: ')
        input_nl = input('enter vietnamese question: ')
        decoded = predict(exp_config, model_config_args, logdir, input_nl, db_id)
        print(f'predicted query = {decoded[0]["inferred_code"]}')
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
