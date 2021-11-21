{
    local exp_id = 0,
    project: "vitext2sql_value",
    logdir: "log/vitext2sql/phobert_value_%d" %exp_id,
    model_config: "../../configs/vitext2sql/model_config/vitext2sql_phobert_value.jsonnet",
    model_config_args: {
        # data 
        data_path: 'data/vitext2sql_syllable_level/',
        use_other_train: true,

        # model
        num_layers: 6,
        sc_link: true,
        cv_link: true,
        loss_type: "softmax", # softmax, label_smooth

        # phobert, opimizer
        opt: "phobertAdamw",   # bertAdamw, torchAdamw
        lr_scheduler: "phobert_warmup_polynomial_group", # bert_warmup_polynomial_group,bert_warmup_polynomial_grou_v2
        bert_token_type: false,
        bert_version: "vinai/phobert-large",
        phobert_lr: 2e-5, 

        # grammar
        include_literals: true,

        # training
        bs: 3,
        att: 0,
        lr: 5e-4,
        clip_grad: 0.3,
        num_batch_accumulated: 4,
        max_steps: 20000,
        save_threshold: 19000,
        use_bert_training: true,
        device: "cuda:0",

        # meta train
        meta_train_opt: "sgd",
        meta_train_lr: 1e-4,
        num_batch_per_train: 3,
        data_scheduler: "db_scheduler",
    },

    eval_section: "val",
    eval_type: "all", # match, exec, all
    eval_method: "spider_beam_search_with_heuristic",
    eval_output: "ie_dir/spider_value",
    eval_beam_size: 3,
    eval_debug: false,
    eval_name: "bert_run_%d_%s_%s_%d_%d" % [exp_id, self.eval_section, self.eval_method, self.eval_beam_size, self.model_config_args.att],

    local _start_step = $.model_config_args.save_threshold / 1000,
    local _end_step = $.model_config_args.max_steps / 1000,
    eval_steps: [ 1000 * x for x in std.range(_start_step, _end_step)],
    # eval_steps: [ $.model_config_args.max_steps ],
}
