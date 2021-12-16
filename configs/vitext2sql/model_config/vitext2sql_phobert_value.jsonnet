local spider_base = import "vitext2sql_base.jsonnet";

function(args) spider_base(args, data_path=args.data_path) {
    local data_path = args.data_path,
    local base_bert_enc_size = if args.bert_version == "vinai/phobert-large" then 1024 else 768,
    local enc_size =  base_bert_enc_size,
    
    data: {
        local PREFIX = data_path,
        local ts = if $.args.use_other_train then
            ['vitext2sql']
        else
            ['vitext2sql'],

        train: {
            name: 'vitext2sql', 
            paths: [
              PREFIX + 'train_%s.json' % [s]
              for s in ts],
            tables_paths: [
              PREFIX + 'tables.json',
            ],
            db_path: PREFIX + 'database',
        },
        val: {
            name: 'vitext2sql', 
            paths: [PREFIX + 'dev.json'],
            tables_paths: [PREFIX + 'tables.json'],
            db_path: PREFIX + 'database',
        },

    },
    model+:{
        encoder: {
            name: 'vitext2sql-phobert',
            bert_version: $.args.bert_version,
            bert_token_type: $.args.bert_token_type,
            linking_config: {
                name: "spider_string_matching",
            },
            rat_config: {
                name: "rat",
                num_heads: 8,
                num_layers: $.args.num_layers,
                enable_latent_relations: false,
            },
        },
        encoder_preproc: {
            context: {
                "name": "vitext2sql-phobert",
                db_paths: data_path + "database",
            },
            bert_version: $.model.encoder.bert_version,
            compute_sc_link: $.args.sc_link,
            compute_cv_link: $.args.cv_link,
            save_path: data_path + 'vitext2sql-phobert-%s,other_train-%s,sc_link=%s,cv_link=%s' % [self.bert_version, $.args.use_other_train, $.args.sc_link, $.args.cv_link],
        },

        decoder+: {
            enc_recurrent_size: enc_size,
            loss_type: $.args.loss_type,
        },

        decoder_preproc+: {
            min_freq: 32,
            value_tokenizer: $.model.encoder_preproc.bert_version,
        },

    },

    optimizer: {
        name: $.args.opt,
        lr: 0.0,
        phobert_lr: 0.0,
    },

    lr_scheduler+: {
        name: $.args.lr_scheduler,
        start_lrs: [$.args.lr, $.args.phobert_lr],
        end_lr: $.args.end_lr,
        num_warmup_steps: $.train.max_steps / 8,
    },
}
