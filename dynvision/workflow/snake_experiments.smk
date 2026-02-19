# set in config_workflow.yaml
DEFAULT_MODEL_ARGS = OrderedDict(**config.model_args)
SEED = config.seed if isinstance(config.seed, list) else [config.seed]


rule train_model_variations:  # call with --config category=<category_name>
    """
    Call with --config category=<category_name> to specify the model attributes to vary.
    The category values are taken from config.experiment_config['categories'].
    """
    input:
        [expand(project_paths.models / "{model_name}" / "{model_name}{model_args}_{seed}" / "{data_name}" / "{status}.pt",
            model_name=config.model_name,
            model_args=args_product(DEFAULT_MODEL_ARGS | {category: config.experiment_config['categories'][category]}),
            seed=config.seed,
            data_name=config.data_name,
            status='trained',
        ) for category in (config.category if isinstance(config.category, list) else [config.category])]
    # params:
    #     symlink_with_dataloader_name=True,
    #     dloader="ffcv" if config.use_ffcv else "torch"
    # shell:
    #     """
    #     if [ "{params.symlink_with_dataloader_name}" = "True" ]; then
    #         for file in {input}; do
    #             model_seed_dir=$(dirname "$(dirname "$file")")
    #             parent_dir=$(dirname "$model_seed_dir")
    #             base_name=$(basename "$model_seed_dir")
    #             seed=${base_name##*_}
    #             prefix=${base_name%_*}
    #             symlink_path="${{parent_dir}}/${{prefix}}+dloader={params.dloader}_${{seed}}"
    #             if [ ! -e "$symlink_path" ]; then
    #                 ln -s "$model_seed_dir" "$symlink_path"
    #             fi
    #         done
    #     fi
    #     """


rule test_model_variations:  # call with --config category=<category_name>
    """
    Call with --config category=<category_name> experiment=<experiment_name> to specify 
    the model attributes to vary and what testing experiments to run on them.
    Will still trigger model training if the model state is missing or not up to date.
    Test only already trained models with `--allowed-rules test_model process_test_data`.
    """
    input:
        [expand(project_paths.reports / "{experiment}" / "{model_name}{model_args}_{seed}" / "{data_name}:{data_group}_{status}" / "test_data.csv",
            experiment=config.experiment,
            model_name=config.model_name,
            model_args=args_product(DEFAULT_MODEL_ARGS | {category: "*"}),
            seed=config.seed,
            data_name=config.data_name,
            data_group=config.data_group,
            status=config.status,
        ) for category in (config.category if isinstance(config.category, list) else [config.category])]


rule plot_model_variations:  # call with --config category=<category_name> experiment=<experiment_name> plot=<plot_name>
    """
    Call with --config plot=<plot_name> to select the types of plots to generate.
    Will still trigger model training and testing if the their outputs are missing or not up to date.
    Plot only existing test results with `--allowed-rules plot_<plot_name>`.
    """

    input:
        [expand(project_paths.figures / "{experiment}" / "{model_name}{model_args}_{seed}" / "{data_name}:{data_group}_{status}" / "{plot}.png",
            experiment=config.experiment,
            model_name=config.model_name,
            model_args=args_product(DEFAULT_MODEL_ARGS | {category: "*"}),
            seed=config.seed + [".".join(config.seed)],
            data_name=config.data_name,
            data_group=config.data_group,
            status=config.status,
            plot=getattr(config, 'plot', 'responses'),
        ) for category in (config.category if isinstance(config.category, list) else [config.category])]


rule plot_model_variation_weights:  # call with --config category=<variable_name>
    input:
        [expand(project_paths.figures / "weights" / "{model_name}{model_args}_{seed}" / "{data_name}_{status}" / "weights.png",
            model_name=config.model_name,
            model_args=args_product(DEFAULT_MODEL_ARGS | {category: "*"}),
            seed=config.seed,
            data_name=config.data_name,
            status=config.status,
        ) for category in (config.category if isinstance(config.category, list) else [config.category])]



