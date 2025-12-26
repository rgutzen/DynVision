# SEED = ["1000", "1001", "1002"] #, "1003", "1004","1005"]  # energy loss weight 0.05 & use_ffcv=False
# SEED = ["2000", "2001", "2002"] # energy loss weight 0.2 & use_ffcv=True
# SEED = ["3000", "3001", "3002"] # energy loss weight not applied? & use_ffcv=True
# SEED = ["4000", "4001", "4002"] # energy loss weight 0.5 & use_ffcv=True
# SEED = ["5000"] #, "5001", "5002"] # ~not! uses shuffled patterns~ 

# ###############
# for cmd in \
#     "rctarget" \
#     "lossrt" \
#     "idle" \
#     "skip" \
# ; do
#     sh snakecharm.sh "$cmd"
#     sleep 120
# done
#     "rctype" \
#     "tsteps" \
#     "timeparams" \
    # "feedback --config batch_size=128" \
#     "stability --config test_batch_size=16" \
#     "references --config test_batch_size=32" \
#     "unrolling --config batch_size=128" \
#     "energyloss --config epochs=350" \
#     "dataloader --config epochs=100" \
#     "training --allowed-rules test_model process_test_data plot_responses" \
# # sh snakecharm.sh "imagenet --config use_distributed_mode=True batch_size=512"
# ###############

# set in config_workflow.yaml
DEFAULT_MODEL_ARGS = OrderedDict(**config.model_args)
STATUS = ['trained-best']
SEED = config.seed if isinstance(config.seed, list) else [config.seed]

rule train_localdepthwise:
    input:
        expand(project_paths.models / "{model_name}" / "{model_name}{model_args}_{seed}" / "{data_name}" / "{status}.pt",
            model_name=config.model_name,
            model_args=args_product(DEFAULT_MODEL_ARGS | {'rctype': 'localdepthwise'}),
            seed=config.seed,
            data_name=config.data_name,
            status='trained',
        )

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


rule recreate_noise_results:  # run with --allowed-rules test_model process_test_data
    input:
        project_paths.reports / "uniformnoise" / "DyRCNNx8:tsteps=20+rctype=full+rctarget=*+dt=2+tau=5+tff=0+trc=6+tsk=0+lossrt=4_0040" / "imagenette:all_trained" / "test_data.csv",


# LOADER_VARIATIONS = [
#     dict(tsteps=20, dloader="{dloader}", dsteps=1, pattern="1"),  # unroll in model steps
#     dict(tsteps=1, dloader="{dloader}", dsteps=20, pattern="1"),  # unroll in data loader
#     # dict(tff=0, trc=6, tsk=0, tfb=30),   # engineering time unrolling
#     # dict(tff=10, trc=6, tsk=20, tfb=10), # biological time unrolling
# ]

# rule train_with_dataloader_variations: # --config epochs=100
#     # run both with --config use_ffcv=True & --config use_ffcv=False
#     # and rename dloader=* in output filename to ffcv or torch respectively
#     input:
#         [expand(project_paths.models / "{model_name}" / "{model_name}{model_args}_{seed}" / "{data_name}" / "{status}.pt",
#             model_name=config.model_name,
#             model_args=args_product(DEFAULT_MODEL_ARGS | {category: config.experiment_config['categories'][category]}),
#             seed=config.seed,
#             data_name=config.data_name,
#             status=config.status,
#         ) for category in config.category]
#     params:
#         dloader = "ffcv" if config.use_ffcv else "torch"
#     shell:
#         """
#         for file in {input}; do
#             target="${file//\\{dloader\\}/{params.dloader}}"
#             if [ "$file" != "$target" ]; then
#                 mv "$file" "$target"
#             fi
#         done
#         """

rule manuscript_figures: # manuscript figures
# sh snakecharm.sh "manuscript_figures --allowed-rules plot_dynamics plot_responses plot_timeparams_tripytch plot_connection_tripytch plot_timestep_tripytch plot_training plot_performance"
    input:
        # TRAINING
        expand(project_paths.figures / "{experiment}" / "{model_name}{model_args}_{seed}" / "{data_name}:{data_group}_{status}" / "{plot}.png",
            experiment="stability",
            model_name=config.model_name,
            model_args=args_product(DEFAULT_MODEL_ARGS | {"energyloss": "*"}),
            seed=config.seed + [".".join(config.seed)],
            data_name=config.data_name,
            data_group=config.data_group,
            status=config.status,
            plot="training",
        ),
        # TIMEPARAMS TRIPYTCH
        expand(project_paths.figures / "{experiment}" / "{model_name}{model_args}_{seed}" / "{data_name}:{data_group}_{status}" / "{plot}.png",
            experiment="response",
            model_name=config.model_name,
            model_args=args_product(DEFAULT_MODEL_ARGS | {"tau": "*", "trc": "*", "tsk": "*"}),
            seed=config.seed + [".".join(config.seed)],
            data_name=config.data_name,
            data_group=config.data_group,
            status=config.status,
            plot="response_tripytch",
        ),
        # CONNECTION TRIPYTCH
        expand(project_paths.figures / "{experiment}" / "{model_name}{model_args}_{seed}" / "{data_name}:{data_group}_{status}" / "{plot}.png",
            experiment="response",
            model_name=config.model_name,
            model_args=args_product(DEFAULT_MODEL_ARGS | {"rctarget": "*", "skip": "*", "feedback": "*"}),
            seed=config.seed + [".".join(config.seed)],
            data_name=config.data_name,
            data_group=config.data_group,
            status=config.status,
            plot="response_tripytch",
        ),
        # TIMESTEP TRIPYTCH
        expand(project_paths.figures / "{experiment}" / "{model_name}{model_args}_{seed}" / "{data_name}:{data_group}_{status}" / "{plot}.png",
            experiment="response",
            model_name=config.model_name,
            model_args=args_product(DEFAULT_MODEL_ARGS | {"tsteps": "*", "lossrt": "*", "idle": "*"}),
            seed=config.seed + [".".join(config.seed)],
            data_name=config.data_name,
            data_group=config.data_group,
            status=config.status,
            plot="response_tripytch",
        ),
        # NEURAL DYNAMICS
        expand(project_paths.figures / "{experiment}" / "{model_name}{model_args}_{seed}" / "{data_name}:{data_group}_{status}" / "{plot}.png",
            experiment="response",
            model_name=config.model_name,
            model_args=args_product(DEFAULT_MODEL_ARGS | {"rctype": "*"}),
            seed=config.seed + [".".join(config.seed)],
            data_name=config.data_name,
            data_group=config.data_group,
            status=config.status,
            plot=[f"dynamics_{focus_layer}" for focus_layer in ['V1', 'V2', 'V4', 'IT']],
        ),
        # NOISE PERFORMANCE
        expand(project_paths.figures / "{experiment}" / "{model_name}{model_args}_{seed}" / "{data_name}:{data_group}_{status}" / "{plot}.png",
            experiment=["uniformnoise", "poissonnoise", "gaussiannoise", "gaussiancorrnoise"],
            model_name=config.model_name,
            model_args=args_product(DEFAULT_MODEL_ARGS | {"rctarget": "*"}),
            seed=config.seed + [".".join(config.seed)],
            data_name=config.data_name,
            data_group=config.data_group,
            status=config.status,
            plot="performance",
        ),
        # REFERENCE MODELS
        expand(project_paths.figures / "{experiment}" / "{model}:{model_args}_{seeds}" / "imagenet:imagenette_init" / "{plot}.png",
            experiment=['response', 'idleresponse', 'hundred'],
            model=['CorNetRT', 'CordsNet'],
            model_args='pretrained=*',
            seeds=SEED[0],
            plot='responses',
        ),
    params:
        figure_folder = project_paths.figures / "manuscript_figures"
    shell:
        """
        mkdir -p {params.figure_folder}
        if [ -n "{input}" ]; then
            for file in {input}; do
            folder_name=$(basename "$(dirname "$file")")
            file_name=$(basename "$file")
            cp "$file" "{params.figure_folder}/${{folder_name}}_${{file_name}}"
            done
        fi
        """

rule unrolling:
    input:
        expand(project_paths.models / "DyRCNNx8" / "DyRCNNx8:tsteps=30+rctype=full+rctarget=output+dt=2+tau=5+tff=10+trc=6+tsk=20+tfb=10+skip=true+feedback=true+lossrt=4_{seed}" / "imagenette" / "{status}.pt",
        seed=SEED,
        status=STATUS)

rule dataloader:
    input:
        expand(project_paths.figures / 'response' / 'DyRCNNx8:pattern=1+{configuration}_{seed}' / 'imagenette:all_{status}' / 'responses.png',
        seed=SEED,
        status=STATUS,
        configuration=[
            'tsteps=20+dataloader=*+dsteps=1',
            'tsteps=1+dataloader=*+dsteps=20',
        ]),

rule imagenet:  # run with --config use_distributed_mode=True
    input:
        expand(project_paths.models / "{model_name}" / "{model_name}{model_args}_{seed}" / "{data_name}" / "{status}.pt",
            model_name=config.model_name,
            model_args=args_product(DEFAULT_MODEL_ARGS | {'tsteps': 10}),
            seed=config.seed,
            data_name='imagenet',
            status='trained',
        ),
