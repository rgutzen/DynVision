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


rule process_all_wandb_data:
    input:
        [expand(project_paths.reports / "wandb" / "{model_name}{model_args}_{seed}_{metric}_summary.csv",
            model_name=config.model_name,
            model_args=args_product(DEFAULT_MODEL_ARGS | {category: "*"}),
            seed='.'.join(config.seed),
            metric='gpu_mem_alloc', #['epoch', 'gpu_mem_alloc']
        ) for category in (config.category if isinstance(config.category, list) else [config.category])]
    params:
        summary_file = project_paths.reports / "wandb" / f"{config.model_name}{args_product(DEFAULT_MODEL_ARGS)}_{'.'.join(config.seed)}_all_gpu_summary.csv"
    run:
        import pandas as pd
        from pathlib import Path
        
        # Read all input CSV files and concatenate
        if not input:
            raise ValueError("No input files provided")
        
        dfs = [pd.read_csv(f) for f in input]
        aggregated_df = pd.concat(dfs, ignore_index=True)
        
        # Save to output file
        output_path = Path(params.summary_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        aggregated_df.to_csv(output_path, index=False)
        print(f"Aggregated {len(input)} files into {output_path}")
        print(f"Total rows: {len(aggregated_df)}")


rule alt_best_model:
    input:
        expand(project_paths.figures / "{experiment}" / "{model_name}{model_args}_{seed}" / "{data_name}:{data_group}_{status}" / "{plot}.png",
                experiment=['duration', 'interval', 'contrast'],
                model_name=config.model_name,
                model_args=args_product(DEFAULT_MODEL_ARGS | {'pattern':'1011', 'energyloss': "1.0", 'rctarget':'*'}),
                seed='6000',
                data_name=config.data_name,
                data_group=config.data_group,
                status=config.status,
                plot=[f'dynamics_groen_{focus_layer}' for focus_layer in ['V1','V2']]
            ),
        expand(project_paths.figures / "{experiment}" / "{model_name}{model_args}_{seed}" / "{data_name}:{data_group}_{status}" / "{plot}.png",
                experiment=['gaussiannoise'],
                model_name=config.model_name,
                model_args=args_product(DEFAULT_MODEL_ARGS | {'pattern':'1011', 'energyloss': "1.0", 'rctarget':'*'}),
                seed='6000',
                data_name=config.data_name,
                data_group=config.data_group,
                status=config.status,
                plot='performance_manuscript'
            )

# rule recreate_noise_results:  # run with --allowed-rules test_model process_test_data
#     input:
#         project_paths.reports / "uniformnoise" / "DyRCNNx8:tsteps=20+rctype=full+rctarget=*+dt=2+tau=5+tff=0+trc=6+tsk=0+lossrt=4_0040" / "imagenette:all_trained" / "test_data.csv",


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

rule plot_training_figures:
    input:
        expand(project_paths.figures / "{experiment}" / "{model_name}{model_args}_{seed}" / "{data_name}:{data_group}_{status}" / "{plot}.png",
            experiment="stability",
            model_name=config.model_name,
            model_args=args_product(DEFAULT_MODEL_ARGS | {"energyloss": "*", 'pattern': "1"})
                     + args_product(DEFAULT_MODEL_ARGS | {"energyloss": "*", 'pattern': "1011"}),    
            seed=config.seed + [".".join(config.seed)],
            data_name=config.data_name,
            data_group=config.data_group,
            status=config.status,
            plot="training",
        )

rule manuscript_figures: # manuscript figures
# sh snakecharm.sh "manuscript_figures --allowed-rules plot_dynamics plot_responses plot_timeparams_tripytch plot_connection_tripytch plot_timestep_tripytch plot_training plot_performance"
    input:
        # TRAINING  # run with --config test_batch_size=16
        # expand(project_paths.figures / "{experiment}" / "{model_name}{model_args}_{seed}" / "{data_name}:{data_group}_{status}" / "{plot}.png",
        #     experiment="stability",
        #     model_name=config.model_name,
        #     model_args=args_product(DEFAULT_MODEL_ARGS | {"energyloss": "*", 'pattern': "1"})
        #              + args_product(DEFAULT_MODEL_ARGS | {"energyloss": "*", 'pattern': "1011"}),    
        #     seed=config.seed + [".".join(config.seed)],
        #     data_name=config.data_name,
        #     data_group=config.data_group,
        #     status=config.status,
        #     plot="training",
        # ),
        # TIMEPARAMS TRIPYTCH
        expand(project_paths.figures / "{experiment}" / "{model_name}{model_args}_{seed}" / "{data_name}:{data_group}_{status}" / "{plot}.png",
            experiment="response",
            model_name=config.model_name,
            model_args=args_product(DEFAULT_MODEL_ARGS | {"tau": "*", "trc": "*", "tsk": "*"}),
            seed=[".".join(config.seed)] + config.seed,
            data_name=config.data_name,
            data_group=config.data_group,
            status=config.status,
            plot="responses_tripytch",
        ),
        # CONNECTION TRIPYTCH
        expand(project_paths.figures / "{experiment}" / "{model_name}{model_args}_{seed}" / "{data_name}:{data_group}_{status}" / "{plot}.png",
            experiment="response",
            model_name=config.model_name,
            model_args=args_product(DEFAULT_MODEL_ARGS | {"pattern": "*", "skip": "*", "feedback": "*"}),
            seed=[".".join(config.seed)] + config.seed,
            data_name=config.data_name,
            data_group=config.data_group,
            status=config.status,
            plot="responses_tripytch",
        ),
        # TIMESTEP TRIPYTCH
        expand(project_paths.figures / "{experiment}" / "{model_name}{model_args}_{seed}" / "{data_name}:{data_group}_{status}" / "{plot}.png",
            experiment="response",
            model_name=config.model_name,
            model_args=args_product(DEFAULT_MODEL_ARGS | {"tsteps": "*", "lossrt": "*", "idle": "*"}),
            seed=[".".join(config.seed)] + config.seed,
            data_name=config.data_name,
            data_group=config.data_group,
            status=config.status,
            plot="responses_tripytch",
        ),
        # RECURRENCE TRIPYTCH
        expand(project_paths.figures / "{experiment}" / "{model_name}{model_args}_{seed}" / "{data_name}:{data_group}_{status}" / "{plot}.png",
            experiment="response",
            model_name=config.model_name,
            model_args=args_product(DEFAULT_MODEL_ARGS | {"rctype": "*", "rctarget": "*", "dt": "*"}),  # "seed": "*"
            seed=[".".join(config.seed)], # + config.seeds
            data_name=config.data_name,
            data_group=config.data_group,
            status=config.status,
            plot="responses_tripytch",
        ),
        # NEURAL DYNAMICS
        expand(project_paths.figures / "{experiment}" / "{model_name}{model_args}_{seed}" / "{data_name}:{data_group}_{status}" / "{plot}.png",
            experiment=["duration", "contrast", "interval"],
            model_name=config.model_name,
            model_args=args_product(DEFAULT_MODEL_ARGS | {"rctype": "*", "energyloss": "1.0", "pattern": "1011"})
                    #  + args_product(DEFAULT_MODEL_ARGS | {"energyloss": "*", "pattern": "1"})
                     + args_product(DEFAULT_MODEL_ARGS | {"energyloss": "*", "pattern": "1011"}),
            seed=config.seed + [".".join(config.seed)],
            data_name=config.data_name,
            data_group=config.data_group,
            status=config.status,
            plot=[f"dynamics_groen_{focus_layer}" for focus_layer in ['V1', 'V2']],
        ),
        # NOISE PERFORMANCE
        expand(project_paths.figures / "{experiment}" / "{model_name}{model_args}_{seed}" / "{data_name}:{data_group}_{status}" / "{plot}.png",
            experiment=["gaussiannoise"], # "gaussiancorrnoise"], #"phasescramblednoise", 
            model_name=config.model_name,
            model_args=args_product(DEFAULT_MODEL_ARGS | {"rctarget": "*"}) 
                    #  + args_product(DEFAULT_MODEL_ARGS | {"rctype": "*"})
                     + args_product(DEFAULT_MODEL_ARGS | {"energyloss": "*", "rctarget": "middle", "pattern": "1"}),
                    #  + args_product(DEFAULT_MODEL_ARGS | {"energyloss": "*", "pattern": "1011"}),
                    #  + args_product(DEFAULT_MODEL_ARGS | {"feedback": "*"}),
            seed=config.seed + [".".join(config.seed)],
            data_name=config.data_name,
            data_group=config.data_group,
            status=config.status,
            plot="performance_manuscript",
        ),
        # # DEVELOPMENT DURING TRAINING
        # expand(project_paths.figures / "{experiment}" / "{model_name}{model_args}_{seed}" / "{data_name}:{data_group}_{status}" / "{plot}.png",
        #     experiment="responseintermediate",
        #     model_name=config.model_name,
        #     model_args=args_product(DEFAULT_MODEL_ARGS | {"energyloss": "*", "pattern": "1"})
        #              + args_product(DEFAULT_MODEL_ARGS | {"energyloss": "*", "pattern": "1011"}),
        #     seed=config.seed + [".".join(config.seed)],
        #     data_name=config.data_name,
        #     data_group=config.data_group,
        #     status=config.status,
        #     plot='responses', # set hue=epoch
        # ),
        # # REFERENCE MODELS
        # expand(project_paths.figures / "{experiment}" / "{model}:{model_args}_{seeds}" / "imagenet:imagenette_init" / "{plot}.png",
        #     experiment=['response', 'idleresponse', 'hundred'],
        #     model=['CorNetRT', 'CordsNet'],
        #     model_args='pretrained=*',
        #     seeds=SEED[0],
        #     plot='responses',
        # ),
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

rule plot_performance_manuscript:
    """Plot performance metrics with Jang et al. (2021) benchmarks.

    This rule generates publication-ready performance figures including:
    - Panel A: Performance traces over time (multiple subplots)
    - Panel B: Max accuracy vs parameter for models
    - Panel C: Jang et al. (2021) human and DNN benchmarks

    Input/Output follow pattern:
    - Input: {experiment}/{model_identifier}/{data_name}:{data_group}_{status}/test_data.csv
    - Output: {experiment}/{model_identifier}/{data_name}:{data_group}_{status}/performance_manuscript.png
    """
    input:
        data = expand(
            project_paths.reports
            / '{{experiment}}'
            / '{{model_name}}{{args1}}{{category}}=*{{args2}}_{seed}'
            / '{{data_name}}:{{data_group}}_{{status}}'
            / 'test_data.csv',
            seed=lambda w: w.seeds.split('.'),
        ),
        data_ffonly = expand(
            project_paths.reports
            / '{{experiment}}ffonly'
            / '{{model_name}}{{args1}}{{category}}=*{{args2}}_{seed}'
            / '{{data_name}}:{{data_group}}_{{status}}'
            / 'test_data.csv',
            seed=lambda w: w.seeds.split('.'),
        ),
        data_feedforward = expand(
            project_paths.reports
            / '{{experiment}}'
            / '{{model_name}}{model_args}_{seed}'
            / '{{data_name}}:{{data_group}}_{{status}}'
            / 'test_data.csv',
            model_args=args_product(DEFAULT_MODEL_ARGS | {"rctype": "none", "dt": "*"}),
            seed=lambda w: w.seeds.split('.'),
        ),
        script = SCRIPTS / 'visualization' / 'plot_performance_manuscript.py'
    params:
        # data_ffonly = lambda w: [project_paths.reports / f'{w.experiment}ffonly' / f'{w.model_name}{w.args1}{w.category}=*{w.args2}_{seed}' / f'{w.data_name}:{w.data_group}_{w.status}' / 'test_data.csv' for seed in w.seeds.split('.')],
        data_category2 = expand(
            project_paths.reports
            / '{experiment}'
            / '{model_name}{model_args}_{seed}'
            / '{data_name}:{data_group}_{status}'
            / 'test_data.csv',
            experiment=lambda w: w.experiment,
            model_name=lambda w: w.model_name,
            data_name=lambda w: w.data_name,
            data_group=lambda w: w.data_group,
            status=lambda w: w.status,
            model_args=args_product(DEFAULT_MODEL_ARGS | {"energyloss": "*", "pattern": "1011", "rctarget": "middle"}),
            seed=lambda w: w.seeds.split('.'),
        ),
        row = None,
        subplot = 'parameter',
        hue = 'category',
        parameter = lambda w: config.experiment_config[w.experiment]['parameter'],
        category = lambda w: w.category,
        category2 = 'energyloss',
        accuracy_measure = getattr(config, 'plot_accuracy_measure', "accuracy"),
        confidence_measure = getattr(config, 'plot_confidence_measure', "none"),
        jang_noise_type = getattr(config, 'jang_noise_type', 'gaussian'),
        dt = getattr(config, 'dt', 2),
        palette = lambda w: json.dumps(config.palette),
        naming = lambda w: json.dumps(config.naming),
        ordering = lambda w: json.dumps(config.ordering),
        subplot_filter = [0.1, 0.4, 0.7, 1.0],
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
        ),
    output:
        project_paths.figures / '{experiment}' / '{model_name}{args1}{category}=*{args2}_{seeds}' / '{data_name}:{data_group}_{status}' / 'performance_manuscript.png',
    # group: "visualization"
    shell:
        """
        {params.execution_cmd} \
            --data {input.data:q} \
            --data-ffonly {input.data_ffonly:q} \
            --data-feedforward {input.data_feedforward:q} \
            --data-category2 {params.data_category2:q} \
            --output {output:q} \
            --row {params.row} \
            --subplot {params.subplot} \
            --hue {params.hue} \
            --parameter-key {params.parameter} \
            --category-key {params.category} \
            --category2-key {params.category2} \
            --experiment-names {wildcards.experiment} \
            --accuracy-measure {params.accuracy_measure} \
            --confidence-measure {params.confidence_measure} \
            --jang-noise-type {params.jang_noise_type} \
            --dt {params.dt} \
            --palette {params.palette:q} \
            --naming {params.naming:q} \
            --ordering {params.ordering:q} \
            --subplot-filter {params.subplot_filter} \
        """
            # --plot-individual-seeds


rule plot_dynamics_manuscript:
    """Plot dynamics with Groen et al. 2022 empirical data (hierarchical structure).

    This rule generates 4-panel dynamics figures that include:
    - Panel A: Ridge plots of different layers at reference parameter value
    - Panel B: Ridge plots of focus layer across parameter values
    - Panel C: Groen et al. (2022) empirical V1 data (duration/contrast/interval)
    - Panel D: Summary metrics plot

    The Groen data shown in panel C depends on the experiment type:
    - Duration experiment → Figure 2B (Temporal Summation)
    - Interval experiment → Figure 3B (Recovery from Adaptation)
    - Contrast experiment → Figure 4B (Time-to-Peak)
    """
    input:
        data = expand(
            project_paths.reports
            / '{{experiment}}'
            / '{{model_name}}{{args1}}{{category}}=*{{args2}}_{seeds}'
            / '{{data_name}}:{{data_group}}_{{status}}'
            / 'test_data.csv',
            seeds=lambda w: w.seeds.split('.'),
        ),
        script = SCRIPTS / 'visualization' / 'plot_dynamics_with_groen.py',
        groen_data = lambda w: [
            project_paths.data.external / "groen2022_csv" / f"groen2022_{w.experiment}_data.csv"
        ] if w.experiment in ['duration', 'contrast', 'interval'] else []
    params:
        data_category2 = expand(
            project_paths.reports
            / '{experiment}'
            / '{model_name}{model_args}_{seed}'
            / '{data_name}:{data_group}_{status}'
            / 'test_data.csv',
            experiment=lambda w: w.experiment,
            model_name=lambda w: w.model_name,
            data_name=lambda w: w.data_name,
            data_group=lambda w: w.data_group,
            status=lambda w: w.status,
            model_args=args_product(DEFAULT_MODEL_ARGS | {"energyloss": "*", "pattern": "1011", "rctarget": "output"}),
            seed=lambda w: w.seeds.split('.'),
        ),
        category2 = "energyloss",
        parameter = lambda w: config.experiment_config[w.experiment]['parameter'],
        dt = getattr(config, 'dt', 2),
        palette = lambda w: json.dumps(config.palette),
        naming = lambda w: json.dumps(config.naming),
        ordering = lambda w: json.dumps(config.ordering),
        # subplot_filter = [1, 3, 5, 10, 15, 20]
        groen_data_dir = project_paths.data.external / "groen2022_csv",
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
        ),
    output:
        project_paths.figures / '{experiment}' / '{model_name}{args1}{category}=*{args2}_{seeds}' / '{data_name}:{data_group}_{status}' / 'dynamics_groen_{focus_layer}.png',
    # group: "visualization"
    shell:
        """
        {params.execution_cmd} \
            --data {input.data:q} \
            --data-category2 {params.data_category2:q} \
            --category2 {params.category2} \
            --output {output:q} \
            --parameter {params.parameter} \
            --experiment {wildcards.experiment} \
            --category {wildcards.category} \
            --focus-layer {wildcards.focus_layer} \
            --dt {params.dt} \
            --palette {params.palette:q} \
            --naming {params.naming:q} \
            --ordering {params.ordering:q} \
            --groen-data {params.groen_data_dir}
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
