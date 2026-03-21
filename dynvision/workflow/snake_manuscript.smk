# SEED = ["1000", "1001", "1002"] #, "1003", "1004","1005"]  # activity loss weight 0.05 & use_ffcv=False
# SEED = ["2000", "2001", "2002"] # activity loss weight 0.2 & use_ffcv=True
# SEED = ["3000", "3001", "3002"] # activity loss weight not applied? & use_ffcv=True
# SEED = ["4000", "4001", "4002"] # activity loss weight 0.5 & use_ffcv=True
# SEED = ["5000"] #, "5001", "5002"] # ~not! uses shuffled patterns~ 
# SEED = 7000+ uses ffcv=False, activityloss on all timesteps
# SEED = 8000+ uses ffcv=False, activityloss on relu output
# SEED = 9000 uses ffcv=False, signed activity loss on pre-relu activity
# SEED = 9010 absolute activity loss on pre-relu activity
# SEED = 9020 absolute activity loss on recurrence output
# SEED = 9100+ uses ffcv=False, signed activity loss on recurrent output

# Figures naming convention:
# {plot_type}_{category}=*{+model_args}_{focus_layer}_{seed}.png

rule figures_with_updated_local:
    input:
        # NEURAL DYNAMICS
        expand(project_paths.figures / "{experiment}" / "{model_name}{model_args}_{seed}" / "{data_name}:{data_group}_{status}" / "{plot}.png",
            # experiment=["duration", "contrast", "interval"],
            experiment=["dynamics"],
            model_name=config.model_name,
            model_args=args_product(DEFAULT_MODEL_ARGS | {"rctype": "*", "activityloss": "1.0", "pattern": "1011"}),
            seed=config.seed + [".".join(config.seed)],
            data_name=config.data_name,
            data_group=config.data_group,
            status=config.status,
            # plot=[f"dynamics_groen_{focus_layer}" for focus_layer in ['V1', 'V2']],
            plot=[f"dynamics_{focus_layer}_v_groen+activityloss" for focus_layer in ['V2+V2+V1', 'V1+V1+V2']]
        ),

rule figure_unrolling:
    input:
        # UNROLLING
        expand(project_paths.figures / "unrolling" / "{model_name}{model_args}_{seed}" / "{data_name}:{data_group}_{status}" / "responses_tff={tff}+tsk={tsk}+tfb={tfb}.png",
            model_name=config.model_name,
            model_args=args_product(DEFAULT_MODEL_ARGS | {"feedback": "add"}),
            seed=7001,
            data_name=config.data_name,
            data_group='one',
            status=config.status,
            tff=[10], # 0 in engineering time
            tsk=[20], # 0 in engineering time
            tfb=[10], # 30 in engineering time
        ),

rule reference_models:
    input:
        # REFERENCE MODELS
        expand(project_paths.figures / "{experiment}" / "{model}:{model_args}_{seeds}" / "imagenet:all_init" / "{plot}.png",
            experiment=['response', 'idleresponse', 'hundred'],
            model=['CorNetRT', 'CordsNet'],
            model_args='pretrained=*',
            seeds="9000",
            plot='responses',
        ),

rule manuscript_figures: # manuscript figures
# sh snakecharm.sh "manuscript_figures --allowed-rules plot_dynamics plot_responses plot_timeparams_tripytch plot_connection_tripytch plot_timestep_tripytch plot_training plot_performance"
    input:
        # TRAINING  
        expand(project_paths.figures / "{experiment}" / "{model_name}{model_args}_{seed}" / "{data_name}:{data_group}_{status}" / "{plot}.png",
            experiment="response",
            model_name=config.model_name,
            model_args=args_product(DEFAULT_MODEL_ARGS | {"activityloss": "*", 'pattern': "1"})
                     + args_product(DEFAULT_MODEL_ARGS | {"activityloss": "*", 'pattern': "1011"}),    
            seed=config.seed + [".".join(config.seed)],
            data_name=config.data_name,
            data_group=config.data_group,
            status=config.status,
            plot="training",
        ),
        # # TIMEPARAMS TRIPYTCH
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
            model_args=args_product(DEFAULT_MODEL_ARGS | {"rctype": "*", "rctarget": "*", "seed": "*"}),  # "seed": "*"
            seed=[".".join(config.seed)], # + config.seeds
            data_name=config.data_name,
            data_group=config.data_group,
            status=config.status,
            plot="responses_tripytch",
        ),
        # NEURAL DYNAMICS
        expand(project_paths.figures / "{experiment}" / "{model_name}{model_args}_{seed}" / "{data_name}:{data_group}_{status}" / "{plot}.png",
            # experiment=["duration", "contrast", "interval"],
            experiment=["dynamics"],
            model_name=config.model_name,
            model_args=args_product(DEFAULT_MODEL_ARGS | {"rctype": "*", "activityloss": "1.0", "pattern": "1011"}) 
                     + args_product(DEFAULT_MODEL_ARGS | {"rctarget": "*", "activityloss": "1.0", "pattern": "1011"}),   
            seed=config.seed + [".".join(config.seed)],
            data_name=config.data_name,
            data_group=config.data_group,
            status=config.status,
            # plot=[f"dynamics_groen_{focus_layer}" for focus_layer in ['V1', 'V2']],
            plot=[f"dynamics_{focus_layer}_v_groen+activityloss" for focus_layer in ['V2+V2+V1', 'V1+V1+V2']]
        ),
        # NOISE PERFORMANCE
        expand(project_paths.figures / "{experiment}" / "{model_name}{model_args}_{seed}" / "{data_name}:{data_group}_{status}" / "{plot}.png",
            experiment=["gaussiannoise"], # "gaussiancorrnoise"], #"phasescramblednoise", 
            model_name=config.model_name,
            model_args=args_product(DEFAULT_MODEL_ARGS | {"rctarget": "*", "pattern": "1"})
                     + args_product(DEFAULT_MODEL_ARGS | {"feedback": "*", "rctarget": "middle"})
                     + args_product(DEFAULT_MODEL_ARGS | {"rctype": "*", "rctarget": "middle"}),
            seed=config.seed + [".".join(config.seed)],
            data_name=config.data_name,
            data_group=config.data_group,
            status=config.status,
            plot="performance_manuscript",
        ),
        # # DEVELOPMENT DURING TRAINING
        expand(project_paths.figures / "{experiment}" / "{model_name}{model_args}_{seed}" / "{data_name}:{data_group}_{status}" / "{plot}.png",
            experiment="responseintermediate",
            model_name=config.model_name,
            model_args=args_product(DEFAULT_MODEL_ARGS | {"activityloss": "*", "pattern": "1"})
                     + args_product(DEFAULT_MODEL_ARGS | {"activityloss": "*", "pattern": "1011"}),
            seed=config.seed + [".".join(config.seed)],
            data_name=config.data_name,
            data_group=config.data_group,
            status=config.status,
            plot='responses', # set hue=epoch
        ),
        # WEIGHTS
         expand(project_paths.figures / "weights" / "{model_name}{model_args}_{seed}" / "{data_name}_{status}" / "weights.png",
            model_name=config.model_name,
            model_args=args_product(DEFAULT_MODEL_ARGS | {"rctarget": "*"})
                     + args_product(DEFAULT_MODEL_ARGS | {"rctarget": "middle", "activityloss": "*"}) 
                     + args_product(DEFAULT_MODEL_ARGS | {"rctype": "*", "activityloss": "1.0", "pattern": "1011"}) 
                     + args_product(DEFAULT_MODEL_ARGS | {"rctype": "full", "activityloss": "*", "pattern": "1011"})
                     + args_product(DEFAULT_MODEL_ARGS | {"feedback": "*"}),
            seed=config.seed,
            data_name=config.data_name,
            status=config.status,
        ),
        # UNROLLING
        expand(project_paths.figures / "unrolling" / "{model_name}{model_args}_{seed}" / "{data_name}:{data_group}_{status}" / "responses_tff={tff}+tsk={tsk}+tfb={tfb}.png",
            model_name=config.model_name,
            model_args=args_product(DEFAULT_MODEL_ARGS | {"feedback": "add"}),
            seed=config.seed[-1],
            data_name=config.data_name,
            data_group='one',
            status=config.status,
            tff=[10], # 0 in engineering time
            tsk=[20], # 0 in engineering time
            tfb=[10], # 30 in engineering time
        ),
        # STABILITY
        expand(project_paths.figures / "{experiment}" / "{model_name}{model_args}_{seed}" / "{data_name}:{data_group}_{status}" / "{plot}.png",
            experiment="stability",
            model_name=config.model_name,
            model_args=args_product(DEFAULT_MODEL_ARGS | {"activityloss": "*", 'pattern': "1"})
                     + args_product(DEFAULT_MODEL_ARGS | {"activityloss": "*", 'pattern': "1011"})
                     + args_product(DEFAULT_MODEL_ARGS | {"rctarget": "*"}),
            seed=config.seed + [".".join(config.seed)],
            data_name=config.data_name,
            data_group=config.data_group,
            status=config.status,
            plot="responses",
        ),
        # # REFERENCE MODELS
        # expand(project_paths.figures / "{experiment}" / "{model}:{model_args}_{seeds}" / "imagenet:imagenette_init" / "{plot}.png",
        #     experiment=['response', 'idleresponse', 'hundred'],
        #     model=['CorNetRT', 'CordsNet'],
        #     model_args='pretrained=*',
        #     seeds=SEED[0],
        #     plot='responses',
        # ),


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


rule plot_stability: # run with --config test_batch_size=16
    input:
        expand(project_paths.figures / "{experiment}" / "{model_name}{model_args}_{seed}" / "{data_name}:{data_group}_{status}" / "{plot}.png",
            experiment="stability",
            model_name=config.model_name,
            model_args=args_product(DEFAULT_MODEL_ARGS | {"activityloss": "*", 'pattern': "1"})
                     + args_product(DEFAULT_MODEL_ARGS | {"activityloss": "*", 'pattern': "1011"})
                     + args_product(DEFAULT_MODEL_ARGS | {"rctarget": "*"}), 
            seed=config.seed + [".".join(config.seed)],
            data_name=config.data_name,
            data_group=config.data_group,
            status=config.status,
            plot="responses",
        )

rule plot_middle_model:
    input:
        expand(project_paths.figures / "{experiment}" / "{model_name}{model_args}_{seed}" / "{data_name}:{data_group}_{status}" / "{plot}.png",
            experiment=["contrast", "duration", "interval"],
            model_name=config.model_name,
            model_args=args_product(DEFAULT_MODEL_ARGS | {"activityloss": "*", 'pattern': "1"})
                     + args_product(DEFAULT_MODEL_ARGS | {"activityloss": "*", 'pattern': "1011"}),    
            seed=[".".join(config.seed)],
            data_name=config.data_name,
            data_group=config.data_group,
            status=config.status,
            plot=[f"dynamics_groen_{layer}" for layer in ["V1", "V2"]],
        )

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
        data_category2 = expand(
            project_paths.reports
            / '{{experiment}}'
            / '{{model_name}}{model_args}_{seed}'
            / '{{data_name}}:{{data_group}}_{{status}}'
            / 'test_data.csv',
            model_args=args_product(DEFAULT_MODEL_ARGS | {"activityloss": "*", "rctarget": "middle"}),
            seed=lambda w: w.seeds.split('.'),
        ),
        script = SCRIPTS / 'visualization' / 'plot_performance_manuscript.py'
    params:
        # data_ffonly = lambda w: [project_paths.reports / f'{w.experiment}ffonly' / f'{w.model_name}{w.args1}{w.category}=*{w.args2}_{seed}' / f'{w.data_name}:{w.data_group}_{w.status}' / 'test_data.csv' for seed in w.seeds.split('.')],
        row = None,
        subplot = 'parameter',
        hue = 'category',
        parameter = lambda w: config.experiment_config[w.experiment]['parameter'],
        category = lambda w: w.category,
        category2 = 'activityloss',
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
            --data-category2 {input.data_category2:q} \
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
            project_paths.data.external / "groen-et-al_2022" / f"groen2022_{w.experiment}_data.csv"
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
            model_args=args_product(DEFAULT_MODEL_ARGS | {"activityloss": "*", "pattern": "1011", "rctarget": "output"}),
            seed=lambda w: w.seeds.split('.'),
        ),
        category2 = "activityloss",
        parameter = lambda w: config.experiment_config[w.experiment]['parameter'],
        dt = getattr(config, 'dt', 2),
        palette = lambda w: json.dumps(config.palette),
        naming = lambda w: json.dumps(config.naming),
        ordering = lambda w: json.dumps(config.ordering),
        # subplot_filter = [1, 3, 5, 10, 15, 20]
        groen_data_dir = project_paths.data.external / "groen-et-al_2022",
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

def data_category2(w, experiment, model_args={"activityloss": "*", "pattern": "1011"}):
    return expand(
            project_paths.reports
            / '{experiment}'
            / '{model_name}{model_args}_{seed}'
            / '{data_name}:{data_group}_{status}'
            / 'test_data.csv',
            experiment=experiment,
            model_name=lambda w: w.model_name,
            data_name=lambda w: w.data_name,
            data_group=lambda w: w.data_group,
            status=lambda w: w.status,
            model_args=args_product(DEFAULT_MODEL_ARGS | model_args),
            seed=lambda w: w.seeds.split('.'),
        )

rule plot_all_dynamics_manuscript:
    """Plot comprehensive dynamics manuscript figure with all three experiments.

    Creates a composite figure showing:
    - Panel A: Performance traces (accuracy + confidence) for a middle parameter value
    - Panel B: Layer response ridge plots for focus layer across parameter values
    - Panels C, D, E: Summary statistics for interval (i), duration (ii), contrast (iii)
      experiments arranged in columns

    Input:
        - Three separate data files for interval, duration, contrast experiments
        - Three separate category2 data files for each experiment
        - Three separate Groen data files for empirical comparison

    Output:
        Comprehensive dynamics figure with Groen data comparison
    """
    input:
        # Main data for each experiment
        data_interval = expand(
            project_paths.reports
            / 'interval'
            / '{{model_name}}{{args1}}{{category}}=*{{args2}}_{seed}'
            / '{{data_name}}:{{data_group}}_{{status}}'
            / 'test_data.csv',
            seed=lambda w: w.seeds.split('.'),
        ),
        data_duration = expand(
            project_paths.reports
            / 'duration'
            / '{{model_name}}{{args1}}{{category}}=*{{args2}}_{seed}'
            / '{{data_name}}:{{data_group}}_{{status}}'
            / 'test_data.csv',
            seed=lambda w: w.seeds.split('.'),
        ),
        data_contrast = expand(
            project_paths.reports
            / 'contrast'
            / '{{model_name}}{{args1}}{{category}}=*{{args2}}_{seed}'
            / '{{data_name}}:{{data_group}}_{{status}}'
            / 'test_data.csv',
            seed=lambda w: w.seeds.split('.'),
        ),
        # Category2 data paths (constructed from wildcards)
        data_category2_interval = lambda w: data_category2(w, experiment='interval'),
        data_category2_duration = lambda w: data_category2(w, experiment='duration'),
        data_category2_contrast = lambda w: data_category2(w, experiment='contrast'),
        # Groen empirical data files
        groen_data_interval = project_paths.data.external / 'groen-et-al_2022' / 'groen2022_interval_data.csv',
        groen_data_duration = project_paths.data.external / 'groen-et-al_2022' / 'groen2022_duration_data.csv',
        groen_data_contrast = project_paths.data.external / 'groen-et-al_2022' / 'groen2022_contrast_data.csv',
        script = SCRIPTS / 'visualization' / 'plot_all_dynamics_manuscript.py'
    params:
        # Parameter names for each experiment
        parameter_interval = lambda w: config.experiment_config.get('interval', {}).get('parameter', 'interval'),
        parameter_duration = lambda w: config.experiment_config.get('duration', {}).get('parameter', 'duration'),
        parameter_contrast = lambda w: config.experiment_config.get('contrast', {}).get('parameter', 'contrast'),
        # Category names (same across all experiments)
        category = lambda w: w.category,
        category2 = lambda w: w.category2,
        # Common parameters
        dt = getattr(config, 'dt', 2),
        palette = lambda w: json.dumps(config.palette),
        naming = lambda w: json.dumps(config.naming),
        ordering = lambda w: json.dumps(config.ordering),
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
        ),
    output:
        project_paths.figures / 'dynamics' / '{model_name}{args1}{category}=*{args2}_{seeds}' / '{data_name}:{data_group}_{status}' / 'dynamics_{focus_layers}_v_groen+{category2}.png',
    wildcard_constraints:
        focus_layers = r'\*|(?:V1|V2|V4|IT)\+(?:V1|V2|V4|IT)\+(?:V1|V2|V4|IT)',
    shell:
        """
        {params.execution_cmd} \
            --data-interval {input.data_interval:q} \
            --data-duration {input.data_duration:q} \
            --data-contrast {input.data_contrast:q} \
            --data-category2-interval {input.data_category2_interval:q} \
            --data-category2-duration {input.data_category2_duration:q} \
            --data-category2-contrast {input.data_category2_contrast:q} \
            --groen-data-interval {input.groen_data_interval:q} \
            --groen-data-duration {input.groen_data_duration:q} \
            --groen-data-contrast {input.groen_data_contrast:q} \
            --output {output:q} \
            --focus-layers {wildcards.focus_layers} \
            --parameter-interval {params.parameter_interval} \
            --parameter-duration {params.parameter_duration} \
            --parameter-contrast {params.parameter_contrast} \
            --category {params.category} \
            --category2 {params.category2} \
            --dt {params.dt} \
            --palette {params.palette:q} \
            --naming {params.naming:q} \
            --ordering {params.ordering:q}
        """

# rule plot_unrolling:
#     input:
#         engineering_time_data = project_paths.models \
#             / '{model_name}' \
#             / '{model_name}:{args1}tff=0+trc=6+tsk=0+tfb=34{args2}+unrolled=false_{seed}_{data_name}_{status}_{data_loader}{data_args}_{data_group}_test_responses.pt',
#         biological_time_data = project_paths.models \
#             / '{model_name}' \
#             / '{model_name}:{args1}tff=10+trc=6+tsk=20+tfb=14{args2}+unrolled=true_{seed}_{data_name}_{status}_{data_loader}{data_args}_{data_group}_test_responses.pt',
#         script = SCRIPTS / 'visualization' / 'plot_unrolling.py'
#     params:
#         t_feedforward = 10 // 2,  # tff / dt
#         execution_cmd = lambda w, input: build_execution_command(
#             script_path=input.script,
#             use_distributed=False,
#         ),
#     output:
#         project_paths.figures / 'unrolling' / 'unrolling_{model_name}:{args1}tff=*{args2}_{seed}_{data_name}_{status}_{data_loader}{data_args}_{data_group}.png',
#     shell:
#         """
#         {params.execution_cmd} \
#             --engineering_time_data {input.engineering_time_data:q} \
#             --biological_time_data {input.biological_time_data:q} \
#             --t_feedforward {params.t_feedforward} \
#             --output {output:q}
#         """

rule plot_unrolling:
    input:
        engineering_time_data = project_paths.reports \
            / 'unrolling' \
            / '{model_name}{args1}tff={e_tff}+trc={trc}+tsk={e_tsk}{args2}_{seed}' \
            / '{data_name}:{data_group}_{status}' \
            / 'StimulusDuration:dsteps=40+stim=20+idle=20' / 'test_responses.pt',
        biological_time_data = project_paths.reports \
            / 'unrolling' \
            / '{model_name}{args1}tff={e_tff}+trc={trc}+tsk={e_tsk}{args2}_{seed}' \
            / '{data_name}:{data_group}_{status}' \
            / 'StimulusDuration:dsteps=40+stim=20+idle=20+tff={b_tff}+tsk={b_tsk}+tfb={b_tfb}' / 'test_responses.pt',
        script = SCRIPTS / 'visualization' / 'plot_unrolling.py'
    params:
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
        ),
        # Convert feedforward delay from ms to timesteps (tff / dt)
        t_feedforward = lambda w: int(int(w.b_tff) // getattr(config, 'dt', 2)),
        dt = getattr(config, 'dt', 2),
    output:
        project_paths.figures / 'unrolling' / '{model_name}{args1}tff={e_tff}+trc={trc}+tsk={e_tsk}{args2}_{seed}' / '{data_name}:{data_group}_{status}' / 'responses_tff={b_tff}+tsk={b_tsk}+tfb={b_tfb}.png',
    shell:
        """
        {params.execution_cmd} \
            --engineering_time_data {input.engineering_time_data:q} \
            --biological_time_data {input.biological_time_data:q} \
            --t_feedforward {params.t_feedforward} \
            --dt {params.dt} \
            --output {output:q}
        """


rule unrolling:
    input:
        expand(project_paths.models / "DyRCNNx8" / "DyRCNNx8:tsteps=30+rctype=full+rctarget=output+dt=2+tau=5+tff=10+trc=6+tsk=20+tfb=10+skip=true+feedback=true+lossrt=4_{seed}" / "imagenette" / "{status}.pt",
        seed=SEED,
        status=config.status)

rule dataloader:
    input:
        expand(project_paths.figures / 'response' / 'DyRCNNx8:pattern=1+{configuration}_{seed}' / 'imagenette:all_{status}' / 'responses.png',
        seed=SEED,
        status=config.status,
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
