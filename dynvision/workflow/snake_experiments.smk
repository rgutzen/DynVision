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
MODEL_NAME = config.model_name[0] if isinstance(config.model_name, list) else config.model_name
MODEL_FMT = "{model_name}:{args}_{seed}_{data_name}_{status}.pt"
DATA_NAME = config.data_name
DATA_GROUP = config.data_group
STATUS = 'trained-best'
SEED = config.seed if isinstance(config.seed, list) else [config.seed]

def model_path(override=None, model_name=MODEL_NAME, seeds=SEED, data_name=DATA_NAME, status=STATUS):
    """Generate list of model file paths based on parameter combinations."""
    arg_dict = DEFAULT_MODEL_ARGS.copy()
    if override is not None:
        arg_dict.update(override)
    return [(project_paths.models / model_name / f"{model_name}{args}_{seed}_{data_name}_{status}.pt") for seed in seeds for args in args_product(arg_dict)]

def result_path(experiment, category, model_name=MODEL_NAME, seeds=SEED, data_name=DATA_NAME, data_group=DATA_GROUP, status=STATUS, plot=None):
    """Generate list of result file paths based on parameter combinations."""
    experiments = [experiment] if isinstance(experiment, str) else list(experiment)
    arg_dict = DEFAULT_MODEL_ARGS.copy()
    arg_dict.update({category: "*"})
    folder = project_paths.reports if plot is None else project_paths.reports
    file = "test_data.csv" if plot is None else f"{plot}.png"
    paths = []
    seeds = seeds if isinstance(seeds, list) else [seeds]
    for seed in seeds:
        for exp in experiments:
            for args in args_product(arg_dict):
                paths.append(
                    folder
                    / exp
                    / f"{exp}_{model_name}{args}_{seed}_{data_name}_{status}_{data_group}/{file}"
                )
    return paths

rule train_model_variation:  # call with --config var=<variable_name>
    input:
        model_path(
            override={config.var: config.experiment_config['categories'][config.var]} if hasattr(config, 'var') else None,
            status='trained',
        ) if hasattr(config, 'var') else [],

rule test_model_variation:  # call with --config var=<variable_name>
    input:
        result_path(
            experiment=config.experiment,
            category=config.var if hasattr(config, 'var') else None,
            status=STATUS,
        ) if hasattr(config, 'var') else [],

rule plot_model_variation:  # call with --config var=<variable_name>
    input:
        result_path(
            experiment=config.experiment,
            category=get_param('var', None),
            status=STATUS,
            plot=get_param('plot', 'responses'),
        ) if hasattr(config, 'var') else [],

rule run_noise_recreation_attempt:
    input:
        project_paths.figures / "uniformnoise" / "uniformnoise_DyRCNNx8:tsteps=20+dt=2+tau=5+tff=0+trc=6+tsk=0+lossrt=4+energyloss=0.1+pattern=1+rctype=full+rctarget=output+skip=true+feedback=false+dloader=*_6000_imagenette_trained-epoch=149_all/performance.png",

# rule run_pattern1_ffcv:
#     input:
#         model_path(
#             override={config.var: config.experiment_config['categories'][config.var], "pattern": "1", "dloader": "ffcv"},
#             status='trained',
#         )

# rule run_pattern1_torch: # call with --config use_ffcv=False
#     input:
#         model_path(
#             override={config.var: config.experiment_config['categories'][config.var], "pattern": "1", "dloader": "torch"},
#             status='trained')

# rule run_pattern1011_torch: # call with --config use_ffcv=False
#     input:
#         model_path(
#             override={config.var: config.experiment_config['categories'][config.var], "pattern": "1011", "dloader": "torch"},
#             status='trained',
#         )

rule recreate_noise_results:  # run with --allowed-rules test_model process_test_data
    input:
        project_paths.reports / "uniformnoise" / "uniformnoise_DyRCNNx8:tsteps=20+rctype=full+rctarget=*+dt=2+tau=5+tff=0+trc=6+tsk=0+lossrt=4_0040_imagenette_trained_all/test_data.csv",

PARAM_VARIATIONS = [
    dict(lossrt=config.experiment_config['categories']['lossrt']),
    dict(tsteps=config.experiment_config['categories']['tsteps']),
    dict(idle=config.experiment_config['categories']['idle']),
    dict(tau=config.experiment_config['categories']['tau']),
    dict(trc=config.experiment_config['categories']['trc']),
    dict(tsk=config.experiment_config['categories']['tsk']),
    dict(skip=config.experiment_config['categories']['skip']),
    dict(feedback=config.experiment_config['categories']['feedback']),
    dict(rctarget=config.experiment_config['categories']['rctarget']),
    dict(rctype=config.experiment_config['categories']['rctype']),
]
TRAIN_VARIATIONS = [
    dict(energyloss=config.experiment_config['categories']['energyloss'], 
        pattern=config.experiment_config['categories']['pattern'], 
    ),
]
LOADER_VARIATIONS = [
    dict(tsteps=20, dloader="{dloader}", dsteps=1, pattern="1"),  # unroll in model steps
    dict(tsteps=1, dloader="{dloader}", dsteps=20, pattern="1"),  # unroll in data loader
    # dict(tff=0, trc=6, tsk=0, tfb=30),   # engineering time unrolling
    # dict(tff=10, trc=6, tsk=20, tfb=10), # biological time unrolling
]


rule train_with_parameter_variations: # --config epochs=300
    input:
        [model_path(override=var, status='trained') for var in PARAM_VARIATIONS]

rule train_with_loss_variations: # --config epochs=500
    input:
        [model_path(override=var, status='trained') for var in TRAIN_VARIATIONS]

rule train_with_dataloader_variations: # --config epochs=100
    # run both with --config use_ffcv=True & --config use_ffcv=False
    # and rename dloader=* in output filename to ffcv or torch respectively
    input:
        # dataloader variations
        [model_path(override=var, status='trained', seeds=SEED[0] if isinstance(SEED, list) else SEED) for var in LOADER_VARIATIONS],
    params:
        dloader = "ffcv" if config.use_ffcv else "torch"
    shell:
        """
        for file in {input}; do
            target="${file//\\{dloader\\}/{params.dloader}}"
            if [ "$file" != "$target" ]; then
                mv "$file" "$target"
            fi
        done
        """

rule manuscript_figures: # manuscript figures  
# sh snakecharm.sh "manuscript_figures --allowed-rules plot_dynamics plot_responses plot_timeparams_tripytch plot_connection_tripytch plot_timestep_tripytch plot_training plot_performance"
    input:
        # neural dynamics comparison figures
        expand(project_paths.figures / "{experiment}" / "{experiment}_DyRCNNx8:tsteps=20+{categories}+dt=2+tau=5+tff=0+trc=6+tsk=0+skip=true+lossrt=4_{seeds}_imagenette_{status}_all/dynamics_{focus_layer}.png",
            experiment=['duration', 'contrast', 'interval'],
            focus_layer=['V1', 'V2', 'V4', 'IT'],
            categories=['rctype=*+rctarget=output'], # , 'rctype=full+rctarget=*'],
            seeds='.'.join(SEED),
            status=STATUS,
        ),
        # timestep tripytch
        expand(project_paths.figures / '{experiment}' / '{experiment}_DyRCNNx8:tsteps=*+rctype=full+rctarget=output+dt=2+tau=5+tff=0+trc=6+tsk=0+skip=true+lossrt=*+idle=*_{seeds}_imagenette_{status}_all' / 'response_tripytch.png',
            experiment=['response'],
            seeds='.'.join(SEED),
            status=STATUS,
        ),
        # timeparams tripytch
        expand(project_paths.figures / '{experiment}' / '{experiment}_DyRCNNx8:tsteps=20+rctype=full+rctarget=output+dt=2+tau=*+tff=0+trc=*+tsk=*+skip=true+lossrt=4_{seeds}_imagenette_{status}_all' / 'response_tripytch.png',
            experiment=['response'],
            seeds='.'.join(SEED),
            status=STATUS,
        ),
        # connection tripytch
        expand(project_paths.figures / '{experiment}' / '{experiment}_DyRCNNx8:rctype=full+dt=2+tau=5+tff=0+trc=5+tsk=0+rctarget=*+skip=*+feedback=*+lossrt=4_{seeds}_imagenette_{status}_all' / 'response_tripytch.png',
            experiment=['response'],
            seeds='.'.join(SEED),
            status=STATUS,
        ),
        # noise performance
        expand(project_paths.figures / "{experiment}" / "{experiment}_DyRCNNx8:tsteps=20+rctype=full+rctarget=*+dt=2+tau=5+tff=0+trc=6+tsk=0+skip=true+lossrt=4_{seeds}_imagenette_{status}_all/performance.png",
            experiment='noise',
            seeds='.'.join(SEED),
            status=STATUS,
        ),
        # stability
        expand(project_paths.figures / "{experiment}" / "{experiment}_DyRCNNx8:tsteps=20+{categories}+dt=2+tau=5+tff=0+trc=6+tsk=0+skip=true+lossrt=4_{seeds}_imagenette_{status}_all/responses.png",
            experiment='stability',
            categories=['rctype=*+rctarget=output', 'rctype=full+rctarget=*'],
            seeds='.'.join(SEED),
            status=STATUS,
        ),
        # training progress
        expand(project_paths.figures / 'response' / 'response_DyRCNNx8:tsteps=20+{categories}+dt=2+tau=5+tff=0+trc=6+tsk=0+skip=true+lossrt=4_{seeds}_imagenette_{status}_all' / 'training.png',
            categories=['rctype=*+rctarget=output'], #, 'rctype=full+rctarget=*'],
            seeds='.'.join(SEED),
            status=STATUS,
        ),
        # reference models
        expand(project_paths.figures / "{experiment}" / "{experiment}_{model}:pretrained=*_{seeds}_imagenet_init_imagenette/responses.png",
            experiment=['response', 'idleresponse', 'hundred'],
            model=['CorNetRT', 'CordsNet'],
            seeds=SEED[0],
        ),
        # energy loss scan
        expand(project_paths.figures / 'stability' / 'stability_DyRCNNx8:tsteps=20+rctype=full+rctarget=output+dt=2+tau=5+tff=0+trc=6+tsk=0+skip=true+lossrt=4+energyloss=*_{seeds}_imagenette_{status}_all' / 'training.png',
            seeds='.'.join(SEED),
            status=STATUS,
        ),
        # dataloader comparison
        expand(project_paths.figures / 'response' / 'response_DyRCNNx8:{configuration}+pattern=1+shufflepattern=false_{seed}_imagenette_{status}_all' / 'responses.png',
        seed=SEED,
        status=STATUS,
        configuration=[
            'tsteps=20+dloader=*+dsteps=1',
            'tsteps=1+dloader=*+dsteps=20',
        ]),
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
    

# rule train_with_parameter_variations_: # --config epochs=300
#     # default: DyRCNNx8:tsteps=20+rctype=full+rctarget=output+dt=2+tau=5+tff=0+trc=6+tsk=0+skip=true+lossrt=4
#     input:
#         # lossrt
#         expand(project_paths.models / "DyRCNNx8" / "DyRCNNx8:tsteps=20+rctype=full+rctarget=output+dt=2+tau=5+tff=0+trc=6+tsk=0+skip=true+lossrt={lossrt}_{seed}_imagenette_trained.pt",
#             lossrt=config.experiment_config['categories'].get("lossrt"),
#             seed=SEED,
#         ),
#         # tsteps
#         expand(project_paths.models / "DyRCNNx8" / "DyRCNNx8:tsteps={tsteps}+rctype=full+rctarget=output+dt=2+tau=5+tff=0+trc=6+tsk=0+skip=true+lossrt=4_{seed}_imagenette_trained.pt",
#             tsteps=config.experiment_config['categories'].get("tsteps"),
#             seed=SEED,
#         ),
#         # idle
#         expand(project_paths.models / "DyRCNNx8" / "DyRCNNx8:tsteps=20+rctype=full+rctarget=output+dt=2+tau=5+tff=0+trc=6+tsk=0+skip=true+lossrt=4+idle={idle}_{seed}_imagenette_trained.pt",
#             idle=config.experiment_config['categories'].get("idle"),
#             seed=SEED,
#         ),
#         # tau
#         expand(project_paths.models / "DyRCNNx8" / "DyRCNNx8:tsteps=20+rctype=full+rctarget=output+rctarget=output+dt=2+tau={tau}+tff=0+trc=6+tsk=0+skip=true+lossrt=4_{seed}_imagenette_trained.pt",
#             tau=config.experiment_config['categories'].get("tau"),
#             seed=SEED,
#         ),
#         # trc
#         expand(project_paths.models / "DyRCNNx8" / "DyRCNNx8:tsteps=20+rctype=full+rctarget=output+dt=2+tau=5+tff=0+trc={trc}+tsk=0+skip=true+lossrt=4_{seed}_imagenette_trained.pt",
#             trc=config.experiment_config['categories'].get("trc"),
#             seed=SEED,
#         ),
#         # tsk
#         expand(project_paths.models / "DyRCNNx8" / "DyRCNNx8:tsteps=20+rctype=full+rctarget=output+dt=2+tau=5+tff=0+trc=6+tsk={tsk}+skip=true+lossrt=4_{seed}_imagenette_trained.pt",
#             tsk=config.experiment_config['categories'].get("tsk"),
#             seed=SEED,
#         ),
#         # skip
#         expand(project_paths.models / "DyRCNNx8" / "DyRCNNx8:tsteps=20+rctype=full+rctarget=output+dt=2+tau=5+tff=0+trc=6+tsk=0+skip={skip}+lossrt=4_{seed}_imagenette_trained.pt",
#             skip=config.experiment_config['categories'].get("skip"),
#             seed=SEED,
#         ),
rule train_feedback:
    input:
        # feedback
        expand(project_paths.models / "DyRCNNx8" / "DyRCNNx8:tsteps=20+rctype=full+rctarget=output+dt=2+tau=5+tff=0+trc=6+tsk=0+tfb=30+skip=true+feedback={feedback}+lossrt=4_{seed}_imagenette_trained.pt",
            feedback=config.experiment_config['categories'].get("feedback"),
            seed=SEED,
        )

#         # rctarget
#         expand(project_paths.models / "DyRCNNx8" / "DyRCNNx8:tsteps=20+rctype=full+rctarget={rctarget}+dt=2+tau=5+tff=0+trc=6+tsk=0+skip=true+lossrt=4_{seed}_imagenette_trained.pt",
#             rctarget=config.experiment_config['categories'].get("rctarget"),
#             seed=SEED,
#         ),
rule train_rctype: # --config epochs=300
    input:
        # rctype
        expand(project_paths.models / "DyRCNNx8" / "DyRCNNx8:tsteps=20+rctype={rctype}+rctarget=output+dt=2+tau=5+tff=0+trc=6+tsk=0+skip=true+lossrt=4_{seed}_imagenette_trained.pt",
            rctype=config.experiment_config['categories'].get("rctype"),
            seed=SEED,
        )
#         # engineering vs biological time unrolling
#         expand(project_paths.models / "DyRCNNx8" / "DyRCNNx8:tsteps=20+rctype=full+rctarget=output+dt=2+tau=5+{unrolling}+skip=true+feedback=true+lossrt=4_{seed}_imagenette_trained.pt",
#             unrolling=['tff=0+trc=6+tsk=0+tfb=30', 'tff=10+trc=6+tsk=20+tfb=10'],
#             seed=SEED[0],
#         ),


rule idle:
    input:
        expand(project_paths.reports / "{experiment}" / "{experiment}_DyRCNNx8:tsteps=20+rctype=full+rctarget=output+dt=2+tau=5+tff=0+trc=6+tsk=0+skip=true+lossrt=4+idle=*_{seed}_imagenette_{status}_all" / "test_data.csv",
            seed=SEED,
            experiment=['response'],
            status=STATUS)

rule feedback:  # run with --config batch_size=128
    input:
        expand(project_paths.reports / "{experiment}" / "{experiment}_DyRCNNx8:tsteps=20+rctype=full+rctarget=output+dt=2+tau=5+tff=0+trc=6+tsk=0+tfb=30+skip=true+feedback=*+lossrt=4_{seed}_imagenette_{status}_all" / "test_data.csv",
            seed=SEED,
            experiment=['response', 'responseffonly'],
            status=STATUS)

rule skip:
    input:
        expand(project_paths.reports / "{experiment}" / "{experiment}_DyRCNNx8:tsteps=20+rctype=full+rctarget=output+dt=2+tau=5+tff=0+trc=6+tsk=0+skip=*+lossrt=4_{seed}_imagenette_{status}_all" / "test_data.csv",
            seed=SEED,
            experiment=['response', 'responseffonly'],
            status=STATUS)

rule tsteps:
    input:
        expand(project_paths.reports / "{experiment}" / "{experiment}_DyRCNNx8:tsteps=*+rctype=full+rctarget=output+dt=2+tau=5+tff=0+trc=6+tsk=0+skip=true+lossrt=4_{seed}_imagenette_{status}_all" / "test_data.csv",
            seed=SEED,
            experiment=['response'],
            status=STATUS)

rule lossrt:
    input:
        expand(project_paths.reports / "{experiment}" / "{experiment}_DyRCNNx8:tsteps=20+rctype=full+rctarget=output+dt=2+tau=5+tff=0+trc=6+tsk=0+skip=true+lossrt=*_{seed}_imagenette_{status}_all" / "test_data.csv",
        seed=SEED,
        experiment=['response'],
        status=STATUS)

rule rctarget:
    input:
        expand(project_paths.reports / "{experiment}" / "{experiment}_DyRCNNx8:tsteps=20+rctype=full+rctarget=*+dt=2+tau=5+tff=0+trc=6+tsk=0+skip=true+lossrt=4_{seed}_imagenette_{status}_all" / "test_data.csv",
        seed=SEED,  # non-oscillatory seeds only
        # experiment=['responseffonly', 'response', 'duration', 'contrast', 'interval', 
        experiment=['uniformnoise', 'uniformnoiseffonly', 'poissonnoise', 'poissonnoiseffonly'],
        # experiment=['gaussiannoise', 'gaussiannoiseffonly', 'gaussiancorrnoise', 'gaussiancorrnoiseffonly'],
        # 'poissonnoise', 'poissonnoiseffonly', 'phasescramblednoise', 'phasescramblednoiseffonly'],
        status=STATUS),

rule rctype:
    input:
        expand(project_paths.reports / "{experiment}" / "{experiment}_DyRCNNx8:tsteps=20+rctype=*+rctarget=output+dt=2+tau=5+tff=0+trc=6+tsk=0+skip=true+lossrt=4_{seed}_imagenette_{status}_all" / "test_data.csv",
        seed=SEED,
        experiment=['response', 'duration', 'contrast', 'interval', 'responseffonly'],
        status=STATUS),

rule timeparams:  # add rctarget=output
    input:
        expand(project_paths.reports / "{experiment}" / "{experiment}_DyRCNNx8:tsteps=20+rctype=full+dt=2+tau=*+tff=0+trc=6+tsk=0+skip=true+lossrt=4_{seed}_imagenette_{status}_all" / "test_data.csv",
        seed=SEED,
        experiment=['response'],
        status=STATUS),
        expand(project_paths.reports / "{experiment}" / "{experiment}_DyRCNNx8:tsteps=20+rctype=full+dt=2+tau=5+tff=0+trc=*+tsk=0+skip=true+lossrt=4_{seed}_imagenette_{status}_all" / "test_data.csv",
        seed=SEED,
        experiment=['response'],
        status=STATUS),
        expand(project_paths.reports / "{experiment}" / "{experiment}_DyRCNNx8:tsteps=20+rctype=full+dt=2+tau=5+tff=0+trc=6+tsk=*+skip=true+lossrt=4_{seed}_imagenette_{status}_all" / "test_data.csv",
        seed=SEED,
        experiment=['response'],
        status=STATUS),

rule stability:  # run with --config test_batch_size=16
    input:
        expand(project_paths.reports / "{experiment}" / "{experiment}_DyRCNNx8:tsteps=20+rctype=full+rctarget=*+dt=2+tau=5+tff=0+trc=6+tsk=0+skip=true+lossrt=4_{seed}_imagenette_{status}_all" / "test_data.csv",
        seed=SEED,
        experiment=['stability'],
        status=STATUS),
        expand(project_paths.reports / "{experiment}" / "{experiment}_DyRCNNx8:tsteps=20+rctype=*+rctarget=output+dt=2+tau=5+tff=0+trc=6+tsk=0+skip=true+lossrt=4_{seed}_imagenette_{status}_all" / "test_data.csv",
        seed=SEED,
        experiment=['stability'],
        status=STATUS)

rule unrolling:
    input:
        expand(project_paths.models / "DyRCNNx8" / "DyRCNNx8:tsteps=30+rctype=full+rctarget=output+dt=2+tau=5+tff=10+trc=6+tsk=20+tfb=10+skip=true+feedback=true+lossrt=4_{seed}_imagenette_{status}.pt",
        seed=SEED,
        status=STATUS)

rule energyloss:
    input:
        expand(project_paths.reports / '{experiment}' / '{experiment}_DyRCNNx8:tsteps=20+dt=2+lossrt=4+pattern={pattern}+energyloss=*_{seeds}_imagenette_{status}_all' / 'test_data.csv',
            seeds='.'.join(SEED),
            status="trained-epoch=149",
            pattern=['1', '1011'],
            experiment=['uniformnoise'] #, 'uniformnoiseffonly', 'duration', 'interval'],
        ),

rule training:
    input:
        expand(project_paths.figures / '{experiment}' / '{experiment}_DyRCNNx8:tsteps=20+dt=2+lossrt=4+pattern={pattern}+energyloss=*_{seeds}_imagenette_{status}_all/training.png',
        pattern=['1'], # 011'], #, '1'],
        seeds='.'.join(SEED),
        status=STATUS,
        experiment=['stability']
        ),
        # 'responseintermediate_DyRCNNx8:tsteps=20+rctype=full+rctarget=output+dt=2+tau=5+tff=0+trc=6+tsk=0+skip=true+lossrt=4+energyloss=*_{seeds}_imagenette_trained_all' / 'responses.png',
#                     seeds=SEED[0], #'.'.join(SEED),
#                 ),

rule dataloader:
    input:
        expand(project_paths.figures / 'response' / 'response_DyRCNNx8:pattern=1+{configuration}_{seed}_imagenette_{status}_all' / 'responses.png',
        seed=SEED,
        status=STATUS,
        configuration=[
            'tsteps=20+dataloader=*+dsteps=1',
            'tsteps=1+dataloader=*+dsteps=20',
        ]),

rule references:
    input:
        expand(project_paths.figures / "{experiment}" / "{experiment}_{model}:pretrained=*_{seeds}_imagenet_init_imagenette/responses.png",
            experiment=['response', 'hundred'],
            model=['CorNetRT', 'CordsNet'],
            seeds=SEED[0], #'.'.join(SEED),
        ),
        

rule imagenet:  # run with --config use_distributed_mode=True
    input:
        expand(project_paths.models / "DyRCNNx8" / "DyRCNNx8:tsteps=10+rctype=full+dt=2+tau=5+tff=0+trc=6+tsk=0+lossrt=4_{seed}_imagenet_trained.pt",
        seed=SEED,
        )
