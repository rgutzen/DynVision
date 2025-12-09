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


rule train_model_variations:  # call with --config category=<category_name>
    input:
        [expand(project_paths.models / "{model_name}" / "{model_name}{model_args}_{seed}" / "{data_name}" / "{status}.pt",
            model_name=config.model_name,
            model_args=args_product(DEFAULT_MODEL_ARGS | {category: config.experiment_config['categories'][category]}),
            seed=config.seed,
            data_name=config.data_name,
            status=config.status,
        ) for category in config.category]
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
    input:
        [expand(project_paths.reports / "{experiment}" / "{model_name}{model_args}_{seed}" / "{data_name}:{data_group}_{status}" / "test_data.csv",
            experiment=config.experiment,
            model_name=config.model_name,
            model_args=args_product(DEFAULT_MODEL_ARGS | {category: "*"}),
            seed=config.seed,
            data_name=config.data_name,
            data_group=config.data_group,
            status=config.status,
        ) for category in config.category]

rule plot_model_variations:  # call with --config category=<category_name> experiment=<experiment_name> plot=<plot_name>
    input:
        [expand(project_paths.figures / "{experiment}" / "{model_name}{model_args}_{seed}" / "{data_name}:{data_group}_{status}" / "{plot}.png",
            experiment=config.experiment,
            model_name=config.model_name,
            model_args=args_product(DEFAULT_MODEL_ARGS | {category: "*"}),
            seed=config.seed,
            data_name=config.data_name,
            data_group=config.data_group,
            status=config.status,
            plot=getattr(config, 'plot', 'responses'),
        ) for category in config.category]

rule plot_model_variation_weights:  # call with --config category=<variable_name>
    input:
        [expand(project_paths.figures / "weights" / "{model_name}{model_args}_{seed}" / "{data_name}_{status}" / "weights.png",
            model_name=config.model_name,
            model_args=args_product(DEFAULT_MODEL_ARGS | {category: "*"}),
            seed=config.seed,
            data_name=config.data_name,
            status=config.status,
        ) for category in config.category]


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
        # neural dynamics comparison figures
        expand(project_paths.figures / "{experiment}" / "DyRCNNx8:tsteps=20+{categories}+dt=2+tau=5+tff=0+trc=6+tsk=0+skip=true+lossrt=4_{seeds}" / "imagenette:all_{status}" / "dynamics_{focus_layer}.png",
            experiment=['duration', 'contrast', 'interval'],
            focus_layer=['V1', 'V2', 'V4', 'IT'],
            categories=['rctype=*+rctarget=output'], # , 'rctype=full+rctarget=*'],
            seeds='.'.join(SEED),
            status=STATUS,
        ),
        # timestep tripytch
        expand(project_paths.figures / '{experiment}' / 'DyRCNNx8:tsteps=*+rctype=full+rctarget=output+dt=2+tau=5+tff=0+trc=6+tsk=0+skip=true+lossrt=*+idle=*_{seeds}' / 'imagenette:all_{status}' / 'response_tripytch.png',
            experiment=['response'],
            seeds='.'.join(SEED),
            status=STATUS,
        ),
        # timeparams tripytch
        expand(project_paths.figures / '{experiment}' / 'DyRCNNx8:tsteps=20+rctype=full+rctarget=output+dt=2+tau=*+tff=0+trc=*+tsk=*+skip=true+lossrt=4_{seeds}' / 'imagenette:all_{status}' / 'response_tripytch.png',
            experiment=['response'],
            seeds='.'.join(SEED),
            status=STATUS,
        ),
        # connection tripytch
        expand(project_paths.figures / '{experiment}' / 'DyRCNNx8:rctype=full+dt=2+tau=5+tff=0+trc=5+tsk=0+rctarget=*+skip=*+feedback=*+lossrt=4_{seeds}' / 'imagenette:all_{status}' / 'response_tripytch.png',
            experiment=['response'],
            seeds='.'.join(SEED),
            status=STATUS,
        ),
        # noise performance
        expand(project_paths.figures / "{experiment}" / "DyRCNNx8:tsteps=20+rctype=full+rctarget=*+dt=2+tau=5+tff=0+trc=6+tsk=0+skip=true+lossrt=4_{seeds}" / "imagenette:all_{status}" / "performance.png",
            experiment='noise',
            seeds='.'.join(SEED),
            status=STATUS,
        ),
        # stability
        expand(project_paths.figures / "{experiment}" / "DyRCNNx8:tsteps=20+{categories}+dt=2+tau=5+tff=0+trc=6+tsk=0+skip=true+lossrt=4_{seeds}" / "imagenette:all_{status}" / "responses.png",
            experiment='stability',
            categories=['rctype=*+rctarget=output', 'rctype=full+rctarget=*'],
            seeds='.'.join(SEED),
            status=STATUS,
        ),
        # training progress
        expand(project_paths.figures / 'response' / 'DyRCNNx8:tsteps=20+{categories}+dt=2+tau=5+tff=0+trc=6+tsk=0+skip=true+lossrt=4_{seeds}' / 'imagenette:all_{status}' / 'training.png',
            categories=['rctype=*+rctarget=output'], #, 'rctype=full+rctarget=*'],
            seeds='.'.join(SEED),
            status=STATUS,
        ),
        # reference models
        expand(project_paths.figures / "{experiment}" / "{model}:pretrained=*_{seeds}" / "imagenet:imagenette_init" / "responses.png",
            experiment=['response', 'idleresponse', 'hundred'],
            model=['CorNetRT', 'CordsNet'],
            seeds=SEED[0],
        ),
        # energy loss scan
        expand(project_paths.figures / 'stability' / 'DyRCNNx8:tsteps=20+rctype=full+rctarget=output+dt=2+tau=5+tff=0+trc=6+tsk=0+skip=true+lossrt=4+energyloss=*_{seeds}' / 'imagenette:all_{status}' / 'training.png',
            seeds='.'.join(SEED),
            status=STATUS,
        ),
        # dataloader comparison
        expand(project_paths.figures / 'response' / 'DyRCNNx8:{configuration}+pattern=1+shufflepattern=false_{seed}' / 'imagenette:all_{status}' / 'responses.png',
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
        expand(project_paths.models / "DyRCNNx8" / "DyRCNNx8:tsteps=20+rctype=full+rctarget=output+dt=2+tau=5+tff=0+trc=6+tsk=0+tfb=30+skip=true+feedback={feedback}+lossrt=4_{seed}" / "imagenette" / "trained.pt",
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
        expand(project_paths.models / "DyRCNNx8" / "DyRCNNx8:tsteps=20+rctype={rctype}+rctarget=output+dt=2+tau=5+tff=0+trc=6+tsk=0+skip=true+lossrt=4_{seed}" / "imagenette" / "trained.pt",
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
        expand(project_paths.reports / "{experiment}" / "DyRCNNx8:tsteps=20+rctype=full+rctarget=output+dt=2+tau=5+tff=0+trc=6+tsk=0+skip=true+lossrt=4+idle=*_{seed}" / "imagenette:all_{status}" / "test_data.csv",
            seed=SEED,
            experiment=['response'],
            status=STATUS)

rule feedback:  # run with --config batch_size=128
    input:
        expand(project_paths.reports / "{experiment}" / "DyRCNNx8:tsteps=20+rctype=full+rctarget=output+dt=2+tau=5+tff=0+trc=6+tsk=0+tfb=30+skip=true+feedback=*+lossrt=4_{seed}" / "imagenette:all_{status}" / "test_data.csv",
            seed=SEED,
            experiment=['response', 'responseffonly'],
            status=STATUS)

rule skip:
    input:
        expand(project_paths.reports / "{experiment}" / "DyRCNNx8:tsteps=20+rctype=full+rctarget=output+dt=2+tau=5+tff=0+trc=6+tsk=0+skip=*+lossrt=4_{seed}" / "imagenette:all_{status}" / "test_data.csv",
            seed=SEED,
            experiment=['response', 'responseffonly'],
            status=STATUS)

rule tsteps:
    input:
        expand(project_paths.reports / "{experiment}" / "DyRCNNx8:tsteps=*+rctype=full+rctarget=output+dt=2+tau=5+tff=0+trc=6+tsk=0+skip=true+lossrt=4_{seed}" / "imagenette:all_{status}" / "test_data.csv",
            seed=SEED,
            experiment=['response'],
            status=STATUS)

rule lossrt:
    input:
        expand(project_paths.reports / "{experiment}" / "DyRCNNx8:tsteps=20+rctype=full+rctarget=output+dt=2+tau=5+tff=0+trc=6+tsk=0+skip=true+lossrt=*_{seed}" / "imagenette:all_{status}" / "test_data.csv",
        seed=SEED,
        experiment=['response'],
        status=STATUS)

rule rctarget:
    input:
        expand(project_paths.reports / "{experiment}" / "DyRCNNx8:tsteps=20+rctype=full+rctarget=*+dt=2+tau=5+tff=0+trc=6+tsk=0+skip=true+lossrt=4_{seed}" / "imagenette:all_{status}" / "test_data.csv",
        seed=SEED,  # non-oscillatory seeds only
        # experiment=['responseffonly', 'response', 'duration', 'contrast', 'interval',
        experiment=['uniformnoise', 'uniformnoiseffonly', 'poissonnoise', 'poissonnoiseffonly'],
        # experiment=['gaussiannoise', 'gaussiannoiseffonly', 'gaussiancorrnoise', 'gaussiancorrnoiseffonly'],
        # 'poissonnoise', 'poissonnoiseffonly', 'phasescramblednoise', 'phasescramblednoiseffonly'],
        status=STATUS),

rule rctype:
    input:
        expand(project_paths.reports / "{experiment}" / "DyRCNNx8:tsteps=20+rctype=*+rctarget=output+dt=2+tau=5+tff=0+trc=6+tsk=0+skip=true+lossrt=4_{seed}" / "imagenette:all_{status}" / "test_data.csv",
        seed=SEED,
        experiment=['response', 'duration', 'contrast', 'interval', 'responseffonly'],
        status=STATUS),

rule timeparams:  # add rctarget=output
    input:
        expand(project_paths.reports / "{experiment}" / "DyRCNNx8:tsteps=20+rctype=full+dt=2+tau=*+tff=0+trc=6+tsk=0+skip=true+lossrt=4_{seed}" / "imagenette:all_{status}" / "test_data.csv",
        seed=SEED,
        experiment=['response'],
        status=STATUS),
        expand(project_paths.reports / "{experiment}" / "DyRCNNx8:tsteps=20+rctype=full+dt=2+tau=5+tff=0+trc=*+tsk=0+skip=true+lossrt=4_{seed}" / "imagenette:all_{status}" / "test_data.csv",
        seed=SEED,
        experiment=['response'],
        status=STATUS),
        expand(project_paths.reports / "{experiment}" / "DyRCNNx8:tsteps=20+rctype=full+dt=2+tau=5+tff=0+trc=6+tsk=*+skip=true+lossrt=4_{seed}" / "imagenette:all_{status}" / "test_data.csv",
        seed=SEED,
        experiment=['response'],
        status=STATUS),

rule stability:  # run with --config test_batch_size=16
    input:
        expand(project_paths.reports / "{experiment}" / "DyRCNNx8:tsteps=20+rctype=full+rctarget=*+dt=2+tau=5+tff=0+trc=6+tsk=0+skip=true+lossrt=4_{seed}" / "imagenette:all_{status}" / "test_data.csv",
        seed=SEED,
        experiment=['stability'],
        status=STATUS),
        expand(project_paths.reports / "{experiment}" / "DyRCNNx8:tsteps=20+rctype=*+rctarget=output+dt=2+tau=5+tff=0+trc=6+tsk=0+skip=true+lossrt=4_{seed}" / "imagenette:all_{status}" / "test_data.csv",
        seed=SEED,
        experiment=['stability'],
        status=STATUS)

rule unrolling:
    input:
        expand(project_paths.models / "DyRCNNx8" / "DyRCNNx8:tsteps=30+rctype=full+rctarget=output+dt=2+tau=5+tff=10+trc=6+tsk=20+tfb=10+skip=true+feedback=true+lossrt=4_{seed}" / "imagenette" / "{status}.pt",
        seed=SEED,
        status=STATUS)

rule energyloss:
    input:
        expand(project_paths.reports / '{experiment}' / 'DyRCNNx8:tsteps=20+dt=2+lossrt=4+pattern={pattern}+energyloss=*_{seeds}' / 'imagenette:all_{status}' / 'test_data.csv',
            seeds='.'.join(SEED),
            status="trained-epoch=149",
            pattern=['1', '1011'],
            experiment=['uniformnoise'] #, 'uniformnoiseffonly', 'duration', 'interval'],
        ),

rule training:
    input:
        expand(project_paths.figures / '{experiment}' / 'DyRCNNx8:tsteps=20+dt=2+lossrt=4+pattern={pattern}+energyloss=*_{seeds}' / 'imagenette:all_{status}' / 'training.png',
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
        expand(project_paths.figures / 'response' / 'DyRCNNx8:pattern=1+{configuration}_{seed}' / 'imagenette:all_{status}' / 'responses.png',
        seed=SEED,
        status=STATUS,
        configuration=[
            'tsteps=20+dataloader=*+dsteps=1',
            'tsteps=1+dataloader=*+dsteps=20',
        ]),

rule references:
    input:
        expand(project_paths.figures / "{experiment}" / "{model}:pretrained=*_{seeds}" / "imagenet:imagenette_init" / "responses.png",
            experiment=['response', 'hundred'],
            model=['CorNetRT', 'CordsNet'],
            seeds=SEED[0], #'.'.join(SEED),
        ),


rule imagenet:  # run with --config use_distributed_mode=True
    input:
        expand(project_paths.models / "DyRCNNx8" / "DyRCNNx8:tsteps=10+rctype=full+dt=2+tau=5+tff=0+trc=6+tsk=0+lossrt=4_{seed}" / "imagenet" / "trained.pt",
        seed=SEED,
        )
