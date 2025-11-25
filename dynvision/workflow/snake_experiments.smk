# SEED = ["1000", "1001", "1002"] #, "1003", "1004","1005"]  # energy loss weight 0.05 & use_ffcv=False
# SEED = ["2000", "2001", "2002"] # energy loss weight 0.2 & use_ffcv=True
# SEED = ["3000", "3001", "3002"] # energy loss weight not applied? & use_ffcv=True
# SEED = ["4000", "4001", "4002"] # energy loss weight 0.5 & use_ffcv=True
SEED = ["5000", "5001", "5002"] # pattern = 10011 & shuffle pattern = True
STATUS = 'trained-best'

# ###############
# for cmd in \
#     "rctarget" \
#     "rctype" \
    # "skip" \
    # "feedback --config batch_size=128" \
#     "tsteps" \
#     "lossrt" \
#     "idle" \
#     "timeparams" \
#     "stability --config test_batch_size=16" \
#     "references --config test_batch_size=32" \
#     "unrolling --config batch_size=128" \
#     "energyloss --config epochs=201" \
#     "dataloader --config epochs=100" \
#     "training --allowed-rules test_model process_test_data plot_responses" \
# ; do
#     sh snakecharm.sh "$cmd"
#     sleep 120
# done
# # sh snakecharm.sh "imagenet --config use_distributed_mode=True batch_size=512"
# ###############


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
        # unrolling figure
        expand(project_paths.models / "DyRCNNx8" / "DyRCNNx8:tsteps=30+rctype=full+rctarget=output+dt=2+tau=5+tff=10+trc=6+tsk=20+tfb=10+skip=true+feedback=true+lossrt=4_{seed}_imagenette_{status}.pt",
        seed=SEED[0],
        status=STATUS,
        ),
        # energy loss scan
        expand(project_paths.figures / 'response' / 'response_DyRCNNx8:tsteps=20+rctype=full+rctarget=output+dt=2+tau=5+tff=0+trc=6+tsk=0+skip=true+lossrt=4+energyloss=*_{seeds}_imagenette_{status}_all' / 'responses.png',
            seeds=SEED[0], #'.'.join(SEED),
            status=STATUS,
        ),
        # dataloader comparison
        expand(project_paths.figures / 'response' / 'response_DyRCNNx8:{configuration}_{seed}_imagenette_{status}_all' / 'responses.png',
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
    
rule idle:
    input:
        expand(project_paths.reports / "{experiment}" / "{experiment}_DyRCNNx8:tsteps=20+rctype=full+rctarget=output+dt=2+tau=5+tff=0+trc=6+tsk=0+skip=true+lossrt=4+idle=*_{seed}_imagenette_{status}_all" / "test_data.csv",
            seed=SEED,
            experiment=['response'],
            status=STATUS)

rule feedback:  # run with --config batch_size=128
    input:
        expand(project_paths.reports / "{experiment}" / "{experiment}_DyRCNNx8:tsteps=30+rctype=full+rctarget=output+dt=2+tau=5+tff=0+trc=6+tsk=0+tfb=30+skip=true+feedback=*+lossrt=4_{seed}_imagenette_{status}_all" / "test_data.csv",
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
        seed=["1002", "1003", "1004"],  # non-oscillatory seeds only
        experiment=['responseffonly', 'response', 'duration', 'contrast', 'interval', 'uniformnoise', 'uniformnoiseffonly', 'poissonnoise', 'poissonnoiseffonly', 'gaussiannoise', 'gaussiannoiseffonly', 'gaussiancorrnoise', 'gaussiancorrnoiseffonly', 'phasescramblednoise', 'phasescramblednoiseffonly'],
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
        expand(project_paths.figures / 'responseintermediate' / 'responseintermediate_DyRCNNx8:tsteps=20+dt=2+lossrt=4+pattern={pattern}+shufflepattern={shuffle}+energyloss=*_{seeds}_imagenette_{status}_all' / 'responses.png',
                    seeds='.'.join(SEED),
                    pattern='1011',
                    shuffle='true',
                    status=STATUS,
                ),

rule ckptmodels:
    input:
        expand(project_paths.models / "DyRCNNx8" / "DyRCNNx8:tsteps=20+dt=2+lossrt=4+pattern={pattern}+shufflepattern={shuffle}+energyloss={eloss}_{seeds}_imagenette_{status}.pt",
            seeds=SEED[0], #.'.join(SEED),
            pattern='1011',
            shuffle='true',
            status="trained-epoch=49",
            eloss=config.experiment_config['categories'].get('energyloss')
        ),


rule energylossprep:
    input:
        expand(project_paths.models / "DyRCNNx8" / "DyRCNNx8:tsteps=20+rctype=full+rctarget=output+dt=2+tau=5+tff=0+trc=6+tsk=0+skip=true+lossrt=4+energyloss={weight}_{seed}_imagenette_trained-epoch=349.pt",
        seed=SEED,
        weight=['0.0', '0.01', '0.05', '0.1', '0.2', '0.5', '1.0', '2.0', '4.0']
        ),
# rule training:
#     input:
#         expand(project_paths.figures / 'responseintermediate' / 'responseintermediate_DyRCNNx8:tsteps=20+rctype=full+rctarget=output+dt=2+tau=5+tff=0+trc=6+tsk=0+skip=true+lossrt=4+energyloss=*_{seeds}_imagenette_trained_all' / 'responses.png',
#                     seeds=SEED[0], #'.'.join(SEED),
#                 ),

rule dataloader:
    input:
        expand(project_paths.figures / 'response' / 'response_DyRCNNx8:{configuration}_{seed}_imagenette_{status}_all' / 'response.png',
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
