"""Model training and evaluation workflow.

This workflow handles all model-related operations including:
- Model initialization
- Training configuration
- Model training
- Model evaluation
- Result collection
"""

logger = logging.getLogger('workflow.runtime')

rule init_model:
    """Initialize a model with specified configuration.

    Input:
        script: Model initialization script
        dataset: Training dataset for shape inference

    Output:
        Initialized model state dict (hierarchical: {model_identifier}/{data_name}/init.pt)

    Parameters:
        config_path: Path to configuration file
        model_arguments: Model-specific arguments
        init_with_pretrained: Whether to initialize from pretrained weights
    """
    input:
        script = SCRIPTS / 'runtime' / 'init_model.py',
        dataset_ready = project_paths.data.interim \
            / '{data_name}' \
            / 'train_all.ready'
    params:
        base_config_path = WORKFLOW_CONFIG_PATH,
        model_arguments = lambda w: parse_arguments(w.model_args),
        dataset_path = lambda w: project_paths.data.interim / w.data_name / 'train_all',
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
        ),
    priority: 0
    output:
        model_state = project_paths.models \
            / '{model_name}' \
            / '{model_name}{model_args}_{seed}' \
            / '{data_name}' \
            / 'init.pt'
    shell:
        """
        {params.execution_cmd} \
            --config_path {params.base_config_path:q} \
            --model_name {wildcards.model_name} \
            --dataset_path {params.dataset_path:q} \
            --data_name {wildcards.data_name} \
            --seed {wildcards.seed} \
            --output {output.model_state:q} \
            {params.model_arguments}
        """

checkpoint train_model:
    """Train a model on specified dataset.

    This is a checkpoint rule that enables data-dependent DAG re-evaluation.

    Input:
        model_state: Initial model state
        dataset_train: Training dataset
        dataset_val: Validation dataset
        script: Training script

    Output:
        Trained model state dict (hierarchical: {model_identifier}/{data_name}/trained.pt)
    """
    input:
        model_state = project_paths.models \
            / '{model_name}' \
            / '{model_name}{model_args}_{seed}' \
            / '{data_name}' \
            / 'init.pt',
        dataset_ready = project_paths.data.interim \
            / '{data_name}' \
            / 'train_all.ready',
        dataset_train = lambda w: project_paths.data.processed \
            / '{data_name}' \
            / 'train_all' \
            / 'train.beton' if config.use_ffcv else [],
        dataset_val = lambda w: project_paths.data.processed \
            / '{data_name}' \
            / 'train_all' \
            / 'val.beton' if config.use_ffcv else [],
        script = SCRIPTS / 'runtime' / 'train_model.py'
    params:
        base_config_path = WORKFLOW_CONFIG_PATH,
        data_group = "all",
        model_arguments = lambda w: parse_arguments(w.model_args),
        dataset_link = lambda w: project_paths.data.interim / w.data_name / 'train_all',
        resolution = lambda w: config.data_resolution[w.data_name],
        normalize = lambda w: json.dumps((
            config.data_statistics[w.data_name]['mean'],
            config.data_statistics[w.data_name]['std']
        )),
        checkpoint_dir = lambda w: project_paths.models / w.model_name / f'{w.model_name}{w.model_args}_{w.seed}' / w.data_name,
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=get_param('use_distributed_mode', False)(w),
        ),
    priority: 2
    output:
        model_state = project_paths.models \
            / '{model_name}' \
            / '{model_name}{model_args}_{seed}' \
            / '{data_name}' \
            / 'trained.pt',
    shell:
        """
        {params.execution_cmd} \
            --config_path {params.base_config_path:q} \
            --input_model_state {input.model_state:q} \
            --output_model_state {output.model_state:q} \
            --checkpoint_dir {params.checkpoint_dir:q} \
            --model_name {wildcards.model_name} \
            --dataset_link {params.dataset_link:q} \
            --dataset_train {input.dataset_train:q} \
            --dataset_val {input.dataset_val:q} \
            --data_name {wildcards.data_name} \
            --data_group {params.data_group} \
            --seed {wildcards.seed} \
            --resolution {params.resolution} \
            --normalize {params.normalize:q} \
            {params.model_arguments}
        """

use rule train_model as train_model_distributed with:
    output:
        # todo: find more general fix to automatically switch slurm resource requests for distributed mode
        model_state = project_paths.models \
            / '{model_name}' \
            / '{model_name}{model_args}_{seed}' \
            / '{data_name,imagenet}' \
            / 'trained.pt',

rule test_model:
    """Evaluate a trained model on test data.

    Uses {model_identifier} wildcard in form: {model_args}_{seed} (e.g., :tsteps=20+dt=2_42_6000)

    Uses polymorphic {test_identifier} wildcard that matches:
    - Compressed mode: abc123 (hash of data_loader + data_args)
    - Uncompressed mode: StimulusNoise:dsteps=20+... (full spec)

    Input:
        model_state: Trained model state
        dataset: Test dataset
        script: Evaluation script

    Output:
        Test results in hierarchical structure:
        {experiment}/{model_name}{model_identifier}/{data_name}:{data_group}_{status}/{test_identifier}/

    Parameters:
        config_path: Path to configuration file with wildcards and modes applied
        batch_size: Evaluation batch size
        data_group: Data grouping configuration
        model_arguments: Model-specific arguments
        data_loader: Parsed from test_identifier
        data_arguments: Parsed from test_identifier
        loss: Loss function configuration
        enable_progress_bar: Whether to show progress bar
    """
    input:
        model_state = project_paths.models \
            / '{model_name}' \
            / '{model_name}{model_identifier}' \
            / '{data_name}' \
            / '{status}.pt',
        dataset_ready = project_paths.data.interim \
            / '{data_name}' \
            / 'test_{data_group}.ready',
        script = SCRIPTS / 'runtime' / 'test_model.py'
    params:
        base_config_path = WORKFLOW_CONFIG_PATH,
        model_arguments = lambda w: parse_arguments(w.model_identifier),
        # Parse test_identifier to get loader and args
        data_loader = lambda w: parse_test_identifier(w.test_identifier)[0],
        data_args_string = lambda w: parse_test_identifier(w.test_identifier)[1],
        data_arguments = lambda w: parse_arguments(parse_test_identifier(w.test_identifier)[1]),
        seed = lambda w: w.model_identifier.split('_')[-1],
        dataset_path = lambda w: project_paths.data.interim / w.data_name / f'test_{w.data_group}',
        normalize = lambda w: (
            # Allow override via --config normalize=null
            config.normalize if hasattr(config, 'normalize') else json.dumps((
                config.data_statistics[w.data_name]['mean'],
                config.data_statistics[w.data_name]['std']
            ))
        ),
        batch_size = config.test_batch_size,
        enable_progress_bar = True,
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
        ),
    priority: 1
    output:
        responses = project_paths.reports \
            / '{experiment}' \
            / '{model_name}{model_identifier}' \
            / '{data_name}:{data_group}_{status}' \
            / '{test_identifier}' / 'test_responses.pt',
        results = project_paths.reports \
            / '{experiment}' \
            / '{model_name}{model_identifier}' \
            / '{data_name}:{data_group}_{status}' \
            / '{test_identifier}' / 'test_outputs.csv',
        meta_data = project_paths.reports \
            / '{experiment}' \
            / '{model_name}{model_identifier}' \
            / '{data_name}:{data_group}_{status}' \
            / '{test_identifier}' / 'test_outputs.csv.config.yaml'  # generated automatically by TestingParams
    shell:
        """
        {params.execution_cmd} \
            --config_path {params.base_config_path:q} \
            --input_model_state {input.model_state:q} \
            --output_results {output.results:q} \
            --output_responses {output.responses:q} \
            --model_name {wildcards.model_name} \
            --data_name {wildcards.data_name} \
            --dataset_path {params.dataset_path:q} \
            --data_loader {params.data_loader} \
            --data_group {wildcards.data_group} \
            --seed {params.seed} \
            --normalize {params.normalize:q} \
            --enable_progress_bar {params.enable_progress_bar} \
            {params.model_arguments} \
            {params.data_arguments} \
            --batch_size {params.batch_size} \
        """

rule best_checkpoint_to_statedict:
    """Convert Lightning checkpoints to state dictionaries."""
    input:
        model = project_paths.models / '{model_name}' / '{model_name}{model_args}_{seed}' / '{data_name}' / 'trained.pt',
        script = project_paths.scripts.utils / 'checkpoint_to_statedict.py'
    params:
        checkpoint_dir = lambda w: project_paths.models / w.model_name / f'{w.model_name}{w.model_args}_{w.seed}' / w.data_name,
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
        ),
    output:
        project_paths.models / '{model_name}' / '{model_name}{model_args}_{seed}' / '{data_name}' / 'trained-best.pt'
    shell:
        """
        {params.execution_cmd} \
            --checkpoint_dir {params.checkpoint_dir:q} \
            --output {output:q}
        """

checkpoint intermediate_checkpoint_to_statedict:
    """Convert Lightning checkpoints to state dictionaries."""
    input:
        # model = project_paths.models / '{model_name}' / '{model_name}{model_identifier}' / '{data_name}' / 'trained.pt',   # comment out to retrieve checkpoints from unfinished trainings
        script = project_paths.scripts.utils / 'checkpoint_to_statedict.py'
    params:
        checkpoint_dir = lambda w: project_paths.models / w.model_name / f'{w.model_name}{w.model_args}_{w.seed}' / w.data_name,
        checkpoint_globs = lambda w: f"trained*.ckpt",
        output_dir = lambda w: project_paths.models / w.model_name / f"{w.model_name}{w.model_args}_{w.seed}" / f"{w.data_name}",
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
        ),
    output:
        project_paths.models / '{model_name}' / '{model_name}{model_args}_{seed}' / '{data_name}' / 'trained-epoch={epoch}.pt'
        # /scratch/rg5022/rhythmic_visual_attention/models/DyRCNNx8/DyRCNNx8:tsteps=20+dt=2+lossrt=4+pattern=1+energyloss=0.2_5000/imagenette/trained-epoch=149.pt
    shell:
        """
        {params.execution_cmd} \
            --checkpoint_dir {params.checkpoint_dir:q} \
            --output_dir {params.output_dir:q} \
            --checkpoint_globs {params.checkpoint_globs:q}
        """


rule process_test_data:
    """Process test data by combining layer responses and test performance metrics.

    This unified rule combines the functionality of process_plotting_data and
    process_test_performance to create a single comprehensive dataset.

    Uses hash-compressed model identifiers for category sweeps to avoid
    filesystem length limitations.

    Input:
        models: Trained model files (triggers train_model checkpoint)
        responses: Model layer response files (.pt) from test_model
        test_outputs: Test output files (.csv) from test_model
        script: Processing script

    Output:
        Unified test data CSV with layer metrics and performance metrics
        Output path: {experiment}/{model_identifier}/{data_name}:{data_group}_{status}/test_data.csv
    """
    input:
        # Collect test outputs from all category values
        test_responses = expand(project_paths.reports \
            / '{{experiment}}' \
            / "{{model_name}}{{args1}}{{category}}={cat_value}{{args2}}_{{seed}}" \
            / '{{data_name}}:{{data_group}}_{status}' \
            / '{test_identifier}' / 'test_responses.pt',
            status = lambda w: config.experiment_config[w.experiment].get('status', w.status),
            cat_value = lambda w: config.experiment_config['categories'].get(w.category, []),
            test_identifier = lambda w: get_test_specs_for_experiment(w.experiment),
        ),
        test_outputs = expand(project_paths.reports \
            / '{{experiment}}' \
            / "{{model_name}}{{args1}}{{category}}={cat_value}{{args2}}_{{seed}}" \
            / '{{data_name}}:{{data_group}}_{status}' \
            / '{test_identifier}' / 'test_outputs.csv',
            status = lambda w: config.experiment_config[w.experiment].get('status', w.status),
            cat_value = lambda w: config.experiment_config['categories'].get(w.category, []),
            test_identifier = lambda w: get_test_specs_for_experiment(w.experiment),
        ),
        test_configs = expand(project_paths.reports \
            / '{{experiment}}' \
            / "{{model_name}}{{args1}}{{category}}={cat_value}{{args2}}_{{seed}}" \
            / '{{data_name}}:{{data_group}}_{status}' \
            / '{test_identifier}' / 'test_outputs.csv.config.yaml',
            status = lambda w: config.experiment_config[w.experiment].get('status', w.status),
            cat_value = lambda w: config.experiment_config['categories'].get(w.category, []),
            test_identifier = lambda w: get_test_specs_for_experiment(w.experiment),
        ),
        script = SCRIPTS / 'visualization' / 'process_test_data.py'
    params:
        measures = ['response_avg', 'response_std', 'guess_confidence', 'first_label_confidence', 'accuracy_top3', 'accuracy_top5'], # 'spatial_variance', 'feature_variance', 'classifier_top5', 'label_confidence',
        parameter = lambda w: config.experiment_config[w.experiment]['parameter'],
        # Pass category values to script for proper labeling
        additional_parameters = 'epoch',
        batch_size = 1,
        remove_input_responses = True,
        fail_on_missing_inputs = False,
        sample_resolution = 'sample',  # 'sample' or 'class'
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
        ),
    priority: 3,
    output:
        test_data = project_paths.reports \
            / '{experiment}' \
            / '{model_name}{args1}{category}=*{args2}_{seed}' \
            / '{data_name}:{data_group}_{status}' \
            / 'test_data.csv',
    shell:
        """
        {params.execution_cmd} \
            --responses {input.test_responses:q} \
            --test_outputs {input.test_outputs:q} \
            --test_configs {input.test_configs:q} \
            --output {output.test_data:q} \
            --parameter {params.parameter} \
            --category {wildcards.category} \
            --measures {params.measures} \
            --batch_size {params.batch_size} \
            --sample_resolution {params.sample_resolution} \
            --additional_parameters {params.additional_parameters} \
            --remove_input_responses {params.remove_input_responses} \
            --fail_on_missing_inputs {params.fail_on_missing_inputs}
        """

rule process_wandb_data:
    input:
        data = Path('{path}.csv'),
        script = SCRIPTS / 'visualization' / 'process_wandb_data.py'
    params:
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
        ),
    output:
        Path('{path}_summary.csv'),
    shell:
        """
        {params.execution_cmd} \
            --data {input.data:q} \
            --output {output:q}
        """
