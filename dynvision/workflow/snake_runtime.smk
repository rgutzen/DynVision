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
        model_arguments = lambda w: parse_arguments(w, 'model_args'),
        dataset_path = lambda w: project_paths.data.interim / w.data_name / 'train_all',
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
        ),
    priority: 0
    output:
        model_state = project_paths.models \
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

    This is a checkpoint rule that creates symlinks for hash-based access.
    After training, creates:
    - Hash documentation file: {model_identifier}/{data_name}/{hash}.hash
    - Symlink: {model_name}:hash={hash} -> {model_name}{model_args}_{seed}

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
        model_arguments = lambda w: parse_arguments(w, 'model_args'),
        dataset_link = lambda w: project_paths.data.interim / w.data_name / 'train_all',
        resolution = lambda w: config.data_resolution[w.data_name],
        normalize = lambda w: json.dumps((
            config.data_statistics[w.data_name]['mean'],
            config.data_statistics[w.data_name]['std']
        )),
        # Symlink and hash parameters
        model_folder = lambda w: project_paths.models / f"{w.model_name}{w.model_args}_{w.seed}",
        symlink_folder = lambda w: project_paths.models / f"{w.model_name}{compute_hash(w.model_args, w.seed)}",
        hash_file = lambda w: project_paths.models / f"{w.model_name}{w.model_args}_{w.seed}" / w.data_name / f"{compute_hash(w.model_args, w.seed).lstrip(':')}.hash",
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=get_param('use_distributed_mode', False)(w),
        ),
    priority: 2
    output:
        model_state = project_paths.models \
            / '{model_name}{model_args}_{seed}' \
            / '{data_name}' \
            / 'trained.pt'
    shell:
        """
        {params.execution_cmd} \
            --config_path {params.base_config_path:q} \
            --input_model_state {input.model_state:q} \
            --output_model_state {output.model_state:q} \
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

        # Document hash for this model
        echo "{wildcards.model_args}_{wildcards.seed}" > {params.hash_file}

        # Create symlink at model level (if not exists)
        if [ ! -e {params.symlink_folder} ]; then
            ln -s {params.model_folder} {params.symlink_folder}
        fi
        """

use rule train_model as train_model_distributed with:
    output:
        # todo: find more general fix to automatically switch slurm resource requests for distributed mode
        model_state = project_paths.models \
            / '{model_name}{model_args}_{seed}' \
            / '{data_name,imagenet}' \
            / 'trained.pt'

rule test_model:
    """Evaluate a trained model on test data.

    Uses polymorphic {model_identifier} wildcard that matches:
    - Full form: {model_args}_{seed} (e.g., tsteps=20+dt=2_42)
    - Hash form: hash={hash_id} (e.g., hash=a7f3c9d4)

    Input:
        model_state: Trained model state (accessed via symlink if hashed)
        dataset: Test dataset
        script: Evaluation script

    Output:
        Test results in hierarchical structure:
        {experiment}/{model_name}:{model_identifier}/{data_name}:{data_group}_{status}/{data_loader}{data_args}/

    Parameters:
        config_path: Path to configuration file with wildcards and modes applied
        batch_size: Evaluation batch size
        data_group: Data grouping configuration
        model_arguments: Model-specific arguments
        data_arguments: Data-specific arguments
        loss: Loss function configuration
        enable_progress_bar: Whether to show progress bar
    """
    input:
        model_state = project_paths.models \
            / '{model_name}:{model_identifier}' \
            / '{data_name}' \
            / '{status}.pt',
        dataset_ready = project_paths.data.interim \
            / '{data_name}' \
            / 'test_{data_group}.ready',
        script = SCRIPTS / 'runtime' / 'test_model.py'
    params:
        base_config_path = WORKFLOW_CONFIG_PATH,
        model_arguments = lambda w: parse_arguments(w, 'model_args') if 'hash=' not in w.model_identifier else "",
        data_arguments = lambda w: parse_arguments(w, 'data_args'),
        # Extract seed from model_identifier (either from full form or hash file)
        seed = lambda w: (
            w.model_identifier.split('_')[-1] if 'hash=' not in w.model_identifier
            else open(list(Path(f"{project_paths.models}/{w.model_name}:{w.model_identifier}/{w.data_name}").glob('*.hash'))[0]).read().strip().split('_')[-1]
        ),
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
            / '{model_name}:{model_identifier}' \
            / '{data_name}:{data_group}_{status}' \
            / '{data_loader}{data_args}' / 'test_responses.pt',
        results = project_paths.reports \
            / '{experiment}' \
            / '{model_name}:{model_identifier}' \
            / '{data_name}:{data_group}_{status}' \
            / '{data_loader}{data_args}' / 'test_outputs.csv'
    log:
        project_paths.logs / "slurm" / "rule_test_model" \
            / '{experiment}' \
            / '{model_name}:{model_identifier}' \
            / '{data_name}:{data_group}_{status}' \
            / '{data_loader}{data_args}.log'
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
            --data_loader {wildcards.data_loader} \
            --data_group {wildcards.data_group} \
            --seed {params.seed} \
            --normalize {params.normalize:q} \
            --enable_progress_bar {params.enable_progress_bar} \
            {params.model_arguments} \
            {params.data_arguments} \
            --batch_size {params.batch_size} \
        """

rule best_checkpoint_to_statedict:
    """Convert Lightning checkpoints to state dictionaries (hierarchical structure)."""
    input:
        model = project_paths.models / '{model_name}{model_args}_{seed}' / '{data_name}' / 'trained.pt',
        script = project_paths.scripts.utils / 'checkpoint_to_statedict.py'
    params:
        checkpoint_dir = lambda w: project_paths.models / f"{w.model_name}" / 'checkpoints',
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
        ),
    output:
        project_paths.models / '{model_name}{model_args}_{seed}' / '{data_name}' / 'trained-best.pt'
    shell:
        """
        {params.execution_cmd} \
            --checkpoint_dir {params.checkpoint_dir:q} \
            --output {output:q}
        """

checkpoint intermediate_checkpoint_to_statedict:
    """Convert Lightning checkpoints to state dictionaries (hierarchical structure)."""
    input:
        # model = project_paths.models / '{model_name}{model_args}_{seed}' / '{data_name}' / 'trained.pt',
        script = project_paths.scripts.utils / 'checkpoint_to_statedict.py'
    params:
        checkpoint_dir = lambda w: project_paths.models / f"{w.model_name}" / 'checkpoints',
        checkpoint_globs = lambda w: f"{w.model_name}{w.model_args}_{w.seed}_{w.data_name}_trained*.ckpt",
        output_dir = lambda w: project_paths.models / f"{w.model_name}{w.model_args}_{w.seed}" / f"{w.data_name}",
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
        ),
    output:
        project_paths.models / '{model_name}{model_args}_{seed}' / '{data_name}' / 'trained-epoch={epoch}.pt'
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
        # Trigger train_model checkpoint for all category values
        models = lambda w: [project_paths.models \
            / f"{{model_name}}{{args1}}{w.category_str.strip('*')}={cat_value}{{args2}}_{{seed}}" \
            / "{{data_name}}" \
            / f"{config.experiment_config[w.experiment].get('status', w.status)}.pt"
            for cat_value in config.experiment_config[w.experiment]['categories'].get(w.category_str.strip('=*'), [])],
        # Use hash-compressed identifiers for test outputs
        responses = lambda w: expand(project_paths.reports \
            / '{{experiment}}' \
            / f"{{{{model_name}}}}:{compute_hash(f'{{{{args1}}}}{w.category_str.strip('*')}={{cat_value}}{{{{args2}}}}', '{{seed}}')}" \
            / '{{data_name}}:{{data_group}}_{status}' \
            / '{data_loader}{data_args}' / 'test_responses.pt',
            cat_value = config.experiment_config[w.experiment]['categories'].get(w.category_str.strip('=*'), []),
            status = config.experiment_config[w.experiment].get('status', w.status),
            data_loader = config.experiment_config[w.experiment]['data_loader'],
            data_args = args_product(config.experiment_config[w.experiment]['data_args']),
        ),
        test_outputs = lambda w: expand(project_paths.reports \
            / '{{experiment}}' \
            / f"{{{{model_name}}}}:{compute_hash(f'{{{{args1}}}}{w.category_str.strip('*')}={{cat_value}}{{{{args2}}}}', '{{seed}}')}" \
            / '{{data_name}}:{{data_group}}_{status}' \
            / '{data_loader}{data_args}' / 'test_outputs.csv',
            cat_value = config.experiment_config[w.experiment]['categories'].get(w.category_str.strip('=*'), []),
            status = config.experiment_config[w.experiment].get('status', w.status),
            data_loader = config.experiment_config[w.experiment]['data_loader'],
            data_args = args_product(config.experiment_config[w.experiment]['data_args']),
        ),
        script = SCRIPTS / 'visualization' / 'process_test_data.py'
    params:
        measures = ['response_avg', 'response_std', 'guess_confidence', 'first_label_confidence', 'accuracy_top3', 'accuracy_top5'], # 'spatial_variance', 'feature_variance', 'classifier_top5', 'label_confidence',
        parameter = lambda w: config.experiment_config[w.experiment]['parameter'],
        category = lambda w: w.category_str.strip('=*'),
        # Pass category values to script for proper labeling
        cat_values = lambda w: config.experiment_config[w.experiment]['categories'].get(w.category_str.strip('=*'), []),
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
            / '{model_name}{args1}{category_str}{args2}_{seed}' \
            / '{data_name}:{data_group}_{status}' \
            / 'test_data.csv',
    shell:
        """
        {params.execution_cmd} \
            --responses {input.responses:q} \
            --test_outputs {input.test_outputs:q} \
            --output {output.test_data:q} \
            --parameter {params.parameter} \
            --category {params.category} \
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
