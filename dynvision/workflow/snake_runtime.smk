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
        Initialized model state dict

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
            / '{model_name}' \
            / '{model_name}{model_args}_{seed}_{data_name}_init.pt'
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

rule train_model:
    """Train a model on specified dataset.

    Input:
        model_state: Initial model state
        dataset_train: Training dataset
        dataset_val: Validation dataset
        script: Training script
    
    Output:
        Trained model state dict
    """
    input:
        model_state = project_paths.models \
            / '{model_name}' \
            / '{model_name}{model_args}_{seed}_{data_name}_init.pt',
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
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=get_param('use_distributed_mode', False)(w),
        ),
    priority: 2
    output:
        model_state = project_paths.models \
            / '{model_name}' \
            / '{model_name}{model_args}_{seed}_{data_name}_trained.pt'
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
        """

use rule train_model as train_model_distributed with:
    output:
        # todo: find more general fix to automatically switch slurm resource requests for distributed mode
        model_state = project_paths.models \
            / '{model_name}' \
            / '{model_name}{model_args}_{seed}_{data_name,imagenet}_trained.pt'

rule test_model:
    """Evaluate a trained model on test data.

    Input:
        model_state: Trained model state
        dataset: Test dataset
        script: Evaluation script
    
    Output:
        responses: Model responses
        results: Evaluation results

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
            / '{model_name}' \
            / '{model_name}{model_args}_{seed}_{data_name}_{status}.pt',
        dataset_ready = project_paths.data.interim \
            / '{data_name}' \
            / 'test_{data_group}.ready',
        script = SCRIPTS / 'runtime' / 'test_model.py'
    params:
        base_config_path = WORKFLOW_CONFIG_PATH,
        model_arguments = lambda w: parse_arguments(w, 'model_args'),
        data_arguments = lambda w: parse_arguments(w, 'data_args'),
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
            / '{data_loader}' \
            / '{model_name}{model_args}_{seed}_{data_name}_{status}_{data_loader}{data_args}_{data_group}' / 'test_responses.pt',
        results = project_paths.reports \
            / '{data_loader}' \
            / '{model_name}{model_args}_{seed}_{data_name}_{status}_{data_loader}{data_args}_{data_group}' / 'test_outputs.csv'
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
            --seed {wildcards.seed} \
            --normalize {params.normalize:q} \
            --enable_progress_bar {params.enable_progress_bar} \
            {params.model_arguments} \
            {params.data_arguments} \
            --batch_size {params.batch_size} \
        """

rule best_checkpoint_to_statedict:
    """Convert Lightning checkpoints to state dictionaries."""
    input:
        model = project_paths.models / '{model_name}' / '{model_name}{model_args}_{seed}_{data_name}_trained.pt',
        script = project_paths.scripts.utils / 'checkpoint_to_statedict.py'
    params:
        checkpoint_dir = lambda w: project_paths.models / f"{w.model_name}" / 'checkpoints',
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
        ),
    output:
        project_paths.models / '{model_name}' / '{model_name}{model_args}_{seed}_{data_name}_trained-best.pt'
    shell:
        """
        {params.execution_cmd} \
            --checkpoint_dir {params.checkpoint_dir:q} \
            --output {output:q}
        """

checkpoint intermediate_checkpoint_to_statedict:
    """Convert Lightning checkpoints to state dictionaries."""
    input:
        # model = project_paths.models / '{model_name}' / '{model_name}{model_args}_{seed}_{data_name}_trained.pt',
        script = project_paths.scripts.utils / 'checkpoint_to_statedict.py'
    params:
        checkpoint_dir = lambda w: project_paths.models / f"{w.model_name}" / 'checkpoints',
        checkpoint_globs = lambda w: f"{w.model_name}{w.model_args}_{w.seed}_{w.data_name}_trained*.ckpt",
        output_dir = lambda w: project_paths.models / f"{w.model_name}",
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
        ),
    output:
        project_paths.models / '{model_name}' / '{model_name}{model_args}_{seed}_{data_name}_trained-epoch={epoch}.pt'
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
    
    Input:
        responses: Model layer response files (.pt)
        test_outputs: Test output files (.csv)
        script: Processing script
    
    Output:
        Unified test data CSV with layer metrics and performance metrics
    """
    input:
        responses = expand(project_paths.reports \
            / '{data_loader}' \
            / '{{model_name}}:{{args1}}{category}{category_value}{{args2}}_{{seed}}_{{data_name}}_{status}_{data_loader}{data_args}_{{data_group}}' / 'test_responses.pt',
            category = lambda w: w.category_str.strip('*'),
            category_value = lambda w: config.experiment_config['categories'].get(w.category_str.strip('=*'), '') if w.category_str else "",
            status = lambda w: config.experiment_config[w.experiment].get('status', w.status),
            data_loader = lambda w: config.experiment_config[w.experiment]['data_loader'],
            data_args = lambda w: args_product(config.experiment_config[w.experiment]['data_args']),
        ),
        test_outputs = expand(project_paths.reports \
            / '{data_loader}' \
            / '{{model_name}}:{{args1}}{category}{category_value}{{args2}}_{{seed}}_{{data_name}}_{status}_{data_loader}{data_args}_{{data_group}}' / 'test_outputs.csv',
            category = lambda w: w.category_str.strip('*'),
            category_value = lambda w: config.experiment_config['categories'].get(w.category_str.strip('=*'), '') if w.category_str else "",
            status = lambda w: config.experiment_config[w.experiment].get('status', w.status),
            data_loader = lambda w: config.experiment_config[w.experiment]['data_loader'],
            data_args = lambda w: args_product(config.experiment_config[w.experiment]['data_args']),
        ),
        script = SCRIPTS / 'visualization' / 'process_test_data.py'
    params:
        measures = ['response_avg', 'response_std', 'guess_confidence', 'first_label_confidence', 'accuracy_top3', 'accuracy_top5'], # 'spatial_variance', 'feature_variance', 'classifier_top5', 'label_confidence',
        parameter = lambda w: config.experiment_config[w.experiment]['parameter'],
        category = lambda w: w.category_str.strip('=*'),
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
        test_data = project_paths.reports / '{experiment}' / '{experiment}_{model_name}:{args1}{category_str}{args2}_{seed}_{data_name}_{status}_{data_group}' / 'test_data.csv',
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
