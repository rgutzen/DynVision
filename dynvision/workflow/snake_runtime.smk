"""Model training and evaluation workflow.

This workflow handles all model-related operations including:
- Model initialization
- Training configuration
- Model training
- Model evaluation
- Result collection

Usage:
    # Train a model
    snakemake -c1 train_model model_name=DyRCNNx4 seed=0001

    # Evaluate a model
    snakemake -c1 test_model model_name=DyRCNNx4 seed=0001
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
        dataset = project_paths.data.interim \
            / '{data_name}' \
            / 'train_all' \
            / 'folder.link'
    params:
        config_path = CONFIGS,
        model_arguments = lambda w: parse_arguments(w, 'model_args'),
        init_with_pretrained = config.init_with_pretrained,
        executor_start = config.executor_start if config.use_executor else '',
        executor_close = config.executor_close if config.use_executor else ''
    output:
        model_state = project_paths.models \
            / '{model_name}' \
            / '{model_name}{model_args}_{seed}_{data_name}_init.pt'
    # log:
    #     project_paths.logs / 'init_model_{model_name}{model_args}_{seed}_{data_name}.log'
    benchmark:
        project_paths.benchmarks / 'init_model_{model_name}{model_args}_{seed}_{data_name}.txt'
    shell:
        """
        {params.executor_start}
        python {input.script:q} \
            --config_path {params.config_path:q} \
            --model_name {wildcards.model_name} \
            --dataset {input.dataset:q} \
            --data_name {wildcards.data_name} \
            --seed {wildcards.seed} \
            --output {output.model_state:q} \
            --init_with_pretrained {params.init_with_pretrained} \
            {params.model_arguments} \
        {params.executor_close}
        """
            # > {log} 2>&1

rule train_model:
    """Train a model on specified dataset.

    Input:
        model_state: Initial model state
        dataset_train: Training dataset
        dataset_val: Validation dataset
        script: Training script
    
    Output:
        Trained model state dict

    Parameters:
        config_path: Path to configuration file
        epochs: Number of training epochs
        batch_size: Training batch size
        model_arguments: Model-specific arguments
        learning_rate: Training learning rate
        loss: Loss function configuration
        resolution: Input resolution
        check_val_every_n_epoch: Validation frequency
        accumulate_grad_batches: Gradient accumulation steps
        precision: Training precision
        store_responses: Number of responses to store
        profiler: Training profiler configuration
        enable_progress_bar: Whether to show progress bar
        execution_cmd: Complete execution command with conditional wrappers
    """
    input:
        model_state = project_paths.models \
            / '{model_name}' \
            / '{model_name}{model_args}_{seed}_{data_name}_init.pt',
        dataset_train = project_paths.data.processed \
            / '{data_name}' \
            / 'train_all' \
            / 'train.beton',
        dataset_val = project_paths.data.processed \
            / '{data_name}' \
            / 'train_all' \
            / 'val.beton',
        script = SCRIPTS / 'runtime' / 'train_model.py'
    params:
        config_path = CONFIGS,
        epochs = config.epochs,
        batch_size = config.batch_size if project_paths.iam_on_cluster() else config.debug_batch_size,
        model_arguments = lambda w: parse_arguments(w, 'model_args'),
        n_timesteps = config.n_timesteps, 
        learning_rate = config.learning_rate,
        loss = config.loss,
        resolution = lambda w: config.data_resolution[w.data_name],
        check_val_every_n_epoch = config.check_val_every_n_epoch if project_paths.iam_on_cluster() else config.debug_check_val_every_n_epoch,
        log_every_n_steps = config.log_every_n_steps if project_paths.iam_on_cluster() else config.debug_log_every_n_steps,
        accumulate_grad_batches = config.accumulate_grad_batches if project_paths.iam_on_cluster() else config.debug_accumulate_grad_batches,
        precision = config.precision,
        store_responses = config.store_val_responses,
        profiler = config.profiler,
        use_ffcv = config.use_ffcv,
        enable_progress_bar = config.enable_progress_bar if project_paths.iam_on_cluster() else config.debug_enable_progress_bar,
        use_distributed = config.use_distributed,
        # Build complete execution command with conditional wrappers
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=getattr(config, 'use_distributed', False),
            use_executor=getattr(config, 'use_executor', False)
        ),
        debug_script = SCRIPTS / 'runtime' / 'debug_ddp.py',
    output:
        model_state = project_paths.models \
            / '{model_name}' \
            / '{model_name}{model_args}_{seed}_{data_name}_trained.pt'
    # log:
    #     project_paths.logs / 'train_model_{model_name}{model_args}_{seed}_{data_name}.log'
    benchmark:
        project_paths.benchmarks / 'train_model_{model_name}{model_args}_{seed}_{data_name}.txt'
    shell:
        """
        {params.execution_cmd} \
            --config_path {params.config_path:q} \
            --input_model_state {input.model_state:q} \
            --model_name {wildcards.model_name} \
            --dataset_train {input.dataset_train:q} \
            --dataset_val {input.dataset_val:q} \
            --data_name {wildcards.data_name} \
            --output_model_state {output.model_state:q} \
            --learning_rate {params.learning_rate} \
            --epochs {params.epochs} \
            --batch_size {params.batch_size} \
            --seed {wildcards.seed} \
            --check_val_every_n_epoch {params.check_val_every_n_epoch} \
            --log_every_n_steps {params.log_every_n_steps} \
            --accumulate_grad_batches {params.accumulate_grad_batches} \
            --resolution {params.resolution} \
            --precision {params.precision} \
            --profiler {params.profiler} \
            --store_responses {params.store_responses} \
            --enable_progress_bar {params.enable_progress_bar} \
            --use_ffcv {params.use_ffcv} \
            --loss {params.loss} \
            --n_timesteps {params.n_timesteps} \
            {params.model_arguments} \
        """

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
        config_path: Path to configuration file
        batch_size: Evaluation batch size
        data_group: Data grouping configuration
        model_arguments: Model-specific arguments
        data_arguments: Data-specific arguments
        loss: Loss function configuration
        store_responses: Number of responses to store
        enable_progress_bar: Whether to show progress bar
    """
    input:
        model_state = project_paths.models \
            / '{model_name}' \
            / '{model_name}{model_args}_{seed}_{data_name}_{status}.pt',
        dataset = project_paths.data.interim \
            / '{data_name}' \
            / 'test_{data_group}' \
            / 'folder.link',
        script = SCRIPTS / 'runtime' / 'test_model.py'
    params:
        config_path = CONFIGS,
        batch_size = config.batch_size if project_paths.iam_on_cluster() else 3,
        data_group = lambda w: f"{w.data_name}_{w.data_group}",
        model_arguments = lambda w: parse_arguments(w, 'model_args'),
        data_arguments = lambda w: parse_arguments(w, 'data_args'),
        loss = config.loss,
        store_responses = config.store_test_responses,
        enable_progress_bar = True,
        verbose = config.verbose,
        executor_start = config.executor_start if config.use_executor else '',
        executor_close = config.executor_close if config.use_executor else '',
    output:
        responses = project_paths.models \
            / '{model_name}' \
            / '{model_name}{model_args}_{seed}_{data_name}_{status}_{data_loader}{data_args}_{data_group}_test_responses.pt',
        results = project_paths.reports \
            / '{model_name}' \
            / '{model_name}{model_args}_{seed}_{data_name}_{status}_{data_loader}{data_args}_{data_group}_test_outputs.csv'
    # log:
    #     project_paths.logs / 'test_model_{model_name}{model_args}_{seed}_{data_name}_{status}_{data_loader}{data_args}_{data_group}.log'
    benchmark:
        project_paths.benchmarks / 'test_model_{model_name}{model_args}_{seed}_{data_name}_{status}_{data_loader}{data_args}_{data_group}.txt'
    shell:
        """
        {params.executor_start}
        python {input.script:q} \
            --config_path {params.config_path:q} \
            --input_model_state {input.model_state:q} \
            --model_name {wildcards.model_name} \
            --data_name {wildcards.data_name} \
            --dataset {input.dataset:q} \
            --data_loader {wildcards.data_loader} \
            --output_results {output.results:q} \
            --output_responses {output.responses:q} \
            --data_transform {wildcards.data_name} \
            --target_transform {params.data_group} \
            --loss {params.loss} \
            --batch_size {params.batch_size} \
            --seed {wildcards.seed} \
            --store_responses {params.store_responses} \
            --enable_progress_bar {params.enable_progress_bar} \
            --verbose {params.verbose} \
            {params.model_arguments} \
            {params.data_arguments} \
        {params.executor_close}
        """
            # > {log} 2>&1

logger.info("Model workflow initialized")
logger.info(f"Model directory: {project_paths.models}")
logger.info(f"Reports directory: {project_paths.reports}")