"""Visualization workflow for model analysis and results.

This workflow handles all visualization-related tasks including:
- Confusion matrix generation
- Classifier response analysis
- Weight distribution visualization
- Experiment result plotting
- Adaptation analysis
- Interactive notebook generation
- Enhanced model visualization
- Model architecture visualization
- Enhanced weight analysis
- Temporal dynamics visualization
- Interactive notebook generation

Usage:
    # Generate confusion matrix
    snakemake -j1 plot_confusion_matrix model_name=DyRCNNx4

    # Analyze classifier responses
    snakemake -j1 plot_classifier_responses model_name=DyRCNNx4
"""

logger = logging.getLogger('workflow.visualizations')

# Configuration for interactive visualizations
notebook_config = {
    'template_dir': SCRIPTS / 'visualization' / 'examples',
    'template_name': 'visualization_examples.ipynb',
    'output_dir': project_paths.reports / 'notebooks'
}


rule generate_all_visualizations:
    """Generate all visualizations for a model.

    This rule creates all available visualizations including:
    - Model architecture and temporal flow
    - Weight distributions and layer statistics
    - Classifier responses
    - Interactive notebook

    The visualizations provide comprehensive insights into:
    - Network architecture and connectivity
    - Layer-wise weight patterns and statistics
    - Model behavior and responses
    - Interactive analysis capabilities
    """
    input:
        # Architecture visualizations
        architecture = project_paths.figures / 'architecture' / '{model_name}{data_identifier}_{status}_architecture.png',
        temporal = project_paths.figures / 'architecture' / '{model_name}{data_identifier}_{status}_temporal_flow.png',
        
        # Weight analysis
        distributions = project_paths.figures / 'weights' / '{model_name}{data_identifier}_{status}_distributions.png',
        statistics = project_paths.figures / 'weights' / '{model_name}{data_identifier}_{status}_statistics.png',
        
        # Interactive analysis
        notebook = notebook_config['output_dir'] / '{model_name}' / 'analysis_{model_name}{data_identifier}_{status}.ipynb'
    group: "visualization"

# rule generate_interactive_notebook:
#     """Generate interactive visualization notebook.

#     This rule creates a Jupyter notebook for interactive
#     model analysis and visualization.

#     Input:
#         model: Trained model state
#         template: Example notebook template
    
#     Output:
#         Interactive notebook
#     """
#     input:
#         model = project_paths.models / '{model_name}' / '{model_name}{data_identifier}_{status}.pt',
#         template = notebook_config['template_dir'] / notebook_config['template_name']
#     output:
#         notebook = notebook_config['output_dir'] / '{model_name}' / 'analysis_{model_name}{data_identifier}_{status}.ipynb'
#     group: "visualization"
#     run:
#         output.notebook.parent.mkdir(parents=True, exist_ok=True)
#         generate_analysis_notebook(
#             model=input.model,
#             output_dir=output.notebook.parent,
#             name=output.notebook.stem,
#             template_path=input.template
#         )
#         logger.info(f"Generated interactive notebook: {output.notebook}")

rule plot_layer_analysis:
    input:
        model = project_paths.models / '{model_name}' / '{model_name}{data_identifier}_{status}.pt',
        script = SCRIPTS / 'visualization' / 'enhanced_weight_distributions.py'
    params:
        executor_start = config.executor_start if config.use_executor else '',
        executor_close = config.executor_close if config.use_executor else ''    
    output:
        project_paths.figures / 'weights' / '{model_name}{data_identifier}_{status}_distributions.png',
    group: "visualization"
    shell:
        """
        {params.executor_start}
        python {input.script:q} \
            --model {input.model:q} \
            --output {output:q} 
        {params.executor_close}
        """

rule plot_model_architecture:
    """Visualize model architecture.

    This rule generates static and interactive visualizations
    of model architecture with focus on recurrent connections.

    Input:
        model: Trained model state
        script: Visualization script
    
    Output:
        Architecture visualization
    """
    input:
        state_dict = project_paths.models / '{model_name}' / '{model_name}{data_identifier}_{status}.pt',
        script = SCRIPTS / 'visualization' / 'plot_model_architecture.py'
    params:
        interactive = True,
        executor_start = config.executor_start if config.use_executor else '',
        executor_close = config.executor_close if config.use_executor else ''
    output:
        static = project_paths.figures / 'architecture' / '{model_name}{data_identifier}_{status}_architecture.{format}',
        temporal = project_paths.figures / 'architecture' / '{model_name}{data_identifier}_{status}_temporal_flow.{format}'
    group: "visualization"
    shell:
        """
        {params.executor_start}
        python {input.script:q} \
            --model_name {wildcards.model_name} \
            --state_dict {input.state_dict:q} \
            --output_static {output.static:q} \
            --output_temporal {output.temporal:q} \
            --format {wildcards.format} \
            --interactive {params.interactive}
        {params.executor_close}
        """

rule plot_confusion_matrix:
    """Generate and save confusion matrix visualization.

    Input:
        test_results: Model evaluation results
        dataset: Test dataset for class information
        script: Plotting script
    
    Output:
        Confusion matrix plot

    Parameters:
        palette: Color palette for visualization
    """
    input:
        test_results = project_paths.reports / '{path}_{data_name}_testing_results.csv',
        dataset = project_paths.data.interim / '{data_name}_test' / 'folder.link',
        script = SCRIPTS / 'visualization' / 'plot_confusion_matrix.py'
    params:
        palette = 'cividis',
        executor_start = config.executor_start if config.use_executor else '',
        executor_close = config.executor_close if config.use_executor else ''
    output:
        plot = project_paths.figures / '{path}_{data_name}_confusion.{format}'
    # log:
    #     project_paths.logs / 'plot_confusion_matrix_{path}_{data_name}_{format}.log'
    group: "visualization"
    shell:
        """
        {params.executor_start}
        python {input.script:q} \
            --input {input.test_results:q} \
            --output {output.plot:q} \
            --dataset {input.dataset} \
            --palette {params.palette} \
            --format {wildcards.format} \
        {params.executor_close}
        """
            # > {log} 2>&1

rule plot_classifier_responses:
    """Analyze and visualize classifier responses.

    This rule generates visualizations of model responses
    across different conditions and stimuli.

    Input:
        dataframe: Model output data
        script: Visualization script
    
    Output:
        Directory of response visualizations
    """
    input:
        dataframe = project_paths.reports \
            / '{model_name}' \
            / '{model_name}{data_identifier}_test_outputs.csv',
        script = SCRIPTS / 'visualization' / 'plot_classifier_responses.py'
    params:
        n_units = 10,
        executor_start = config.executor_start if config.use_executor else '',
        executor_close = config.executor_close if config.use_executor else ''
    output:
        directory(project_paths.figures \
            / 'classifier_response' \
            / '{model_name}{data_identifier}')
    # log:
    #     project_paths.logs / 'plot_classifier_responses_{model_name}{data_identifier}.log'
    group: "visualization"
    shell:
        """
        {params.executor_start}
        python {input.script:q} \
            --input {input.dataframe:q} \
            --output {output:q} \
            --n_units {params.n_units}
        {params.executor_close}
        """
            # > {log} 2>&1

rule plot_weight_distributions:
    """Visualize model weight distributions.

    This rule analyzes and plots the distribution of weights
    across different layers of the model.

    Input:
        state: Model state dictionary
        script: Plotting script
    
    Output:
        Weight distribution plot
    """
    input:
        state = project_paths.models \
            / '{model_name}' \
            / '{model_name}{data_identifier}_{status}.pt',
        script = SCRIPTS / 'visualization' / 'plot_weight_distributions.py'
    params:
        executor_start = config.executor_start if config.use_executor else '',
        executor_close = config.executor_close if config.use_executor else ''
    output:
        plot = project_paths.figures \
            / 'weight_distributions' \
            / '{model_name}{data_identifier}_{status}_weights.{format}'
    # log:
    #     project_paths.logs / 'plot_weight_distributions_{model_name}{data_identifier}_{status}_{format}.log'
    group: "visualization"
    shell:
        """
        {params.executor_start}
        python {input.script:q} \
            --input {input.state:q} \
            --output {output.plot:q} \
            --format {wildcards.format} 
        {params.executor_close}
        """
            # > {log} 2>&1

rule plot_experiment_outputs:
    """Visualize experiment results.

    This rule creates comprehensive visualizations of
    experiment outputs and comparisons.

    Input:
        test_outputs: Experiment results
        script: Visualization script
    
    Output:
        Experiment visualization
    """
    input:
        test_outputs = expand(project_paths.reports \
            / '{{model_name}}' \
            / '{{model_name}}:{{category}}={category_value}{args}_{{seed}}_{{data_name}}_{{status}}_{data_loader}{data_args}_{{data_group}}test_outputs.csv',
            category_value = lambda w: config.model_args[w.category],
            args = lambda w: args_product(dict_poped(config.model_args, w.category), prefix=','),
            data_loader = lambda w: config.experiment_config[w.experiment]['data_loader'],
            data_args = lambda w: args_product(config.experiment_config[w.experiment]['data_args']),
        ),
        script = SCRIPTS / 'visualization' / 'plot_experiment_outputs.py'
    params:
        parameter = lambda w: config.experiment_config[w.experiment]['parameter'],
        executor_start = config.executor_start if config.use_executor else '',
        executor_close = config.executor_close if config.use_executor else ''
    output:
        plot = project_paths.figures / '{experiment}' / '{experiment}_{model_name}:{category}=*_{seed}_{data_name}_{status}_{data_group}' / 'experiment_outputs_label{label_target}.{format}'
    # log:
    #     project_paths.logs / 'plot_experiment_outputs_{experiment}_{model_name}:{category}=*_{seed}_{data_name}_{status}_{data_group}_{label_target}_{format}.log'
    group: "visualization"
    shell:
        """
        {params.executor_start}
        python {input.script:q} \
            --test_outputs {input.test_outputs:q} \
            --output {output.plot:q} \
            --parameter {params.parameter} \
            --category {wildcards.category} \
            --format {params.format} \
            --style {params.style} \
        {params.executor_close}
        """
            # > {log} 2>&1

checkpoint plot_adaption:
    """Analyze and visualize model adaptation.

    This rule generates visualizations of how models
    adapt to different conditions over time.

    Input:
        responses: Model responses
        test_outputs: Test results
        script: Analysis script
    
    Output:
        Adaptation analysis plots
    """
    input:
        responses = expand(project_paths.models \
            / '{{model_name}}' \
            / '{{model_name}}:{{args1}}{{category}}={category_value}{{args2}}_{{seed}}_{{data_name}}_{{status}}_{data_loader}{data_args}_{{data_group}}_test_responses.pt',
            category_value = lambda w: config.experiment_config['categories'][w.category],
            data_loader = lambda w: config.experiment_config[w.experiment]['data_loader'],
            data_args = lambda w: args_product(config.experiment_config[w.experiment]['data_args']),
        ),
        test_outputs = expand(project_paths.reports \
            / '{{model_name}}' \
            / '{{model_name}}:{{args1}}{{category}}={category_value}{{args2}}_{{seed}}_{{data_name}}_{{status}}_{data_loader}{data_args}_{{data_group}}_test_outputs.csv',
            category_value = lambda w: config.experiment_config['categories'][w.category],
            data_loader = lambda w: config.experiment_config[w.experiment]['data_loader'],
            data_args = lambda w: args_product(config.experiment_config[w.experiment]['data_args']),
        ),
        script = SCRIPTS / 'visualization' / 'plot_{plot}.py'
    params:
        measures = ['power', 'peak_height', 'peak_time'],
        parameter = lambda w: config.experiment_config[w.experiment]['parameter'],
        executor_start = config.executor_start if config.use_executor else '',
        executor_close = config.executor_close if config.use_executor else ''
    output:
        flag = project_paths.figures / '{experiment}' / '{experiment}_{model_name}:{args1}{category}=*{args2}_{seed}_{data_name}_{status}_{data_group}' / '{plot}.flag'
    # log:
    #     project_paths.logs / 'plot_adaption_{experiment}_{model_name}:{args1}{category}=*{args2}_{seed}_{data_name}_{status}_{data_group}_{plot}.log'
    group: "visualization"
    shell:
        """
        {params.executor_start}
        python {input.script:q} \
            --responses {input.responses:q} \
            --test_outputs {input.test_outputs:q} \
            --output {output.flag:q} \
            --parameter {params.parameter} \
            --category {wildcards.category} \
            --measures {params.measures} \
            --format {wildcards.format} 
        {params.executor_close}
        """
            # > {log} 2>&1

rule plot_experiments:
    """Collect and organize experiment visualizations.

    This rule aggregates visualizations from multiple
    experiments for comparison and analysis.
    """
    input:
        expand(project_paths.figures / '{experiment}' / '{experiment}_{model_name}:{args1}{category}=*{args2}_{seed}_{data_name}_{status}_{data_group}' / 'adaption.flag',
            experiment = config.experiment,
            model_name = config.model_name,
            category = list(config.model_args.keys())[0],
            args1 = "tsteps=20+",
            args2 = args_product(dict_poped(config.model_args, [list(config.model_args.keys())[0], "tsteps"]), prefix='+'),
            seed = config.seed,
            data_name = config.data_name,
            status = config.status,
            data_group = config.data_group,
        )
    group: "visualization"

rule plot_experiments_on_models:
    """Generate comparative visualizations across models.

    This rule creates visualizations that compare
    experiment results across different model architectures.
    """
    input:
        expand(project_paths.figures / '{experiment}' / '{experiment}_{model_name}{{model_args}}_{seed}_{data_name}_{status}_{data_group}' / 'adaption.flag',
            experiment = config.experiment,
            model_name = config.model_name,
            seed = config.seed,
            data_name = config.data_name,
            status = config.status,
            data_group = config.data_group,
        )
    output:
        temp(project_paths.figures / 'plot_experiments_on_models{model_args}.done')
    # log:
    #     project_paths.logs / 'plot_experiments_on_models{model_args}.log'
    shell:
        """
        touch {output:q} 
        """
        # > {log} 2>&1

# Log workflow initialization
logger.info("Visualization workflow initialized")
logger.info(f"Figure directory: {project_paths.figures}")