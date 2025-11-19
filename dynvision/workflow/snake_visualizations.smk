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
"""

logger = logging.getLogger('workflow.visualizations')


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
        dataset_ready = project_paths.data.interim / '{data_name}' / 'test_all.ready',
        script = SCRIPTS / 'visualization' / 'plot_confusion_matrix.py'
    params:
        palette = 'cividis',
        dataset_path = lambda w: project_paths.data.interim / w.data_name / 'test_all',
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
            use_executor=get_param('use_executor', False)(w)
        ),
    output:
        plot = project_paths.figures / '{path}_{data_name}_confusion.{format}'
    # group: "visualization"
    shell:
        """
        {params.execution_cmd} \
            --input {input.test_results:q} \
            --output {output.plot:q} \
            --dataset_path {params.dataset_path:q} \
            --palette {params.palette} \
            --format {wildcards.format} \
        """

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
            / '{data_loader}' \
            / '{model_name}{data_identifier}' \
            / 'test_outputs.csv',
        script = SCRIPTS / 'visualization' / 'plot_classifier_responses.py'
    params:
        n_units = 10,
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
            use_executor=get_param('use_executor', False)(w)
        ),
    output:
        directory(project_paths.figures \
            / 'classifier_response' \
            / '{model_name}{data_identifier}')
    # group: "visualization"
    shell:
        """
        {params.execution_cmd} \
            --input {input.dataframe:q} \
            --output {output:q} \
            --n_units {params.n_units}
        """

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
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
            use_executor=get_param('use_executor', False)(w)
        ),
    output:
        plot = project_paths.figures \
            / 'weight_distributions' \
            / '{model_name}{data_identifier}_{status}_weights.{format}'
    # group: "visualization"
    shell:
        """
        {params.execution_cmd} \
            --input {input.state:q} \
            --output {output.plot:q} 
            --format {wildcards.format} 
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
        measures = ['response_avg', 'response_std', 'label_confidence', 'guess_confidence', 'first_label_confidence', 'classifier_top10', 'accuracy_top3', 'accuracy_top5'], # 'spatial_variance', 'feature_variance', 
        parameter = lambda w: config.experiment_config[w.experiment]['parameter'],
        category = lambda w: w.category_str.strip('=*'),
        additional_parameters = 'epoch',
        batch_size = 1,
        remove_input_responses = True,
        sample_resolution = 'sample',  # 'sample' or 'class'
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
            use_executor=get_param('use_executor', False)(w)
        ),
    priority: 100,
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
            --remove_input_responses {params.remove_input_responses}
        """

rule process_wandb_data:
    input:
        data = Path('{path}.csv'),
        script = SCRIPTS / 'visualization' / 'process_wandb_data.py'
    params:
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
            use_executor=get_param('use_executor', False)(w)
        ),
    output:
        Path('{path}_summary.csv'),
    shell:
        """
        {params.execution_cmd} \
            --data {input.data:q} \
            --output {output:q}
        """

rule plot_performance:
    input:
        data = expand(project_paths.reports / '{experiment}' / '{experiment}_{{model_name}}:{{args1}}{{category_str}}{{args2}}_{seeds}_{{data_name}}_{{status}}_{{data_group}}' / 'test_data.csv',
            experiment = lambda w: ['uniformnoise', 'poissonnoise', 'gaussiannoise', 'gaussiancorrnoise'] if w.experiment == 'noise' else w.experiment,
            seeds = lambda w: w.seeds.split('.'),
            ),
        dataffonly = expand(project_paths.reports / '{experiment}ffonly' / '{experiment}ffonly_{{model_name}}:{{args1}}{{category_str}}{{args2}}_{seeds}_{{data_name}}_{{status}}_{{data_group}}' / 'test_data.csv',
            experiment = lambda w: ['uniformnoise', 'poissonnoise', 'gaussiannoise', 'gaussiancorrnoise'] if w.experiment == 'noise' else w.experiment,
            seeds = lambda w: w.seeds.split('.'),
            ),
        script = SCRIPTS / 'visualization' / 'plot_performance.py'
    params:
        row = 'experiment',
        subplot = 'parameter',
        hue = 'category',
        parameter = lambda w: config.experiment_config[w.experiment]['parameter'],
        category = lambda w: w.category_str.strip('=*'),
        experiment = lambda w: ['uniformnoise', 'poissonnoise', 'gaussiannoise', 'gaussiancorrnoise'] if w.experiment == 'noise' else w.experiment,
        confidence_measure = getattr(config, 'plot_confidence_measure', "first_label_confidence"),
        dt = getattr(config, 'dt', 2),
        palette = lambda w: json.dumps(config.palette),
        naming = lambda w: json.dumps(config.naming),
        ordering = lambda w: json.dumps(config.ordering),
        subplot_filter = [], #[0.1, 0.5, 1.0], #lambda w: config.experiment_config[w.experiment].get('subplot_filter', []),
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
            use_executor=get_param('use_executor', False)(w)
        ),
    output:
        project_paths.figures / '{experiment}' / '{experiment}_{model_name}:{args1}{category_str}{args2}_{seeds}_{data_name}_{status}_{data_group}' / 'performance.png',
    shell:
        """
        {params.execution_cmd} \
            --data {input.data:q} \
            --data-ffonly {input.dataffonly:q} \
            --output {output:q} \
            --row {params.row} \
            --subplot {params.subplot} \
            --hue {params.hue} \
            --parameter-key {params.parameter} \
            --category-key {params.category} \
            --experiment {params.experiment} \
            --confidence-measure {params.confidence_measure} \
            --dt {params.dt} \
            --palette {params.palette:q} \
            --naming {params.naming:q} \
            --ordering {params.ordering:q} \
            --subplot-filter {params.subplot_filter}
        """

rule plot_training:
    input:
        test_data = expand(project_paths.reports / '{{experiment}}' / '{{experiment}}_{{model_name}}:{{args1}}{{category_str}}{{args2}}_{seeds}_{{data_name}}_{{status}}_{{data_group}}' / 'test_data.csv',
            seeds = lambda w: w.seeds.split('.'),
            ),
        script = SCRIPTS / 'visualization' / 'plot_training.py'
    params:
        accuracy_csv = lambda w: project_paths.reports \
            / 'wandb' \
            / f'{w.model_name}:{w.args1}{w.category_str}{w.args2}_{w.seeds}_{w.data_name}_{w.status}_accuracy.csv',
        memory_csv = lambda w: project_paths.reports \
            / 'wandb' \
            / f'{w.model_name}:{w.args1}{w.category_str}{w.args2}_{w.seeds}_{w.data_name}_{w.status}_gpu_mem_alloc.csv',
        epoch_csv = lambda w: project_paths.reports \
            / 'wandb' \
            / f'{w.model_name}:{w.args1}{w.category_str}{w.args2}_{w.seeds}_{w.data_name}_{w.status}_epoch.csv',
        energy_csv= lambda w: project_paths.reports \
            / 'wandb' \
            / f'{w.model_name}:{w.args1}{w.category_str}{w.args2}_{w.seeds}_{w.data_name}_{w.status}_energyloss.csv',
        cross_entropy_csv= lambda w: project_paths.reports \
            / 'wandb' \
            / f'{w.model_name}:{w.args1}{w.category_str}{w.args2}_{w.seeds}_{w.data_name}_{w.status}_crossentropyloss.csv',
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
            use_executor=get_param('use_executor', False)(w)
        ),
        palette = lambda w: json.dumps(config.palette),
        naming = lambda w: json.dumps(config.naming),
        ordering = lambda w: json.dumps(config.ordering),
        dt = getattr(config, 'dt', 2),
    output:
        project_paths.figures / '{experiment}' / '{experiment}_{model_name}:{args1}{category_str}{args2}_{seeds}_{data_name}_{status}_{data_group}' / 'training.png',
    shell:
        """
        {params.execution_cmd} \
            --test_data {input.test_data:q} \
            --accuracy_csv {params.accuracy_csv:q} \
            --memory_csv {params.memory_csv:q} \
            --epoch_csv {params.epoch_csv:q} \
            --energy_csv {params.energy_csv:q} \
            --cross_entropy_csv {params.cross_entropy_csv:q} \
            --palette {params.palette:q} \
            --naming {params.naming:q} \
            --ordering {params.ordering:q} \
            --category {wildcards.category} \
            --dt {params.dt} \
            --output {output:q} 
        """


rule plot_dynamics:
    input:
        data = expand(project_paths.reports / '{{experiment}}' / '{{experiment}}_{{model_name}}:{{args1}}{{category}}=*{{args2}}_{seeds}_{{data_name}}_{{status}}_{{data_group}}' / 'test_data.csv',
            seeds = lambda w: w.seeds.split('.')
            ),
        script = SCRIPTS / 'visualization' / 'plot_dynamics.py'
    params:
        parameter = lambda w: config.experiment_config[w.experiment]['parameter'],
        # focus_layer = 'IT',
        dt = getattr(config, 'dt', 2),
        palette = lambda w: json.dumps(config.palette),
        naming = lambda w: json.dumps(config.naming),
        ordering = lambda w: json.dumps(config.ordering),
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
            use_executor=get_param('use_executor', False)(w)
        ),
    output:
        project_paths.figures / '{experiment}' / '{experiment}_{model_name}:{args1}{category}=*{args2}_{seeds}_{data_name}_{status}_{data_group}' / 'dynamics_{focus_layer}.png',
    # group: "visualization"
    shell:
        """
        {params.execution_cmd} \
            --data {input.data:q} \
            --output {output:q} \
            --parameter {params.parameter} \
            --experiment {wildcards.experiment} \
            --category {wildcards.category} \
            --focus-layer {wildcards.focus_layer} \
            --dt {params.dt} \
            --palette {params.palette:q} \
            --naming {params.naming:q} \
            --ordering {params.ordering:q}
        """


rule plot_responses:
    input:
        data = expand(project_paths.reports / '{{experiment}}' / '{{experiment}}_{{model_name}}:{{args1}}{{category_str}}{{args2}}_{seeds}_{{data_name}}_{{status}}_{{data_group}}' / 'test_data.csv',
            seeds = lambda w: w.seeds.split('.')
            ),
        script = SCRIPTS / 'visualization' / 'plot_responses.py'
    params:
        column = getattr(config, 'column', 'parameter'),  # first_label_index
        subplot = getattr(config, 'subplot', 'layers'),  # classifier_topk
        hue = getattr(config, 'hue', 'category'),
        parameter = lambda w: config.experiment_config[w.experiment]['parameter'],
        category = lambda w: w.category_str.strip('=*'),
        dt = getattr(config, 'dt', 2),
        confidence_measure = getattr(config, 'plot_confidence_measure', "first_label_confidence"),
        accuracy_measure = getattr(config, 'plot_accuracy_measure', "accuracy"),
        palette = lambda w: json.dumps(config.palette),
        naming = lambda w: json.dumps(config.naming),
        ordering = lambda w: json.dumps(config.ordering),
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
            use_executor=get_param('use_executor', False)(w)
        ),
    output:
        project_paths.figures / '{experiment}' / '{experiment}_{model_name}:{args1}{category_str}{args2}_{seeds}_{data_name}_{status}_{data_group}' / 'responses.png',
    shell:
        """
        {params.execution_cmd} \
            --data {input.data:q} \
            --output {output:q} \
            --column {params.column} \
            --subplot {params.subplot} \
            --hue {params.hue} \
            --parameter-key {params.parameter} \
            --category-key {params.category} \
            --experiment {wildcards.experiment} \
            --confidence-measure {params.confidence_measure} \
            --accuracy-measure {params.accuracy_measure} \
            --dt {params.dt} \
            --palette {params.palette:q} \
            --naming {params.naming:q} \
            --ordering {params.ordering:q}
        """

rule plot_timeparams_tripytch:
    input:
        data1 = expand(project_paths.reports / '{{experiment}}' / '{{experiment}}_{{model_name}}:{{args1}}tau=*+tff=0+trc=6+tsk=0{{args2}}_{seeds}_{{data_name}}_{{status}}_{{data_group}}' / 'test_data.csv',
            seeds = lambda w: w.seeds.split('.')
            ),
        data2 = expand(project_paths.reports / '{{experiment}}' / '{{experiment}}_{{model_name}}:{{args1}}tau=5+tff=0+trc=*+tsk=0{{args2}}_{seeds}_{{data_name}}_{{status}}_{{data_group}}' / 'test_data.csv',
            seeds = lambda w: w.seeds.split('.')
            ),
        data3 = expand(project_paths.reports / '{{experiment}}' / '{{experiment}}_{{model_name}}:{{args1}}tau=5+tff=0+trc=6+tsk=*{{args2}}_{seeds}_{{data_name}}_{{status}}_{{data_group}}' / 'test_data.csv',
            seeds = lambda w: w.seeds.split('.')
            ),
        script = SCRIPTS / 'visualization' / 'plot_response_tripytch.py'
    params:
        accuracy1 = lambda w: project_paths.reports / 'wandb' / f'{w.model_name}:{w.args1}tau=*+tff=0+trc=6+tsk=0{w.args2}_{w.seeds}_{w.data_name}_{w.status}_accuracy.csv',
        accuracy2 = lambda w: project_paths.reports / 'wandb' / f'{w.model_name}:{w.args1}tau=5+tff=0+trc=*+tsk=0{w.args2}_{w.seeds}_{w.data_name}_{w.status}_accuracy.csv',
        accuracy3 = lambda w: project_paths.reports / 'wandb' / f'{w.model_name}:{w.args1}tau=5+tff=0+trc=6+tsk=*{w.args2}_{w.seeds}_{w.data_name}_{w.status}_accuracy.csv',
        parameter = lambda w: config.experiment_config[w.experiment]['parameter'],
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
            use_executor=get_param('use_executor', False)(w)
        ),
        category = ' '.join(['tau', 'trc', 'tsk']),
        dt = getattr(config, 'dt', 2),
        palette = lambda w: json.dumps(config.palette),
        naming = lambda w: json.dumps(config.naming),
        ordering = lambda w: json.dumps(config.ordering),
    output:
        project_paths.figures / '{experiment}' / '{experiment}_{model_name}:{args1}tau=*+tff=0+trc=*+tsk=*{args2}_{seeds}_{data_name}_{status}_{data_group}' / 'response_tripytch.png',
    # group: "visualization"
    shell:
        """
        {params.execution_cmd} \
            --data {input.data1:q} \
            --data2 {input.data2:q} \
            --data3 {input.data3:q} \
            --accuracy1 {params.accuracy1:q} \
            --accuracy2 {params.accuracy2:q} \
            --accuracy3 {params.accuracy3:q} \
            --output {output:q} \
            --parameter {params.parameter} \
            --experiment {wildcards.experiment} \
            --category {params.category:q} \
            --dt {params.dt} \
            --palette {params.palette:q} \
            --naming {params.naming:q} \
            --ordering {params.ordering:q}
        """

rule plot_timestep_tripytch:
    input:
        data1 = expand(project_paths.reports / '{{experiment}}' / '{{experiment}}_{{model_name}}:tsteps=*{{args1}}lossrt=4_{seeds}_{{data_name}}_{{status}}_{{data_group}}' / 'test_data.csv',
            seeds = lambda w: w.seeds.split('.')
            ),
        data2 = expand(project_paths.reports / '{{experiment}}' / '{{experiment}}_{{model_name}}:tsteps=20{{args1}}skip=true+lossrt=*_{seeds}_{{data_name}}_{{status}}_{{data_group}}' / 'test_data.csv',
            seeds = lambda w: w.seeds.split('.')
            ),
        data3 = expand(project_paths.reports / '{{experiment}}' / '{{experiment}}_{{model_name}}:tsteps=20{{args1}}lossrt=4+idle=*_{seeds}_{{data_name}}_{{status}}_{{data_group}}' / 'test_data.csv',
            seeds = lambda w: w.seeds.split('.')
            ),
        script = SCRIPTS / 'visualization' / 'plot_response_tripytch.py'
    params:
        accuracy1 = lambda w: project_paths.reports / 'wandb' / f'{w.model_name}:tsteps=*{w.args1}lossrt=4_{w.seeds}_{w.data_name}_{w.status}_accuracy.csv',
        accuracy2 = lambda w: project_paths.reports / 'wandb' / f'{w.model_name}:tsteps=20{w.args1}skip=true+lossrt=*_{w.seeds}_{w.data_name}_{w.status}_accuracy.csv',
        accuracy3 = lambda w: project_paths.reports / 'wandb' / f'{w.model_name}:tsteps=20{w.args1}lossrt=4+idle=*_{w.seeds}_{w.data_name}_{w.status}_accuracy.csv',
        parameter = lambda w: config.experiment_config[w.experiment]['parameter'],
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
            use_executor=get_param('use_executor', False)(w)
        ),
        category = ' '.join(['tsteps', 'lossrt', 'idle']),
        dt = getattr(config, 'dt', 2),
        palette = lambda w: json.dumps(config.palette),
        naming = lambda w: json.dumps(config.naming),
        ordering = lambda w: json.dumps(config.ordering),
    output:
        project_paths.figures / '{experiment}' / '{experiment}_{model_name}:tsteps=*{args1}lossrt=*+idle=*_{seeds}_{data_name}_{status}_{data_group}' / 'response_tripytch.png',
    # group: "visualization"
    shell:
        """
        {params.execution_cmd} \
            --data {input.data1:q} \
            --data2 {input.data2:q} \
            --data3 {input.data3:q} \
            --accuracy1 {params.accuracy1:q} \
            --accuracy2 {params.accuracy2:q} \
            --accuracy3 {params.accuracy3:q} \
            --output {output:q} \
            --parameter {params.parameter} \
            --experiment {wildcards.experiment} \
            --category {params.category:q} \
            --dt {params.dt} \
            --palette {params.palette:q} \
            --naming {params.naming:q} \
            --ordering {params.ordering:q}
        """

rule plot_connection_tripytch:
    input:
        data1 = expand(project_paths.reports / '{{experiment}}' / '{{experiment}}_{{model_name}}:tsteps=20+rctype=full+rctarget=*{{args2}}lossrt=4_{seeds}_{{data_name}}_{{status}}_{{data_group}}' / 'test_data.csv',
            seeds = lambda w: w.seeds.split('.')
            ),
        data2 = expand(project_paths.reports / '{{experiment}}' / '{{experiment}}_{{model_name}}:tsteps=20+rctype=full+rctarget=output{{args2}}skip=*+lossrt=4_{seeds}_{{data_name}}_{{status}}_{{data_group}}' / 'test_data.csv',
            seeds = lambda w: w.seeds.split('.')
            ),
        data3 = expand(project_paths.reports / '{{experiment}}' / '{{experiment}}_{{model_name}}:tsteps=30+rctype=full+rctarget=output{{args2}}tfb=30+feedback=*+lossrt=4_{seeds}_{{data_name}}_{{status}}_{{data_group}}' / 'test_data.csv',
            seeds = lambda w: w.seeds.split('.')
            ),
        script = SCRIPTS / 'visualization' / 'plot_response_tripytch.py'
    params:
        accuracy1 = lambda w: project_paths.reports / 'wandb' / f'{w.model_name}:tsteps=20+rctype=full+rctarget=*{w.args2}lossrt=4_{w.seeds}_{w.data_name}_{w.status}_accuracy.csv',
        accuracy2 = lambda w: project_paths.reports / 'wandb' / f'{w.model_name}:tsteps=20+rctype=full+rctarget=output{w.args2}skip=*+lossrt=4_{w.seeds}_{w.data_name}_{w.status}_accuracy.csv',
        accuracy3 = lambda w: project_paths.reports / 'wandb' / f'{w.model_name}:tsteps=30+rctype=full+rctarget=output{w.args2}tfb=30+feedback=*+lossrt=4_{w.seeds}_{w.data_name}_{w.status}_accuracy.csv',
        parameter = lambda w: config.experiment_config[w.experiment]['parameter'],
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
            use_executor=get_param('use_executor', False)(w)
        ),
        category = ' '.join(['rctarget', 'skip', 'feedback']),
        dt = getattr(config, 'dt', 2),
        outlier_threshold = 10,  # Exclude yscale limits beyond this threshold
        palette = lambda w: json.dumps(config.palette),
        naming = lambda w: json.dumps(config.naming),
        ordering = lambda w: json.dumps(config.ordering),
    output:
        project_paths.figures / '{experiment}' / '{experiment}_{model_name}:rctype=full{args2}rctarget=*+skip=*+feedback=*+lossrt=4_{seeds}_{data_name}_{status}_{data_group}' / 'response_tripytch.png',
    shell:
        """
        {params.execution_cmd} \
            --data {input.data1:q} \
            --data2 {input.data2:q} \
            --data3 {input.data3:q} \
            --accuracy1 {params.accuracy1:q} \
            --accuracy2 {params.accuracy2:q} \
            --accuracy3 {params.accuracy3:q} \
            --output {output:q} \
            --parameter {params.parameter} \
            --experiment {wildcards.experiment} \
            --category {params.category:q} \
            --dt {params.dt} \
            --palette {params.palette:q} \
            --naming {params.naming:q} \
            --ordering {params.ordering:q}
        """
            # --outlier_threshold {params.outlier_threshold} \

rule plot_experiment:
    input:
        data = project_paths.reports / '{experiment}' / '{experiment}_{model_identifier}' / 'layer_power_small.csv',
        script = SCRIPTS / 'visualization' / 'plot_experiment.py'
    params:
        parameter = lambda w: config.experiment_config[w.experiment]['parameter'],
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
            use_executor=get_param('use_executor', False)(w)
        ),
    output:
        project_paths.figures / '{experiment}' / '{experiment}_{model_identifier}' / 'experiment.png',
    shell:
        """
        {params.execution_cmd} \
            --data {input.data:q} \
            --output {output:q} \
            --parameter {params.parameter}
        """

rule plot_unrolling:
    input:
        engineering_time_data = project_paths.models \
            / '{model_name}' \
            / '{model_name}:{args1}tff=0+trc=6+tsk=0+tfb=34{args2}+unrolled=false_{seed}_{data_name}_{status}_{data_loader}{data_args}_{data_group}_test_responses.pt',
        biological_time_data = project_paths.models \
            / '{model_name}' \
            / '{model_name}:{args1}tff=10+trc=6+tsk=20+tfb=14{args2}+unrolled=true_{seed}_{data_name}_{status}_{data_loader}{data_args}_{data_group}_test_responses.pt',
        script = SCRIPTS / 'visualization' / 'plot_unrolling.py'
    params:
        t_feedforward = 10 // 2,  # tff / dt
        execution_cmd = lambda w, input: build_execution_command(
            script_path=input.script,
            use_distributed=False,
            use_executor=get_param('use_executor', False)(w)
        ),
    output:
        project_paths.figures / 'unrolling' / 'unrolling_{model_name}:{args1}tff=*{args2}_{seed}_{data_name}_{status}_{data_loader}{data_args}_{data_group}.png',
    shell:
        """
        {params.execution_cmd} \
            --engineering_time_data {input.engineering_time_data:q} \
            --biological_time_data {input.biological_time_data:q} \
            --t_feedforward {params.t_feedforward} \
            --output {output:q} 
        """
