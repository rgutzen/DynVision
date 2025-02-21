
rule plot_confusion_matrix:
    input:
        test_results = project_paths.reports / '{path}_{data_name}_testing_results.csv',
        dataset = project_paths.data.interim / '{data_name}_test' / 'folder.link',
        script = SCRIPTS / 'visualization' / 'plot_confusion_matrix.py',
    params:
        palette = 'cividis',
    output:
        project_paths.figures / '{path}_{data_name}_confusion.png'
    shell:
        """
        python {input.script:q} \
            --input {input.test_results:q} \
            --output {output:q} \
            --dataset {input.dataset} \
            --palette {params.palette}
        """

rule plot_classifier_responses:
    input:
        dataframe = project_paths.reports \
            / '{model_name}' \
            / '{model_name}{data_identifier}_test_outputs.csv',
        script = SCRIPTS / 'visualization' / 'plot_classifier_responses.py',
    output:
        directory(project_paths.figures \
            / 'classifier_response' \
            / '{model_name}{data_identifier}')
    shell:
        """
        python {input.script:q} \
            --input {input.dataframe:q} \
            --output {output:q} \
        """


rule plot_weight_distributions:
    input:
        state = project_paths.models \
            / '{model_name}' \
            / '{model_name}{data_identifier}_{status}.pt',
        script = SCRIPTS / 'visualization' / 'plot_weight_distributions.py',
    output:
        project_paths.figures \
            / 'weight_distributions' \
            / '{model_name}{data_identifier}_{status}_weights.png'
    shell:
        """
        python {input.script:q} \
            --input {input.state:q} \
            --output {output:q} \
        """

rule plot_experiments:
    input:
        expand(project_paths.figures / '{experiment}' / '{experiment}_{model_name}:{category}=*_{seed}_{data_name}_{status}_{data_group}' / 'adaption.flag',
        experiment = config.experiment,
        model_name = config.model_name,
        category = list(config.model_args.keys())[0],
        args = args_product(dict_poped(config.model_args, list(config.model_args.keys())[0]), prefix=','),
        seed = config.seed,
        data_name = config.data_name,
        status = config.status,
        data_group = config.data_group,
        )
        

rule plot_experiment_outputs:
    input:
        test_outputs = expand(project_paths.reports \
            / '{{model_name}}' \
            / '{{model_name}}:{{category}}={category_value}{args}_{{seed}}_{{data_name}}_{{status}}_{data_loader}{data_args}_{{data_group}}test_outputs.csv',
            category_value = lambda w: config.model_args[w.category],
            args = lambda w: args_product(dict_poped(config.model_args, w.category), prefix=','),
            data_loader = lambda w: config.experiment_config[w.experiment]['data_loader'],
            data_args = lambda w: args_product(config.experiment_config[w.experiment]['data_args']),
        ),
        script = SCRIPTS / 'visualization' / 'plot_experiment_outputs.py',
    params:
        parameter = lambda w: config.experiment_config[w.experiment]['parameter'],
    output:
        project_paths.figures / '{experiment}' / '{experiment}_{model_name}:{category}=*_{seed}_{data_name}_{status}_{data_group}' / 'experiment_outputs_label{label_taget}.png'
    shell:
        """
        python {input.script:q} \
            --test_outputs {input.test_outputs:q} \
            --output {output:q} \
            --parameter {params.parameter} \
            --category {wildcards.category} \
        """


checkpoint plot_adaption:
    input:
        responses = expand(project_paths.models \
            / '{{model_name}}' \
            / '{{model_name}}:{{args1}}{{category}}={category_value}{{args2}}_{{seed}}_{{data_name}}_{{status}}_{data_loader}{data_args}_{{data_group}}_test_responses.pt',
            # category_value = lambda w: config.model_args[w.category],
            category_value = lambda w: config.experiment_config['categories'][w.category],
            # args = lambda w: args_product(dict_poped(config.model_args, w.category), prefix=','),
            data_loader = lambda w: config.experiment_config[w.experiment]['data_loader'],
            data_args = lambda w: args_product(config.experiment_config[w.experiment]['data_args']),
        ),
        test_outputs = expand(project_paths.reports \
            / '{{model_name}}' \
            / '{{model_name}}:{{args1}}{{category}}={category_value}{{args2}}_{{seed}}_{{data_name}}_{{status}}_{data_loader}{data_args}_{{data_group}}_test_outputs.csv',
            # category_value = lambda w: config.model_args[w.category],
            category_value = lambda w: config.experiment_config['categories'][w.category],
            # args = lambda w: args_product(dict_poped(config.model_args, w.category), prefix=','),
            data_loader = lambda w: config.experiment_config[w.experiment]['data_loader'],
            data_args = lambda w: args_product(config.experiment_config[w.experiment]['data_args']),
        ),
        script = SCRIPTS / 'visualization' / 'plot_{plot}.py',
    params:
        measures = ['power', 'peak_height', 'peak_time'],
        parameter = lambda w: config.experiment_config[w.experiment]['parameter'],
    output:
        project_paths.figures / '{experiment}' / '{experiment}_{model_name}:{args1}{category}=*{args2}_{seed}_{data_name}_{status}_{data_group}' / '{plot}.flag'
    shell:
        """
        python {input.script:q} \
            --responses {input.responses:q} \
            --test_outputs {input.test_outputs:q} \
            --output {output:q} \
            --parameter {params.parameter} \
            --category {wildcards.category} \
            --measures {params.measures}
        """

rule plot_adaptions:
    input:
        expand(project_paths.figures / '{experiment}' / '{experiment}_{model_name}:{args1}{category}=*{args2}_{seed}_{data_name}_{status}_{data_group}' / 'adaption.flag',
        experiment = ['duration', 'contrast', 'intedvml'],
        model_name = "FourLayerCNN",
        args1 = "tsteps=20,",
        category = "rctype",
        args2 = "",
        seed = "0039",
        data_name = "cifar100",
        status = "trained",
        data_group = "invertebrates",
        )