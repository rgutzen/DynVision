
rule init_model:
    input:
        script = SCRIPTS / 'runtime' / 'init_model.py',
        dataset = project_paths.data.interim \
            / '{data_name}' \
            / 'train_all' \
            / 'folder.link',
        # dataset = project_paths.data.processed \
        #     / '{data_name}' \
        #     / 'train_all' \
        #     / 'val.beton',
    params:
        model_arguments = lambda w: parse_arguments(w, 'model_args'),
        init_with_pretrained = config.init_with_pretrained,
    resources:
        gpu=0,
        cpu=2,
    output:
        project_paths.models \
            / '{model_name}' \
            / '{model_name}{model_args}_{seed}_{data_name}_init.pt'
    shell:
        """
        python {input.script:q} \
            --model_name {wildcards.model_name} \
            --dataset {input.dataset:q} \
            --data_name {wildcards.data_name} \
            --seed {wildcards.seed} \
            --output {output:q} \
            --init_with_pretrained {params.init_with_pretrained} \
            {params.model_arguments}
        """
    

rule init_pretrained_model:
    input:
        script = SCRIPTS / 'runtime' / 'init_model.py',
    output:
        project_paths.models \
            / '{model_name}' \
            / '{model_name}_0000_imagenet_trained.pt'
    shell:
        """
        python {input.script:q} \
            --model_name {wildcards.model_name} \
            --output {output:q}
        """


rule train_model:
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
        script = SCRIPTS / 'runtime' / 'train_model.py',
    params:
        epochs = config.epochs,
        batch_size = config.batch_size if project_paths.iam_on_cluster() else 3,
        model_arguments = lambda w: parse_arguments(w, 'model_args'),
        learning_rate = config.learning_rate,
        loss = config.loss,
        loss_config = [config.loss_configs[l] for l in list(config.loss)],
        resolution = lambda w: config.data_resolution[w.data_name],
        check_val_every_n_epoch = config.check_val_every_n_epoch,
        accumulate_grad_batches = config.accumulate_grad_batches,
        precision = config.precision,
        store_responses = config.store_val_responses,
        profiler = config.profiler,
        enable_progress_bar = not project_paths.iam_on_cluster(),
    output:
        model_state = project_paths.models \
            / '{model_name}' \
            / '{model_name}{model_args}_{seed}_{data_name}_trained.pt',
    shell:
        """
        python {input.script:q} \
            --input_model_state {input.model_state:q} \
            --model_name {wildcards.model_name} \
            --dataset_train {input.dataset_train:q} \
            --dataset_val {input.dataset_val:q} \
            --data_transform {wildcards.data_name} \
            --output_model_state {output.model_state:q} \
            --learning_rate {params.learning_rate} \
            --epochs {params.epochs} \
            --batch_size {params.batch_size} \
            --seed {wildcards.seed} \
            --check_val_every_n_epoch {params.check_val_every_n_epoch} \
            --accumulate_grad_batches {params.accumulate_grad_batches} \
            --resolution {params.resolution} \
            --precision {params.precision} \
            --profiler {params.profiler} \
            --store_responses {params.store_responses} \
            --enable_progress_bar {params.enable_progress_bar} \
            --loss {params.loss} \
            --loss_config {params.loss_config:q} \
            {params.model_arguments}
        """

rule test_model:
    input:
        model_state = project_paths.models \
            / '{model_name}' \
            / '{model_name}{model_args}_{seed}_{data_name}_{status}.pt',
        dataset = project_paths.data.interim \
            / '{data_name}' 
            / 'test_{data_group}' \
            / 'folder.link',
        script = SCRIPTS / 'runtime' / 'test_model.py',
    params:
        batch_size = config.batch_size if project_paths.iam_on_cluster() else 3,
        data_group = lambda w: f"{w.data_name}_{w.data_group}",
        model_arguments = lambda w: parse_arguments(w, 'model_args'),
        data_arguments = lambda w: parse_arguments(w, 'data_args'),
        loss = config.loss,
        loss_config = [config.loss_configs[l] for l in list(config.loss)],
        store_responses = config.store_test_responses,
        benchmark = config.benchmark,
    resources:
        gpu=0,
        cpu=4,
    output:
        responses = project_paths.models \
            / '{model_name}' \
            / '{model_name}{model_args}_{seed}_{data_name}_{status}_{data_loader}{data_args}_{data_group}_test_responses.pt',
        results = project_paths.reports \
            / '{model_name}' \
            / '{model_name}{model_args}_{seed}_{data_name}_{status}_{data_loader}{data_args}_{data_group}_test_outputs.csv',
    shell:
        """
        python {input.script:q} \
            --input_model_state {input.model_state:q} \
            --model_name {wildcards.model_name} \
            --dataset {input.dataset:q} \
            --data_loader {wildcards.data_loader} \
            --output_results {output.results:q} \
            --output_responses {output.responses:q} \
            --data_transform {wildcards.data_name} \
            --target_transform {params.data_group} \
            --loss {params.loss} \
            --loss_config {params.loss_config:q} \
            --batch_size {params.batch_size} \
            --seed {wildcards.seed} \
            --store_responses {params.store_responses} \
            --benchmark {params.benchmark} \
            {params.model_arguments} \
            {params.data_arguments}
        """


