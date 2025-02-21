# ruleorder: get_mnist > symlink_data_groups > symlink_data_subsets

rule get_data:
    input:
        script = SCRIPTS / 'data' / 'get_data.py',
    params:
        raw_data_path = lambda w: project_paths.data.raw / f'{w.data_name}',
        output = lambda w: project_paths.data.raw \
            / f'{w.data_name}' \
            / f'{w.data_subset}' \
            / 'folder.link',
        ext = 'png',
        data_name = lambda w: ''.join([c.upper() if c.isalpha() else c for c in w.data_name])
    output:
        flag = directory(project_paths.data.raw \
            / '{data_name}' \
            / '{data_subset}' )
            # / '{category}')
    wildcard_constraints:
        data_name = r"cifar10|cifar100|mnist"  # extend as needed
    shell:
        """
        python {input.script:q} \
            --output {params.output:q} \
            --data_name {params.data_name} \
            --raw_data_path {params.raw_data_path:q} \
            --subset {wildcards.data_subset} \
            --ext {params.ext} 
        """


rule symlink_data_subsets:
    input:
        get_data_location
    params:
        parent = lambda wildcards, output: Path(output.flag).parent,
        source = lambda wildcards, output: Path(output.flag).with_suffix('')
    output:
        flag = project_paths.data.interim \
            / '{data_name}' \
            / '{data_subset}_{data_group}' \
            / '{category}.link'
    shell:
        """
        mkdir -p {params.parent:q}
        ln -sf {input:q} {params.source:q}
        touch {output.flag:q}
        """

rule symlink_data_groups:
    input:
        lambda wildcards: expand(project_paths.data.interim \
            / '{{data_name}}' \
            / '{{data_subset}}_{{data_group}}' \
            / '{category}.link',
            category=get_category(wildcards.data_name, wildcards.data_group)
            )
    output:
        flag = project_paths.data.interim \
            / '{data_name}' \
            / '{data_subset}_{data_group}' \
            / 'folder.link'
    shell:
        """
        touch {output.flag:q}
        """

rule build_ffcv_datasets:
    input:
        script = SCRIPTS / 'data' / 'ffcv_datasets.py',
        data = project_paths.data.interim / '{data_name}' / 'train_all' / 'folder.link',
    params:
        train_ratio = config.train_ratio,
        max_resolution = lambda w: config.data_resolution[w.data_name],
    output:
        train = project_paths.data.processed / '{data_name}' / 'train_all' / 'train.beton',
        val = project_paths.data.processed / '{data_name}' / 'train_all' / 'val.beton'
    shell:
        """
        python {input.script:q} \
            --input {input.data:q} \
            --output_train {output.train:q} \
            --output_val {output.val:q} \
            --train_ratio {params.train_ratio} \
            --data_name {wildcards.data_name} \
            --max_resolution {params.max_resolution}
        """