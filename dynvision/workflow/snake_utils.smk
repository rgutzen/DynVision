from types import SimpleNamespace
from collections import defaultdict
from itertools import product
import json
from pathlib import Path
from dynvision.project_paths import project_paths

SCRIPTS = project_paths.scripts_path

configfile: project_paths.scripts.configs / 'config_defaults.yaml'
# the following configs have priority over the default configs:
configfile: project_paths.scripts.configs / 'config_data.yaml'
# configfile: project_paths.scripts.configs / 'config_visualization.yaml'
configfile: project_paths.scripts.configs / 'config_workflow.yaml'
configfile: project_paths.scripts.configs / 'config_experiments.yaml'
config = SimpleNamespace(**config)


wildcard_constraints:
    model_name = '[a-zA-Z0-9]+',
    data_name = '[a-z0-9]+',
    data_subset = '[a-z]+',
    data_group = '[a-z0-9]+',
    data_loader = '[a-zA-Z]+',
    status = '[a-z]+',
    seed = '\d+',
    condition = '(withwave_|nowave_|\s?)',
    category = '(?!folder)[a-z0-9]+',
    model_args = '(:[a-z,=\d\.]+|\s?)',
    data_args = '(:[a-zTF,=\d\.]+|\s?)',
    args = '([a-z,=\d\.]+|\s?)',
    args1 = '([a-z,=\d\.]+,|\s?)',
    args2 = '(,[a-z,=\d\.]+|\s?)',
    parameter = '(contrast|duration|intedvml)',
    experiment = '[a-z]+',
    layer_name = '(layer1|layer2|V1|V2|V4|IT)',

localrules: symlink_data_subsets, symlink_data_groups
ruleorder: symlink_data_groups > symlink_data_subsets

rule checkpoint_to_statedict:
    input:
        checkpoint_dir = project_paths.logs \
            / 'checkpoints' ,
        script = SCRIPTS / 'utils' / 'checkpoint_to_statedict.py'
    output:
        temp(project_paths.models \
            / '{model_identifier}.ckpt2pt')
    shell:
        """
        python {input.script:q} \
            --checkpoint_dir {input.checkpoint_dir:q} \
            --output {output:q}
        """

def get_imagenet_classes(tiny=False):
    index_file = "tinyimagenet_class_index" if tiny else "imagenet_class_index"
    with open(project_paths.references / f"{index_file}.json") as f:
        class_index = json.load(f)  # label: [class, class_name]
        
    imagenet_classes = [v[0] for k,v in class_index.items()]

    return imagenet_classes, class_index

def get_gabordetect_classes():
    # n = config.gabordetect['duration'] // config.gabordetect['time_resolution']
    # m = config.gabordetect['delay'] // config.gabordetect['time_resolution']
    # class_str = lambda p: "".join(["0"] * m + [f"{p}"] * (n-m))
    # class_str_list = [class_str(p) for p in range(5)]

    # workaround since snakemake has problems with long wildcards (#1769)
    return [str(i) for i in range(5)]

def get_category(data_name, data_group):
    if 'imagenet' in data_name:
        imagenet_classes, imagenet_class_index = get_imagenet_classes(tiny=('tiny' in data_name))

        if data_group == 'all':
            return imagenet_classes
        else:
            return [
                imagenet_class_index[str(c)][0] for c in \
                config.data_groups[data_name][data_group]
                ]

    elif data_name == 'cifar10':
        if data_group == 'all':
            return [str(i) for i in range(10)]
        else:
            return config.data_groups[data_name][data_group]

    elif data_name == 'cifar100':
        if data_group == 'all':
            return [str(i) for i in range(100)]
        else:
            return config.data_groups[data_name][data_group]

    elif data_name == 'snakenet':
        if data_group == 'all':
            return ['n01729322', 'n01740131', 'n01744401', 'n01753488', 'n01755581', 'n01756291']
        else:
            return config.data_groups[data_name][data_group]

    elif data_name == 'mnist':
        if data_group == 'all':
            return [str(i) for i in range(10)]
        else:
            return config.data_groups[data_name][data_group]
    
    elif 'gabordetect' in data_name:
        gabordetect_classes = get_gabordetect_classes()
        if data_group == 'all':
            return gabordetect_classes
        else:
            return config.data_groups['gabordetect'][data_group]

    else:
        raise ValueError(f"Unknown data_name: {data_name}")

def get_data_location(wildcards):
    data_name = wildcards.data_name
    data_subset = wildcards.data_subset
    category = wildcards.category

    base_dir = get_data_base_dir(wildcards)

    if (data_name == 'imagenet') and (data_subset == 'test'):
        data_subset = 'val'
    
    data_location = base_dir / data_subset / category
    return data_location

def get_data_base_dir(wildcards):
    data_name = wildcards.data_name

    if data_name in config.mounted_datasets and project_paths.iam_on_cluster():
        base_dir = Path(f'/{data_name}')
    else:
        base_dir = project_paths.data.raw / data_name

    return base_dir

def parse_arguments(wildcards, args_key='model_args', delimiter=',', assigner='=', prefix=":"):
    args = getattr(wildcards, args_key)
    args = args.lstrip(prefix).split(delimiter)

    if len(args) == 1 and not args[0]:
        return ""

    args_dict = {arg.split(assigner)[0]: arg.split(assigner)[1] for arg in args}

    cmd_args = [f"--{key} {value}" for key, value in args_dict.items()]
    
    return " ".join(cmd_args)

def args_product(args_dict=config.model_args, delimiter=',', assigner='=', prefix=':'):
    if not args_dict or not len(args_dict):
        return ['']

    for key, value in args_dict.items():
        if not isinstance(value, list):
            args_dict[key] = [value]

    args_combinations = product(*args_dict.values())
    args_strings = []

    for args_combination in args_combinations:
        arg_string = []
        for key, values in zip(args_dict.keys(), args_combination):
            arg_string.append(f'{key}{assigner}{values}')
        
        args_strings.append(prefix + delimiter.join(arg_string))
    
    return args_strings

def dict_poped(d, key):
    dc = d.copy()
    dc.pop(key, None)
    return dc