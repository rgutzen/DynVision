from types import SimpleNamespace

from snakemake.logging import logger


def in_quotes(object, characters=["\s", "|"]):
    """
    when these characters are in an argument str they need to be put in quotes
    to be passed as command line argument.
    """
    return any([c in str(object) for c in characters])


def create_cla_str(key, value):
    return f'--{key} "{value}"' if in_quotes(value) else f"--{key} {value}"


def dict_to_cla(arg_dict):
    for key, value in arg_dict.items():
        if isinstance(value, list):
            arg_dict[key] = " ".join(str(v) for v in value)

    arg_strings = [create_cla_str(key, value) for key, value in arg_dict.items()]
    return " ".join(arg_strings)


def params(*args, config=None, **kwargs):
    """
    Creates a parameter dictionary that is returned as a string as
    command line argument as `--key value` for each key-value pair in the dict.

    args:
        - if args[0] is a dict, it is added to the parameter dict
        - if args are strings and config is a dict, args (and args.upper())
          are interpreted as keys of config dict and the corresponding elements
          are added to the parameter dict
    config:
        dict from which elements (keys given by args) are selected
    kwargs:
        kwargs are added as key-value pairs are to the parameter dict

    Additionally, all wildcards and named outputs of the current rule are added
    as key-value pairs to the parameter dict.
    """

    param_dict = {}

    if len(args) and isinstance(args[0], dict):
        param_dict = args[0]

    if isinstance(config, SimpleNamespace):
        config = vars(config)

    if isinstance(config, dict):
        for arg in args:
            if not isinstance(arg, str):
                continue
            if arg in config.keys():
                param_dict[arg] = config[arg]
            elif arg.upper() in config.keys():
                param_dict[arg] = config[arg.upper()]
            else:
                logger.info(
                    f"Parameter '{arg}' not found in the config! " "Set to None!"
                )
                param_dict[arg] = None

    param_dict.update(dict(kwargs.items()))

    def add_output_and_wildcards_to_args(wildcards, output):
        for items in [wildcards, output]:
            item_dict = dict(items.items())
            if "data" in item_dict.keys():
                logger.warning(
                    "wildcards or outputs named 'data' " "are being ignored!"
                )
                del item_dict["data"]

            duplicates = [key for key in item_dict.keys() if key in param_dict.keys()]

            for key in duplicates:
                if param_dict[key] != item_dict[key]:
                    logger.warning(
                        f"The keyword {key} is used multiple times "
                        "in the rule's params, wildcards, or output! \n"
                        f"{key}: '{param_dict[key]}' is ignored "
                        f"in favor of '{item_dict[key]}'"
                    )

            param_dict.update(item_dict)
        return dict_to_cla(param_dict)

    return add_output_and_wildcards_to_args
