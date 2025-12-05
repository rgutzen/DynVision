def compute_hash(*args) -> str:
	"""
	create hash from all args. If they already contain 'hash', return it unchanged
	"""
	return hash_str

rule init_model:
	output:
		project_paths.models \
		/ "DyRCNNx8:{model_args}_{seed}" \
		/ "{data_name}:{train_data_group}" \
		/ "init.pt"

checkpoint train_model:
	# for this rule model_args must not contain 'hash'
	input:
		project_paths.models \
		/ "{model_name}:{model_args}_{seed}" \
		/ "{data_name}:{train_data_group}" \
		/ "init.pt"
	params:
		symlink_folder = lambda w: project_paths.models 
			/ f"{w.model_name}:{compute_hash(w.model_args, w.seed)}",  
		target_folder = lambda w: project_paths.models 
			/ f"{w.model_name}:{w.model_args}_{w.seed}"
			/ f"{w.data_name}:{w.train_data_group}",
		model_hash_file = lambda w: project_paths.models 
			/ f"{w.model_name}:{w.model_args}_{w.seed}"
			/ f"{w.data_name}:{w.train_data_group}",
			/ f"{compute_hash(w.model_args, w.seed)}.hash"  # to document the hash in the model folder
	output:
		project_paths.models \
		/ "{model_name}:{model_args}_{seed}" \
		/ "{data_name}:{train_data_group}" \
		/ "trained.pt"
	shell:
		"""
		python ....

		echo "{w.model_args}_{w.seed}" > {params.model_hash_file}	
		ln -s {output.parent} {params.symlink_folder}
		"""

rule test_model:
	# model_identifier can be <model_args>_<seed> or hash
	input:
		project_paths.models \
		/ "{model_name}:{model_identifier}" \
		/ "{data_name}-{train_data_group}" \
		/ "{status}.pt"
	output:
		project_paths.reports \
		/ "{experiment}" \
		/ "{model_name}:{model_identifier}" \
		/ "{data_name}:{train_data_group}_{status}_{data_name}:{data_group}_tested" \
		/ "{data_loader}:{data_args}" \
		/ "test_outputs.csv"

rule process_test_data:
	input:
		models = lambda w: [project_paths.models 
		/ "{{model_name}}:{{args1}}{{category}}={cat_value}{{args2}}_{{seed}}"
		/ "{{data_name}}-{{train_data_group}}" 
		/ "{status}.pt" for cat_value in <look up values in experiment config>...],  # invokes train_model rule which generates also hashed symlinks for the test_model rule
		test_outputs = expand(project_paths.reports 
		/ "{{experiment}}"
		/ "{{model_name}}:{hash_id}"
		/ "{{data_name}}:{train_data_group}_{status}_{data_name}:{data_group}_tested"
		/ "{{data_loader}}:{data_args}" 
		/ "test_outputs.csv", 
			hash_id = lambda w: [compress_model_args(f"{w.args1}{w.category}={cat_value}{w.args2}", w.seed) for cat_value in <look up values in experiment config>...],
			data_args=...
		)  # triggers test_model rule
	params:
		# when model_identifier is hash, we need to look up the category values to pass them to the script
		cat_values = lambda w: config.experiment_config['categories'].get(w.category, ''),
	output:
		project_paths.reports \
		/ "{experiment}" \
		/ "{model_name}:{args1}{category}=*{args2}_{seed}" \
		/ "{data_name}:{train_data_group}_{status}_{data_name}:{data_group}_tested" \
		/ "test_data.csv"

rule plotting_rules:
	input:
		project_paths.reports \
		/ "{experiment}" \
		/ "{model_name}:{model_args}_{seed}" \
		/ "{data_name}:{train_data_group}_{status}_{data_name}:{data_group}" \
		/ "test_data.csv"
	output:
		project_paths.figures \
		/ "{experiment}" \
		/ "{model_name}:{model_args}_{seed}" \
		/ "{data_name}:{train_data_group}_{status}_{data_name}:{data_group}" \
		/ "{plot}.png"