def compute_hash(*args) -> str:
	"""
	create hash from all args. If they already contain 'hash', return it unchanged
	"""
	return hash_str

rule init_model:  # unchanged
	output:
		project_paths.models / "DyRCNNx8:{model_args}_{seed}_{data_name}/init.pt"

checkpoint train_model:
	# for this rule model_args must not contain 'hash'
	input:
		project_paths.models / "DyRCNNx8:{model_args}_{seed}_{data_name}/init.pt"
	params:
		symlink_folder = lambda w: project_paths.models / f"DyRCNNx8:{compute_hash(w.model_args, w.seed, w.data_name)}",  
		target_folder = lambda w: project_paths.models / f"DyRCNNx8:{w.model_args}_{w.seed}_{w.data_name}"
		model_hash_file = lambda w: project_paths.models / f"DyRCNNx8:{w.model_args}_{w.seed}_{w.data_name}" / f"{compute_hash(w.model_args, w.seed, w.data_name)}.hash"  # to document the hash in the model folder
	output:
		project_paths.models / "DyRCNNx8:{model_args}_{seed}_{data_name}/trained.pt"
	shell:
		"""
		python ....

		touch {params.model_hash_file} 
		ln -s {output.parent} {params.symlink_folder}
		"""

rule test_model:
	# model_identifier can be args_seed_dataname or hash
	input:
		project_paths.models / "DyRCNNx8:{model_identifier}/{status}.pt"
	output:
		project_paths.reports / "DyRCNNx8:{model_identifier}_{status}/StimulusNoise:{data_args}_{data_group}/test_outputs.csv"

rule process_test_data:
	input:
		models = expand(project_paths.models / "DyRCNNx8:{model_args}_{seed}_{data_name}" / "{status}.pt", ...),  # invokes train_model rule which generates also hashed symlinks for the test_model rule
		test_outputs = lambda w: expand(project_paths.reports / f"DyRCNNx8:{compress_model_args(w.model_args, w.seed, w.data_name)}_{w.status}/StimulusNoise:{data_args}_{data_group}/test_outputs.csv", ...)  # triggers test_model rule
	output:
		project_paths.reports / "experiment_DyRCNNx8:{model_args}_{seed}_{data_name}_{status}/test_data_{data_group}.csv"

rule plotting_rules:
	input:
		project_paths.reports / "{experiment}_DyRCNNx8:{model_args}_{seed}_{data_name}_{status}/{data_group}_test_data.csv"
	output:
		project_paths.figures / "{experiment}_DyRCNNx8:{model_args}_{seed}_{data_name}_{status}/{data_group}_performance.png"