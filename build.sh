
python entrypoint.py infer \
--model_filepath ./model/id-00000001/model.pt \
--result_filepath ./scratch/output.txt \
--scratch_dirpath ./scratch \
--examples_dirpath ./model/id-00000001/clean-example-data \
--round_training_dataset_dirpath /path/to/train-dataset \
--learned_parameters_dirpath ./learned_parameters \
--metaparameters_filepath ./metaparameters.json \
--schema_filepath=./metaparameters_schema.json 


sudo singularity build cyber-network-c2-mar2024_test_wa_v0.simg detector.def


singularity run \
--bind /work2/project/trojai-cyber-network/model \
--nv \
./cyber-network-feb2024_sts_SRI_trinity_v0.simg \
infer \
--model_filepath /work2/project/trojai-cyber-network/model/id-00000001/model.pt \
--result_filepath=/output.txt \
--scratch_dirpath=/scratch/ \
--examples_dirpath /work2/project/trojai-cyber-network/model/id-00000001/clean-example-data \
--round_training_dataset_dirpath=/path/to/training/dataset/ \
--metaparameters_filepath=/metaparameters.json \
--schema_filepath=/metaparameters_schema.json \
--learned_parameters_dirpath=/learned_parameters/ 



#python entrypoint.py configure \
--scratch_dirpath ./scratch/ \
--metaparameters_filepath ./metaparameters.json \
--schema_filepath ./metaparameters_schema.json \
--learned_parameters_dirpath ./learned_parameters \
--configure_models_dirpath /work2/project/trojai-datasets/cyber-pdf-dec2022-train/models \
--scale_parameters_filepath /work2/project/trojai-datasets/cyber-pdf-dec2022-train/scale_params.npy


#python entrypoint.py configure \
--scratch_dirpath ./scratch/ \
--metaparameters_filepath ./metaparameters.json \
--schema_filepath ./metaparameters_schema.json \
--learned_parameters_dirpath ./learned_parameters \
--configure_models_dirpath /work2/project/trojai-datasets/cyber-pdf-dec2022-train/models \
--scale_parameters_filepath /work2/project/trojai-datasets/cyber-pdf-dec2022-train/scale_params.npy \
--automatic_configuration


