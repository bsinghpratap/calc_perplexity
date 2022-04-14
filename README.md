## Repo to calculate the perplexity for the MIMIC, VA and Pitts dataset 

- Command from the og repo  
`python scripts/run_mlm_no_trainer.py --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --model_name_or_path roberta-base --output_dir /tmp/test-mlm`


`python scripts/run_mlm_no_trainer.py --validation_file=/mnt/nfs/work1/hongyu/brawat/bionlp/calc_perplexity/data/mimic_para_train.csv --model_name_or_path=roberta-base --output_dir=/mnt/nfs/work1/hongyu/brawat/bionlp/calc_perplexity/logs/model_mimic_xx --overwrite_cache`
