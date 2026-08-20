# train the Struct-XLM
python src/Struct_XLM/run_main.py --do_train --do_eval \
--output_dir 'output_struct-xlm/' \
--data_dir 'data/UD_data/' \
--sentence_type 'mean' \
--ml_type 'xlmr' \
--ml_model_path '/home/shenyl/cached_models/xlm-roberta-large' \
--max_len 256 \
--margin 0.4 \
--act_dropout 0.5 \
--tau 0.5 \
--alpha 0.5 \
--learning_rate 6e-6 \
--warm_start \
--batch_size 5

python src/Struct_XLM/run_main.py --do_train --do_eval \
--output_dir 'output_struct-xlm/' \
--data_dir 'data/UD_data/' \
--sentence_type 'mean' \
--ml_type 'xlmr' \
--ml_model_path 'xlm-roberta-large' \
--max_len 256 \
--margin 0.4 \
--act_dropout 0.5 \
--tau 0.5 \
--alpha 0.5 \
--learning_rate 6e-6 \
--warm_start \
--LMpretrain \
--batch_size 5

python src/Struct_XLM/run_main.py --do_train --do_eval \
--output_dir 'output_struct-xlmr_noreward/' \
--data_dir 'data/UD_data/' \
--sentence_type 'mean' \
--ml_type 'xlmr' \
--ml_model_path 'xlm-roberta-large' \
--max_len 256 \
--margin 0.4 \
--act_dropout 0.5 \
--tau 0.5 \
--alpha 0.5 \
--learning_rate 6e-6 \
--warm_start \
--batch_size 5  --LMpretrain --RLpretrain


# script for seven XTREME Benckmark tasks
# 1、Training XNLI and evaluation
python src/xtreme_7/run_classify_XNLI.py --save_only_best_checkpoint --do_predict --do_train \
--data_dir 'src/xtreme_7/data/XNLI' \
--model_type 'xlmr' \
--model_name_or_path '/home/shenyl/cached_models/xlm-roberta-large' \
--output_dir 'src/xtreme_7/output/XNLI' \
--critic_output_dir 'output_struct-xlm/' \
--per_gpu_train_batch_size 16 \
--gradient_accumulation_steps 2 \
--learning_rate 8e-6  \
--weight_decay 0.0 \
--num_train_epochs 10.0 \
--warmup_steps  5000 \

python src/xtreme_7/run_classify_XNLI_test.py --save_only_best_checkpoint --do_predict --do_train \
--data_dir 'src/xtreme_7/data/XNLI' \
--model_type 'xlmr' \
--model_name_or_path 'xlm-roberta-large' \
--output_dir 'src/xtreme_7/output/XNLI' \
--critic_output_dir 'output_struct-xlm-12/' \
--per_gpu_train_batch_size 16 \
--gradient_accumulation_steps 2 \
--learning_rate 8e-6  \
--weight_decay 0.0 \
--num_train_epochs 10.0 \
--warmup_steps  5000 \
--train_language en \
--predict_languages en

 CUDA_VISIBLE_DEVICES=0  python src/xtreme_7/run_classify_XNLI_lora.py --save_only_best_checkpoint --do_predict --do_train \
--data_dir 'src/xtreme_7/data/XNLI' \
--model_type 'xlmr' \
--model_name_or_path 'xlm-roberta-large' \
--output_dir 'src/xtreme_7/output/XNLI_noreward' \
--critic_output_dir 'output_struct-xlmr_nocreward/' \
--per_gpu_train_batch_size 16 \
--gradient_accumulation_steps 2 \
--learning_rate 4e-4  \
--weight_decay 0.0 \
--num_train_epochs 10.0 \
--warmup_steps  5000

python src/xtreme_7/run_classify_XNLI_inforXLM.py --save_only_best_checkpoint --do_predict \
--data_dir 'src/xtreme_7/data/XNLI' \
--output_dir 'src/xtreme_7/output/XNLI_inforxlm' \
--per_gpu_train_batch_size 16 \
--gradient_accumulation_steps 2 \
--learning_rate 4e-4  \
--weight_decay 0.0 \
--num_train_epochs 10.0 \
--warmup_steps  5000

python src/xtreme_7/run_classify_XNLI_inforXLM.py \
--do_train --do_eval --do_predict \
--data_dir src/xtreme_7/data/XNLI/ \
--output_dir /workspace/Struct-XLM-main/src/xtreme_7/output/XNLI_inforxlm/ \
--model_name_or_path microsoft/infoxlm-base \
--overwrite_cache --overwrite_output_dir

python src/xtreme_7/run_classify_XNLI_mmbert.py \
--model_name_or_path jhu-clsp/mmBERT-small \
--data_dir src/xtreme_7/data/XNLI/ \
--output_dir /workspace/Struct-XLM-main/src/xtreme_7/output/XNLI_mmbert/ \
--do_train --do_eval --do_predict \
--learning_rate 1e-6 \
--per_gpu_train_batch_size 8 \
--gradient_accumulation_steps 2 \
--num_train_epochs 3 \
--max_grad_norm 0.5 \
--save_steps 500 \
--logging_steps 100 \
--save_only_best_checkpoint \
--overwrite_output_dir --overwrite_cache

# 2、Training PAWS-X and evaluation
python src/xtreme_7/run_classify_pawsx.py --save_only_best_checkpoint --do_predict --do_train \
--data_dir 'src/xtreme_7/data/PAWS-X' \
--model_type 'xlmr' \
--model_name_or_path '/home/shenyl/cached_models/xlm-roberta-large' \
--output_dir 'src/xtreme_7/output/paws-x' \
--critic_output_dir 'output_struct-xlm/' \
--per_gpu_train_batch_size 16 \
--gradient_accumulation_steps 2 \
--learning_rate 1e-5  \
--weight_decay 0.0 \
--num_train_epochs 10.0 \
--warmup_steps  0.1 \

python src/xtreme_7/run_classify_pawsx_lora.py --save_only_best_checkpoint --do_predict --do_train \
--data_dir 'src/xtreme_7/data/PAWS-X' \
--model_type 'xlmr' \
--model_name_or_path 'xlm-roberta-large' \
--output_dir 'src/xtreme_7/output/pawsx_noreward' \
--critic_output_dir 'output_struct-xlmr_noreward/' \
--per_gpu_train_batch_size 16 \
--gradient_accumulation_steps 2 \
--learning_rate 4e-4  \
--weight_decay 0.0 \
--num_train_epochs 10.0 \
--warmup_steps  0.1 \
--train_language en \
--predict_languages en

python src/xtreme_7/run_classify_pawsx.py --save_only_best_checkpoint --do_predict --do_train \
--data_dir 'src/xtreme_7/data/PAWS-X' \
--model_type 'xlmr' \
--model_name_or_path 'xlm-roberta-large' \
--output_dir 'src/xtreme_7/output/pawsx' \
--critic_output_dir 'output_struct-xlm-12/' \
--per_gpu_train_batch_size 16 \
--gradient_accumulation_steps 2 \
--learning_rate 1e-5  \
--weight_decay 0.0 \
--num_train_epochs 10.0 \
--warmup_steps  0.1 \
--train_language en \
--predict_languages en

python src/xtreme_7/run_classify_pawsx_infoXLM.py \
--do_train --do_eval --do_predict \
--data_dir src/xtreme_7/data/PAWS-X/ \
--output_dir src/xtreme_7/output/pawsx_infoXLM/ \
--model_name_or_path microsoft/infoxlm-base \
--overwrite_cache --overwrite_output_dir

python src/xtreme_7/run_classify_pawsx_mmbert.py \
--model_name_or_path jhu-clsp/mmBERT-small \
--data_dir src/xtreme_7/data/PAWS-X/ \
--output_dir src/xtreme_7/output/pawsx_mmbert/ \
--do_train --do_eval --do_predict \
--learning_rate 1e-6 \
--per_gpu_train_batch_size 8 \
--gradient_accumulation_steps 2 \
--num_train_epochs 3 \
--max_grad_norm 0.5 \
--save_steps 500 \
--logging_steps 100 \
--save_only_best_checkpoint \
--overwrite_output_dir --overwrite_cache

# 3、Training NER and evaluation
python src/xtreme_7/run_tag_ner.py --save_only_best_checkpoint --do_predict --do_train \
--data_dir 'src/xtreme_7/data/NER' \
--labels 'src/xtreme_7/data/NER/label.txt' \
--model_type 'xlmr' \
--model_name_or_path '/home/shenyl/cached_models/xlm-roberta-large' \
--output_dir 'src/xtreme_7/output/NER' \ 
--critic_output_dir 'output_struct-xlm/' \
--per_gpu_train_batch_size 16 \
--gradient_accumulation_steps 2 \
--learning_rate 4e-4  \
--weight_decay 0.0 \
--num_train_epochs 10.0 \
--warmup_steps  0.1 \

python src/xtreme_7/run_tag_ner_lora.py --save_only_best_checkpoint --do_predict --do_train \
--data_dir 'src/xtreme_7/data/NER' \
--labels 'src/xtreme_7/data/NER/label.txt' \
--model_type 'xlmr' \
--model_name_or_path 'xlm-roberta-large' \
--output_dir 'src/xtreme_7/output/NER_noreward' \ 
--critic_output_dir 'output_struct-xlmr_noreward/' \
--per_gpu_train_batch_size 16 \
--gradient_accumulation_steps 2 \
--learning_rate 4e-4  \
--weight_decay 0.0 \
--num_train_epochs 10.0 \
--warmup_steps  0.1

python src/xtreme_7/run_tag_ner_lora.py --save_only_best_checkpoint --do_predict --do_train \
--data_dir 'src/xtreme_7/data/NER' \
--labels 'src/xtreme_7/data/NER/label.txt' \
--model_type 'mbert' \
--model_name_or_path 'bert-base-multilingual-cased' \
--output_dir 'src/xtreme_7/output/NER' \ 
--critic_output_dir 'output_struct-mbert/' \
--per_gpu_train_batch_size 16 \
--gradient_accumulation_steps 2 \
--learning_rate 4e-4  \
--weight_decay 0.0 \
--num_train_epochs 10.0 \
--warmup_steps  0.1

python src/xtreme_7/run_tag_ner_infoXLM.py \
--do_train --do_eval --do_predict \
--data_dir src/xtreme_7/data/NER \
--labels src/xtreme_7/data/NER/label.txt \
--output_dir src/xtreme_7/output/NER_infoXLM/ \
--model_name_or_path microsoft/infoxlm-base \
--overwrite_cache --overwrite_output_dir

python src/xtreme_7/run_tag_ner_mmbert.py \
--model_name_or_path jhu-clsp/mmBERT-small \
--data_dir src/xtreme_7/data/NER/ \
--labels src/xtreme_7/data/NER/label.txt \
--output_dir src/xtreme_7/output/NER_mmbert \
--do_train --do_eval --do_predict \
--learning_rate 1e-6 \
--per_gpu_train_batch_size 8 \
--gradient_accumulation_steps 2 \
--num_train_epochs 3 \
--save_steps 500 \
--logging_steps 100 \
--save_only_best_checkpoint \
--overwrite_output_dir --overwrite_cache

# 4、Training POS and evaluation
python src/xtreme_7/run_tag_udpos.py --save_only_best_checkpoint --do_predict --do_train \
--data_dir 'src/xtreme_7/data/udpos' \
--labels 'src/xtreme_7/data/udpos/label.txt' \
--model_type 'xlmr' \
--model_name_or_path '/home/shenyl/cached_models/xlm-roberta-large' \
--output_dir 'src/xtreme_7/output/udpos' \
--critic_output_dir 'output_struct-xlm/' \
--per_gpu_train_batch_size 16 \
--gradient_accumulation_steps 2 \
--learning_rate 9e-6  \
--weight_decay 0.0 \
--num_train_epochs 10.0 \
--warmup_steps  0.1 \

python src/xtreme_7/run_tag_udpos.py --save_only_best_checkpoint --do_predict --do_train \
--data_dir 'src/xtreme_7/data/udpos' \
--labels 'src/xtreme_7/data/udpos/label.txt' \
--model_type 'xlmr' \
--model_name_or_path 'xlm-roberta-large' \
--output_dir 'src/xtreme_7/output/udpos' \
--critic_output_dir 'output_struct-xlm-12/' \
--per_gpu_train_batch_size 16 \
--gradient_accumulation_steps 2 \
--learning_rate 9e-6  \
--weight_decay 0.0 \
--num_train_epochs 10.0 \
--warmup_steps  0.1 \
--train_langs en \
--predict_langs en

python src/xtreme_7/run_tag_udpos_test_lora.py --save_only_best_checkpoint --do_predict --do_train \
--data_dir 'src/xtreme_7/data/udpos' \
--labels 'src/xtreme_7/data/udpos/label.txt' \
--model_type 'xlmr' \
--model_name_or_path 'xlm-roberta-large' \
--output_dir 'src/xtreme_7/output/udpos_noreward' \
--critic_output_dir 'output_struct-xlmr_noreward/' \
--per_gpu_train_batch_size 16 \
--gradient_accumulation_steps 2 \
--learning_rate 5e-4  \
--weight_decay 0.0 \
--num_train_epochs 10.0 \
--warmup_steps  0.1 \
--train_langs en \
--predict_langs en

python src/xtreme_7/run_tag_udpos_lora_noaction.py --save_only_best_checkpoint --do_predict --do_train \
--data_dir 'src/xtreme_7/data/udpos' \
--labels 'src/xtreme_7/data/udpos/label.txt' \
--model_type 'xlmr' \
--model_name_or_path 'xlm-roberta-large' \
--output_dir 'src/xtreme_7/output/udpos_noaction' \
--critic_output_dir 'output_struct-xlm-12/' \
--per_gpu_train_batch_size 16 \
--gradient_accumulation_steps 2 \
--learning_rate 5e-4  \
--weight_decay 0.0 \
--num_train_epochs 10.0 \
--warmup_steps  0.1

python src/xtreme_7/run_tag_udpos_infoXLM.py \
--do_train --do_eval --do_predict \
--data_dir src/xtreme_7/data/udpos/ \
--labels src/xtreme_7/data/udpos/label.txt \
--output_dir src/xtreme_7/output/udpos_infoXLM/ \
--model_name_or_path microsoft/infoxlm-base \
--overwrite_cache --overwrite_output_dir

python src/xtreme_7/run_tag_udpos_mmbert.py \
--model_name_or_path jhu-clsp/mmBERT-small \
--data_dir src/xtreme_7/data/udpos/ \
--labels src/xtreme_7/data/udpos/label.txt \
--output_dir src/xtreme_7/output/udpos_mmbert/  \
--do_train --do_eval --do_predict \
--learning_rate 1e-6 \
--per_gpu_train_batch_size 8 \
--gradient_accumulation_steps 2 \
--num_train_epochs 3 \
--save_steps 500 \
--logging_steps 100 \
--save_only_best_checkpoint \
--overwrite_output_dir --overwrite_cache

# 5、Training XQuAD and evaluation
python src/xtreme_7/run_squad.py --save_only_best_checkpoint --do_predict --do_train \
--data_dir 'src/xtreme_7/data/MRC' \
--test_file 'xquad' \
--model_type 'xlmr' \
--model_name_or_path 'xlm-roberta-large' \
--output_dir 'src/xtreme_7/output/xquad' \
--critic_output_dir 'output_struct-xlm-12/' \
--per_gpu_train_batch_size 8 \
--gradient_accumulation_steps 4 \
--learning_rate 1e-5  \
--weight_decay 0.0 \
--num_train_epochs 3.0 \
--warmup_steps  0.1 \

python src/xtreme_7/run_squad_lora.py --save_only_best_checkpoint --do_predict --do_train \
--data_dir 'src/xtreme_7/data/MRC' \
--test_file 'xquad' \
--model_type 'xlmr' \
--model_name_or_path 'xlm-roberta-large' \
--output_dir 'src/xtreme_7/output/xquad_noreward' \
--critic_output_dir 'output_struct-xlmr_noreward/' \
--per_gpu_train_batch_size 8 \
--gradient_accumulation_steps 4 \
--learning_rate 5e-4  \
--weight_decay 0.0 \
--num_train_epochs 3.0 \
--warmup_steps  0.1

python src/xtreme_7/run_squad_infoXLM.py \
--do_train --do_eval --do_predict \
--data_dir src/xtreme_7/data/MRC/ \
--test_file 'xquad' \
--output_dir src/xtreme_7/output/xquad_infoXLM/ \
--model_name_or_path microsoft/infoxlm-base \
--overwrite_cache --overwrite_output_dir

python src/xtreme_7/run_squad_mmbert.py \
--model_name_or_path jhu-clsp/mmBERT-small \
--data_dir src/xtreme_7/data/MRC/ \
--output_dir src/xtreme_7/output/xquad_mmbert/ \
--do_train --do_predict \
--test_file 'xquad' \
--learning_rate 1e-5 \
--per_gpu_train_batch_size 4 \
--gradient_accumulation_steps 4 \
--per_gpu_eval_batch_size 8 \
--num_train_epochs 3 \
--max_grad_norm 0.5 \
--save_steps 500 \
--logging_steps 100 \
--save_only_best_checkpoint \
--overwrite_output_dir --overwrite_cache

# 6、Training MLQA and evaluation
python src/xtreme_7/run_squad_lora.py --save_only_best_checkpoint --do_predict --do_train \
--data_dir 'src/xtreme_7/data/MRC' \
--test_file 'mlqa' \
--model_type 'xlmr' \
--model_name_or_path 'xlm-roberta-large' \
--output_dir 'src/xtreme_7/output/mlqa_noreward' \
--critic_output_dir 'output_struct-xlmr_noreward/' \
--per_gpu_train_batch_size 8 \
--gradient_accumulation_steps 4 \
--learning_rate 5e-4  \
--weight_decay 0.0 \
--num_train_epochs 2.0 \
--warmup_steps  0.1 \

python src/xtreme_7/run_squad_lora_noaction.py --save_only_best_checkpoint --do_predict --do_train \
--data_dir 'src/xtreme_7/data/MRC' \
--test_file 'mlqa' \
--model_type 'xlmr' \
--model_name_or_path 'xlm-roberta-large' \
--output_dir 'src/xtreme_7/output/mlqa_noaction' \
--critic_output_dir 'output_struct-xlm-12/' \
--per_gpu_train_batch_size 8 \
--gradient_accumulation_steps 4 \
--learning_rate 5e-4  \
--weight_decay 0.0 \
--num_train_epochs 2.0 \
--warmup_steps  0.1

python src/xtreme_7/run_squad_infoXLM.py \
--do_train --do_eval --do_predict \
--data_dir src/xtreme_7/data/MRC/ \
--test_file 'mlqa' \
--output_dir src/xtreme_7/output/mlqa_infoXLM/ \
--model_name_or_path microsoft/infoxlm-base \
--overwrite_cache --overwrite_output_dir

python src/xtreme_7/run_squad_mmbert.py \
--model_name_or_path jhu-clsp/mmBERT-small \
--data_dir src/xtreme_7/data/MRC/ \
--output_dir src/xtreme_7/output/mlqa_mmbert/ \
--do_train --do_predict \
--test_file mlqa \
--learning_rate 1e-6 \
--per_gpu_train_batch_size 4 \
--gradient_accumulation_steps 4 \
--per_gpu_eval_batch_size 8 \
--num_train_epochs 3 \
--max_grad_norm 0.5 \
--save_steps 500 \
--logging_steps 100 \
--save_only_best_checkpoint \
--overwrite_output_dir --overwrite_cache

# 7、Training TyDiQA and evaluation
python src/xtreme_7/run_tydiqa.py --save_only_best_checkpoint --do_predict --do_train \
--data_dir 'src/xtreme_7/data/MRC' \
--test_file 'tydiqa' \
--model_type 'xlmr' \
--model_name_or_path 'xlm-roberta-large' \
--output_dir 'src/xtreme_7/output/tydiqa' \
--critic_output_dir 'output_struct-xlm-12/' \
--per_gpu_train_batch_size 8 \
--gradient_accumulation_steps 4 \
--learning_rate 1e-5  \
--weight_decay 0.0 \
--num_train_epochs 10.0 \
--warmup_steps  0.1 \

python src/xtreme_7/run_tydiqa_lora.py --save_only_best_checkpoint --do_predict --do_train \
--data_dir 'src/xtreme_7/data/MRC' \
--test_file 'tydiqa' \
--model_type 'xlmr' \
--model_name_or_path 'xlm-roberta-large' \
--output_dir 'src/xtreme_7/output/tydiqa_noreward' \
--critic_output_dir 'output_struct-xlmr_noreward/' \
--per_gpu_train_batch_size 8 \
--gradient_accumulation_steps 4 \
--learning_rate 3.5e-4  \
--weight_decay 0.0 \
--num_train_epochs 10.0 \
--warmup_steps  0.1

python src/xtreme_7/run_tydiqa_infoXLM.py \
--do_train --do_eval --do_predict \
--data_dir src/xtreme_7/data/MRC/ \
--train_file squad/train-v1.1.json \
--predict_file squad/dev-v1.1.json \
--test_file tydiqa \
--output_dir src/xtreme_7/output/tydiqa_infoXLM/ \
--model_name_or_path microsoft/infoxlm-base \
--overwrite_cache --overwrite_output_dir

python src/xtreme_7/run_tydiqa_mmbert.py \
--model_name_or_path jhu-clsp/mmBERT-small \
--data_dir src/xtreme_7/data/MRC/ \
--train_file tydiqa/tydiqa.en.train.json \
--output_dir src/xtreme_7/output/TyDiQA_mmbert \
--do_train --do_predict \
--test_file tydiqa \
--learning_rate 1e-6 \
--per_gpu_train_batch_size 4 \
--gradient_accumulation_steps 4 \
--num_train_epochs 3 \
--save_steps 500 \
--logging_steps 100 \
--save_only_best_checkpoint \
--overwrite_output_dir --threads 1 --overwrite_cache