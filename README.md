This is a fork of the [repo](https://github.com/zijwang/modeling_guilt/tree/master) for "[Modeling Subjective Assessments of Guilt in Newspaper Crime Narratives](https://aclanthology.org/2020.conll-1.5/)".

<!-- # Modeling Subjective Assessments of Guilt in Newspaper Crime Narratives -->

<!-- This is the GitHub repository for our paper "[Modeling Subjective Assessments of Guilt in Newspaper Crime Narratives](https://arxiv.org/abs/2006.09589)" in proceedings of CONLL'20 by Elisa Kreiss\*, Zijian Wang\* and Christopher Potts.  -->



<!-- ## SuspectGuilt Corpus -->

<!-- We are excited to share the SuspectGuilt Corpus: a collection of 1,821 annotated local news crime reports from across the US. We recruited over 2000 workers on Amazon Mechanical Turk who rated  -->

<!-- * how likely they considered the suspect(s) of a story to be guilty (**reader perception**), and  -->
<!-- * how much the author of the report believes that the suspect(s) are guilty (**author belief**).  -->

<!-- For each question, annotators also highlighted why they gave their response. Since guilt judgments are subjective and known to be highly variable, each crime report was annotated by at least 5 workers. -->

<!-- We would like to encourage more research in the domain of crime reporting and specifically guilt perception. **To receive our corpus, please send an email to ekreiss@stanford.edu and zijwang@stanford.edu.** This corpus presents an extensive collection of news articles with rich guilt rating and highlighting annotations, as well as the annotators' self-reported age, gender, and native language information. Note that we don't own any distribution rights of the collected crime reports. -->

<!-- For more details,  please refer to our paper and the `annotation` dir, which contains code for the annotation experiment and a detailed corpus analysis. -->


<!-- ## Modeling -->
<!-- ### Genre Pretraining -->
<!-- Please refer to `modeling/run_lm_finetuning.py`. Example command (requires the 474k dataset to run): -->

<!-- ``` -->
<!-- python run_lm_finetuning.py --fp16  --output_dir ./uncased_model_100k/ --mlm --save_steps 5000 --max_steps 100000  --model_name_or_path bert-base-uncased --do_lower_case --warmup_steps 5000 --do_train --do_eval --eval_all_checkpoints -->
<!-- ``` -->

<!-- ### BERT Model -->
<!-- Please refer to `modeling/model/main.py`. Example command (requires the annotated dataset to run): -->
<!-- ``` -->
<!-- python main.py --num_train_epochs 5 --token_cls --do_lower_case --seed 0 --token_ratio 0 --training_head 0 --output_dir ./temp_model --learning_rate 3e-05 --do_train --do_eval --overwrite_output_dir --eval_all_checkpoint -->
<!-- ``` -->


<!-- ### Analysis -->

<!-- The main results are in `modeling/result_analysis.ipynb`. Please refer to `modeling/interpret.ipynb` -->
<!-- for an example of running Integrated Gradients on our dataset using [captum](https://captum.ai/tutorials/). -->



<!-- ## Citation -->

<!-- ``` -->
<!-- @inproceedings{kreiss2020modeling, -->
<!--   title={Modeling Subjective Assessments of Guilt in Newspaper Crime Narratives}, -->
<!--   author={Kreiss, Elisa and Wang, Zijian and Potts, Christopher}, -->
<!--   booktitle={Proceedings of the 24th Conference on Computational Natural Language Learning (CoNLL)}, -->
<!--   year={2020}, -->
<!--   url={https://www.aclweb.org/anthology/2020.conll-1.5}, -->
<!--   pages = {56--68}, -->
<!--   publisher = {Association for Computational Linguistics}, -->
<!-- } -->
<!-- ``` -->
