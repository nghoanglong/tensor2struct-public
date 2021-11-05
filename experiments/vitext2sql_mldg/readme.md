# Domain Generalization via Meta-Learning
## Prepare Data
### Vitext2sql - VietNamese Dataset
Download from [VitextSQL](https://github.com/VinAIResearch/ViText2SQL)
 
## Train PhoBERT models for Vitext2sql
First, preprocess the data:
```
python run.py preprocess configs/vitext2sql/run_config/vitext2sql_phobert_mldg.jsonnet
```
To meta-train a supervised model
```
python run.py meta_train configs/vitext2sql/run_config/vitext2sql_phobert_mldg.jsonnet
```
To eval model
```
python run.py eval configs/vitext2sql/run_config/vitext2sql_phobert_mldg.jsonnet
```