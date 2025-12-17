## list all incorrect predictions
#python -m src.inference \
#--ckpt runs/first_run_epochs/best_model \
#--data-csv dataset/splits/test.csv \
#--out-csv runs/first_run_epochs/misclassified_test.csv \
#--labeled


# inference on unlabelled infer data
python -m src.inference \
--ckpt runs/first_run_epochs/best_model \
--data-csv dataset/processed/infer_data.csv \
--out-csv runs/first_run_epochs/infer_data.csv \
--batch-size 512
