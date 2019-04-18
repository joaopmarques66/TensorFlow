### Rename all files to sequential order on linux
`ls | cat -n | while read n f; do mv "$f" `printf "%04d.jpg" $n`; done`


## Pipeline
1. Train model - `python train.py`
2. Test model - `python test.py`
3. Serve - `tensorflow_model_server --port=9000 --model_name=inception --model_base_path=/home/joao/repos/TensorFlowTest/TransferLearning/export_model/ &> tfs_log &`
4. Boot Flask API `python ../Flask/main.py`
5. Test API `http://127.0.0.1:5000/predict` with `file: image`