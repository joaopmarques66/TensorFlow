### Rename all files to sequential order on linux
`ls | cat -n | while read n f; do mv "$f" `printf "%04d.jpg" $n`; done`


## Pipeline
1. Train model - `python train.py`
2. Test model - `python test.py`
3. Serve - `tensorflow_model_server --port=9000 --model_name=inception --model_base_path=/home/joao/repos/TensorFlowTest/TransferLearning/export_model/ &> tfs_log &`
4. Boot Flask API `python ../Flask/main.py`
5. Test API `http://127.0.0.1:5000/predict` with `file: image`


## Convert to TFLite
```
tflite_convert \
  --output_file=/tmp/model.tflite \
  --graph_def_file=./export_model/saved_model.pb \
  --input_arrays=input_image \
  --output_arrays=MobilenetV1/Predictions/Reshape_1
```