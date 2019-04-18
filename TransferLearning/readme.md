### Rename all files to sequential order on linux
`ls | cat -n | while read n f; do mv "$f" `printf "%04d.jpg" $n`; done`


## Train graph
`python retrain.py \
        --saved_model_dir './export_model' \
        --image_dir /home/joao/repos/FoodDataset/images \
        --tfhub_module https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1 `
## Freeze

`freeze_graph \
  --input_saved_model_dir .../TransferLearning/export_model/  \
  --input_checkpoint .../TransferLearning/export_model/variables/variables \
  --input_binary true  \
  --output_graph ./frozen.pb \
  --output_node_names final_result \
    --clear_devices`

## To TFLite
//not working
` tflite_convert \
  --output_file= ./model.tflite \
  --input_arrays=image \
  --saved_model_dir=./export_model \
  --output_arrays=final_result ` 


## Serve

`tensorflow_model_server --port=9000 --model_name=inception --model_base_path=/home/joao/repos/TensorFlowTest/TransferLearning/experimental/Inceptionv3-1_epochs-numclasses_32`

### test 
`python test_serve.py -i ~/repos/FoodDataset/testing/pizza/2.jpg`