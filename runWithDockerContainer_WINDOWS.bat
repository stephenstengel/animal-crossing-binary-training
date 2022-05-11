docker build -t "animal-crossing-binary-training" .
docker run --gpus all --rm -v %cd%:/app/ -v %cd%\..\animal-crossing-loader\:/animal-crossing-loader/ -w /app/ animal-crossing-binary-training python3 train-model.py 
