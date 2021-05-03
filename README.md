# Algorand Style Transfer Backend

This is a simple REST API (Flask) that exposes an AI Style Transfer model.

## Run

```shell script
git clone https://github.com/03303/algo-style-transfer.git
cd algo-style-transfer
docker build . -t tensorflow_flask
docker run -p 7000:7000 -dti tensorflow_flask
```

## References:
- [Tensorflow's Neural style transfer](https://www.tensorflow.org/tutorials/generative/style_transfer)
