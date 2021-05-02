__version__ = 'v1.0.0'

import argparse
import base64
from pathlib import Path

import numpy as np
import PIL.Image
import tempfile
import requests

import io
import os

import tensorflow as tf
import tensorflow_hub as hub

from flask import Flask, request, send_from_directory
from flask_cors import CORS, cross_origin

import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger('root')
FORMAT = '%(asctime)s - [%(levelname)7s (%(filename)15s:%(lineno)4s) - %(funcName)15s()]  %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)

os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

app = Flask(__name__)
CORS(app)


class StyleTransferService:

  def __init__(
      self,
      host='0.0.0.0',
      port=7000,
      jwt='SUPERsecret',
      ssl_context=None,
      use_gunicorn=False):

    logger.info('Starting...')

    self.host = host
    self.port = port
    self.jwt = jwt
    self.ssl_context = ssl_context
    self.use_gunicorn = use_gunicorn
    self.model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

  @staticmethod
  def _tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
      assert tensor.shape[0] == 1
      tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

  @staticmethod
  def _download_img(url):
    header = {'User-Agent': 'Mozilla/5.0 (Windows NT x.y; Win64; x64; rv:9.0) Gecko/20100101 Firefox/10.0'}
    r = requests.get(url, headers=header, allow_redirects=True)
    return base64.b64encode(r.content).decode('utf-8')

  def load_img(self, b64_img_str):
    max_dim = 512
    with tempfile.NamedTemporaryFile() as temp_file:
      if 'https://' in b64_img_str or 'http://' in b64_img_str:
        b64_img_str = self._download_img(b64_img_str)
      temp_file.write(base64.b64decode(b64_img_str))
      temp_file.seek(0)
      img = tf.io.read_file(temp_file.name)
      img = tf.image.decode_image(img, channels=3)
      img = tf.image.convert_image_dtype(img, tf.float32)
      shape = tf.cast(tf.shape(img)[:-1], tf.float32)
      long_dim = max(shape)
      scale = max_dim / long_dim
      new_shape = tf.cast(shape * scale, tf.int32)
      img = tf.image.resize(img, new_shape)
      img = img[tf.newaxis, :]
    return img

  def is_authenticated(self, jwt):
    return self.jwt == jwt

  def serve(self):
    @app.route('/style-transfer', methods=['POST'])
    @cross_origin()
    def style_transfer():
      jwt = request.environ.get('HTTP_AUTHORIZATION', None)
      if self.is_authenticated(jwt):
        content = request.json.get('content', None)
        style = request.json.get('style', None)
        if content and style:
          content_image = self.load_img(content)
          style_image = self.load_img(style)
          stylized_image = self.model(tf.constant(content_image), tf.constant(style_image))[0]
          output_img = self._tensor_to_image(stylized_image)
          buffer = io.BytesIO()
          output_img.save(buffer, format="JPEG")
          return {'output': base64.b64encode(buffer.getvalue()).decode('utf-8')}, 200
        return {}, 503
      return {}, 403

    # Using same logger
    app.logger.handlers = logger.handlers
    app.logger.setLevel(logging.ERROR)
    app.config.update(SESSION_COOKIE_NAME='lsi_charts_{}'.format(self.port))

    if self.use_gunicorn:
      s = 'SOME!@#$SECRET-{}{}{}'.format(__version__, self.host, self.port)
      app.config.update(SECRET_KEY=s.encode('utf-8'))
      return app

    # Running Flask App...
    app.config.update(SECRET_KEY=os.urandom(24))
    app.run(
      debug=False,
      host=self.host,
      port=self.port,
      use_reloader=False,
      threaded=True,
      passthrough_errors=True,
      ssl_context=self.ssl_context)


def setup(use_gunicorn=False):
  if not use_gunicorn:
    parser = argparse.ArgumentParser()
    parser.add_argument('--host',
                        type=str,
                        default=os.environ.get('STYLE_TRANSFER_HOST', '0.0.0.0'),
                        help='Service host.')
    parser.add_argument('--port',
                        type=int,
                        default=os.environ.get('STYLE_TRANSFER_PORT', 7000),
                        help='Service port.')
    parser.add_argument('--jwt',
                        type=str,
                        default=os.environ.get('STYLE_TRANSFER_SECRET', 'SUPERsecret'),
                        help='Service JWT secret.')
    parser.add_argument('--cert',
                        type=str,
                        default=os.environ.get('STYLE_TRANSFER_CERT', ''),
                        help='Path to certificate file.')
    parser.add_argument('--certkey',
                        type=str,
                        default=os.environ.get('STYLE_TRANSFER_CERTKEY', ''),
                        help='Path to cert key.')
    parser.add_argument('--log',
                        type=str,
                        default=os.environ.get('STYLE_TRANSFER_LOGFILE', None),
                        help='Log file.')
    args = parser.parse_args()

    ssl_context = None
    if os.path.exists(args.cert) and os.path.exists(args.certkey):
      ssl_context = (args.cert, args.certkey)

    if args.log:
      log_dir = Path(args.log).absolute().parent
      if not os.path.exists(log_dir):
        os.makedirs(log_dir)
      handler = RotatingFileHandler(args.log, maxBytes=20 * 1024 * 1024, backupCount=5)
      handler.setLevel(logging.INFO)
      handler.setFormatter(logging.Formatter(FORMAT))
      logger.addHandler(handler)

    service = StyleTransferService(
      host=args.host,
      port=args.port,
      jwt=args.jwt,
      ssl_context=ssl_context,
      use_gunicorn=use_gunicorn)

  else:
    cert = os.environ.get('STYLE_TRANSFER_CERT', '')
    certkey = os.environ.get('STYLE_TRANSFER_CERTKEY', '')
    log_file = os.environ.get('STYLE_TRANSFER_LOGFILE', None)

    ssl_context = None
    if os.path.exists(cert) and os.path.exists(certkey):
      ssl_context = (cert, certkey)

    if log_file:
      log_dir = Path(log_file).absolute().parent
      if not os.path.exists(log_dir):
        os.makedirs(log_dir)
      handler = RotatingFileHandler(log_file, maxBytes=20 * 1024 * 1024, backupCount=5)
      handler.setLevel(logging.INFO)
      handler.setFormatter(logging.Formatter(FORMAT))
      logger.addHandler(handler)

    service = StyleTransferService(
      host=os.environ.get('STYLE_TRANSFER_HOST', '0.0.0.0'),
      port=os.environ.get('STYLE_TRANSFER_PORT', 7000),
      jwt=os.environ.get('STYLE_TRANSFER_SECRET', 'SUPERsecret'),
      ssl_context=ssl_context,
      use_gunicorn=use_gunicorn)
  return service


def run():
  service = setup(use_gunicorn=True)
  return service.serve()


if __name__ == '__main__':
  control = setup()
  control.serve()
