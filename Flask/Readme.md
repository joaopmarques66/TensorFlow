# Readme

# DNS
` kookapp.ml `


# Server config
1. `sudo apt-get update`
1. `sudo apt-get upgrade`
1. `sudo apt-get install nginx`
2. `sudo /etc/init.d/nginx start`
3. `sudo chown -R www-data:www-data /var/www/`
4. `sudo usermod -g www-data joao`
5. `sudo chmod 775 -R /var/www/`
6. `git clone https://github.com/joaopmarques66/TensorFlow.git`
7. `virtualenv --system-site-packages -p python3 ./venv`
8. `source ./venv/bin/activate`
9. `pip install --upgrade pip`
10. `pip install Flask`
11. `pip install Flask-API`
12. `pip install tensorflow`
13. `pip install tensorflow_hub`
14. `pip install Pillow`
15. `pip install tensorflow-serving-api`
16. `sudo apt-get install build-essential python python-dev`
17. `sudo apt-get install build-essential python3-dev`


`pip install uwsgi`
`sudo rm /etc/nginx/sites-enabled/default`

```
server {
    listen      3000;
    server_name http://kookapp.ml/;
    charset     utf-8;
    client_max_body_size 75M;

    location / { try_files $uri @kookapp; }
    location @kookapp {
        include uwsgi_params;
        uwsgi_pass unix:/var/www/TensorFlow/Flask/demoapp_uwsgi.sock;
    }    
}
```

`sudo ln -s /etc/nginx/sites-available/kookapp.conf /etc/nginx/sites-enabled/kookapp.conf`


`sudo mkdir -p /var/log/uwsgi`
`sudo chown www-data:www-data /var/log/uwsgi`
`sudo chmod 775 /var/log/uwsgi`

`uwsgi --ini /var/www/TensorFlow/Flask/demoapp_uwsgi.ini`



`/etc/systemd/system/foodId.service`
```
[Unit]
Description="uWSGI server instance for my_app"
After=network.target

[Service]
ExecStart=/var/www/TensorFlow/venv/bin/uwsgi --ini /var/www/TensorFlow/Flask/demoapp_uwsgi.ini

[Install]
WantedBy=multi-user.target

```

`/etc/systemd/system/tensorFlowServe.service`
```
[Unit]
Description="TensorFlow serve model"
After=network.target

[Service]
ExecStart=/var/www/TensorFlow/venv/bin/uwsgi --ini /var/www/TensorFlow/Flask/demoapp_uwsgi.ini

[Install]
WantedBy=multi-user.target
```
#Install tensorflow-model-serve

```
echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list
curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -
sudo apt-get update && sudo apt-get install tensorflow-model-server
```
`/etc/systemd/system/tensorFlowServe.service`
```
[Unit]
Description="TensorFlow serve model"
After=network.target

[Service]
ExecStart=/usr/bin/tensorflow_model_server --port=9000 --model_name=inception --model_base_path=/var/www/TensorFlow/TransferLearning/export_model/

[Install]
WantedBy=multi-user.target
```

`sudo systemctl daemon-reload`
`sudo systemctl enable tensorFlowServe`
`sudo systemctl enable foodId`
