# AI REMBG STREAMLIT

## Interface example

![image](https://i.imgur.com/0YXJtET.jpg)


## Local start

### Build the docker image

```bash
$ docker build -t ai_rembg_streamlit -f Dockerfile .
```

### Run the docker container

```bash
$ docker run -p 8081:8501 ai_rembg_streamlit 
```