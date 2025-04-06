# Stable Diffusion Environment

A quick and dirty client / server model for doing Stable Diffusion computations remotely.

## Why

I'm trying to learn to at least be a real, solid end-user of AI models. I'm also really sick of using `tmux` and the executable from `stable-diffusion.cpp`. It's ungainly.

Also, I don't want to use ComfyUI. I don't want this to be easy. I want to understand what I'm doing.

So, let's do it.

## Structure

- `server` contains what you'll run on the machine that will be doing the generations.
- `client` contains a dirty front-end client that makes calls to the server.

### Server Setup

The server is a Python 3.12 poetry project. You should ensure that you have both installed. I suggest using [pyenv](https://github.com/pyenv/pyenv) with [pipx](https://github.com/pypa/pipx), e.g.

First time tooling setup:

```shell
pyenv install 3.12.9
pyenv global 3.12.9
# restart your shell.
apt install pipx && pipx ensurepath
# restart your shell.
pipx install poetry --python=$(command -v python)
```

Now we can jump down into the `server` directory, and

```shell
poetry install
./run.sh
```

`run.sh` just runs `uvicorn`, so you can feasibly also say

```shell
poetry install
poetry run uvicorn \
    --reload \
    --host 0.0.0.0 \
    --port 9090 \
    --app-dir src/server/ main:app
```

Indeed, the default port for the server is `9090`.

### Server Details and Endpoints

\*\*NOTE: Currently this thing only uses "stabilityai/stable-diffusion-2-1". This was just for testing. We'll use different models going forward."

The server runs on a FastAPI backed by a single-worker `ThreadPool`. You can generate one image at a time, and then poll for the results while you wait. If the pool isn't available for generation, you'll get a 'too many requests' back from the generation endpoint.

You can tell the server roughly what size your machine is with an environment variable, `MACHINE_SIZE`, for which you can choose:

- `big`
- `medium`
- `little`
- `tiny`

If you have a big computer, you know it. Tiny computers are like... stinkpads and stuff with barely any GPU strength. YMMV. The smaller you go, the more enabling of things like attention slicing and CPU offload will happen.

The server will cache models and generation results to disk. It stores the model cache in the server directory adjacent to `main.py` under `model_cache`. The first generation request you make to the server, then, may take a long time, as it's downloading an _entire stable diffusion model_. Be patient. ( Might add an init endpoint or something? Or a script? idk. We'll see )

Generation outputs are cached to disk as jsonl files in the directory `disk_backups` _also_ adjacent to `main.py`. We're not doing a database ( yet ). No need. Even SQLite would be too much here, haha. Output cache files are consolidated... often. Go read the source code for details.

Endpoint overview:

`GET /server/status`: Determine whether the server is available for image generation at the moment. response:

```
{
    "is_available": true
}

or

{
    "is_available": false
}
```

`POST /image/generate`: Generate an image using Stable Diffusion. Requires a JSON post body of the following form:

```
{
    "prompt": "The string you want to use as a prompt during generation."
}
```

If the server is available for computation at request time, it returns a UUID thus:

```
{
    "generation_uuid": "... a uuid ..."
}
```

If it's not available at request time, you'll get a 429 back. Come back later dork.

`GET /image/generate?generation_uuid=...`: Show image generation results as an array of JSON objects of the following form:

```
[
    {
        "uuid": "... the image generation UUID from POST /image/generate ...",
        "prompt": "... the prompt used in generation, a string ...",
        "created_time": "... ISO Formatted date string of the creation time of this generation ...",
        "errors": [
            "Free-form strings",
            "Describing errors encountered
        ],
        "data": "... base64 encoded PNG data, the output of image generation. Can be null if there were errors."
    }
]
```

If `generation_uuid` is provided as a query parameter, it'll only return the generation result for that UUID. It's the UUID from the response of `POST /image/generate`.
