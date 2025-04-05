# Stable Diffusion Environment

A quick and dirty client / server model for doing Stable Diffusion computations remotely.

## Why

I'm trying to learn to at least be a real, solid end-user of AI models. I'm also really sick of using `tmux` and the executable from `stable-diffusion.cpp`. It's ungainly.

Also, I don't want to use ComfyUI. I don't want this to be easy. I want to understand what I'm doing.

So, let's do it.

## Structure

`server` contains what you'll run on the machine that will be doing the generations.

`client` contains a dirty front-end client that makes calls to the server.
