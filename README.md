# simple-serving

Text model serving for [simple-story-chat](https://github.com/jointsome0-lgtm/simple-story-chat): a FastAPI gateway in
front of vLLM, run on rented GPUs. The bot calls it over HTTP. Outside clients with a key may call it too.

Status: design. The API is in [docs/contract-v1.md](docs/contract-v1.md); nothing is implemented yet.

Version 1 serves one text model. Pictures stay in simple-story-chat for now.
