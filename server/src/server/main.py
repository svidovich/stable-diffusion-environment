import json
import os
from _thread import LockType
from base64 import b64decode, b64encode
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import datetime
from http import HTTPStatus
from json import JSONDecodeError
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Lock
from typing import Any, Self
from uuid import UUID, uuid4

import torch
from diffusers import DiffusionPipeline
from fastapi import FastAPI, HTTPException, Request
from loguru import logger
from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    ValidationError,
    field_serializer,
)
from typing_extensions import Annotated

MAX_THREADS = 1
MODEL_SMALL = "stabilityai/stable-diffusion-2-1-unclip-small"


def get_default_disk_flush_directory(ensure_directory: bool = True) -> Path:
    """
    Get the default directory where we'll cache generation results on-disk
    """
    server_directory = Path(__file__).parent
    default_directory = server_directory / "disk_backups"
    if ensure_directory and not default_directory.is_dir():
        os.makedirs(default_directory, exist_ok=True)
    return default_directory


def get_default_disk_flush_filepath(ensure_directory: bool = True) -> Path:
    """
    Get a disk flush filepath based on the current time
    """
    default_directory = get_default_disk_flush_directory(
        ensure_directory=ensure_directory
    )
    date_format = "%Y-%m-%d-%H-%M-%S"
    return (
        default_directory
        / f"{datetime.now().strftime(date_format)}-sde-server-flush.json"
    )


def model_cache_dir(model_name: str, ensure_directory: bool = True) -> Path:
    """
    Get the cache dir for the model name.
    """
    server_directory = Path(__file__).parent
    model_cache_directory = server_directory / "model_cache" / model_name
    if ensure_directory and not model_cache_directory.is_dir():
        os.makedirs(model_cache_directory, exist_ok=True)

    return model_cache_directory


pipeline: DiffusionPipeline | None = None


def load_pipeline(model_name: str) -> DiffusionPipeline:
    """
    Get a singleton DiffusionPipeline.
    """
    global pipeline
    if not pipeline:
        pipeline = DiffusionPipeline.from_pretrained(
            model_name, torch_dtype=torch.float16
        )
        assert pipeline, "Couldn't get the pipeline! Ahh!"
        # looks like typing on this library sucks
        pipeline.to("cuda")  # type: ignore[attr-defined]
    return pipeline


global_pool: ThreadPool | None = None


def get_pool() -> ThreadPool:
    """
    Get a global, singleton, ThreadPool for use in the app.
    """
    global global_pool
    if not global_pool:
        global_pool = ThreadPool(processes=MAX_THREADS)
    return global_pool


def pool_available(pool: ThreadPool) -> bool:
    """
    Based on the number of jobs the pool is working on and the number of processes it has,
    is the pool available?
    """
    if len(pool._cache) < pool._processes:  # type: ignore[attr-defined]
        return True
    return False


def async_pool_execution(pool: ThreadPool, function: Callable, **kwargs) -> None:
    """
    Fire the function execution into the pool and forget about it.
    """
    pool.apply_async(func=function, kwds={**kwargs})


def load_base64(value: Any) -> bytes:
    """
    Try to load base64 from whatever you throw its way. If it's already bytes, just gives back bytes.
    Raises ValueError if it's not bytes or a string.
    Raises something interesting if it's an invalid base64 string.
    """
    if isinstance(value, bytes):
        return value
    if not isinstance(value, str):
        raise ValueError("Input should be a base64 encoded string.")
    return b64decode(s=value)


def load_datetime_from_isoformat(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        raise ValueError("Input should be an ISO formatted date string.")
    return datetime.fromisoformat(value)


class GenerationResult(BaseModel):
    """
    The result of generating an image with stable diffusion.
    """

    uuid: UUID
    prompt: str
    data: Annotated[bytes, BeforeValidator(load_base64)]
    created_time: Annotated[datetime, BeforeValidator(load_datetime_from_isoformat)] = (
        Field(default_factory=datetime.now)
    )
    errors: list[str] = Field(default_factory=list)

    @field_serializer("created_time")
    def serialize_created_time(self, created_time: datetime) -> str:
        """
        Dump datetime to isoformat for JSON formatting.
        """
        return created_time.isoformat()

    @field_serializer("uuid")
    def serialize_uuid(self, uuid: UUID) -> str:
        """
        Dump UUIDs to strings for JSON formatting.
        """
        return str(uuid)

    @field_serializer("data")
    def serialize_data(self, data: bytes) -> str:
        """
        Dump bytes to base64 for JSON formatting.
        """
        return b64encode(data).decode("utf-8")


class GenerationOutputs(BaseModel):
    """
    A container for generation outputs, shared by the app, with a built-in lock and
    functions for loading to disk.
    """

    class Config:
        """
        All to appease pydantic with regards to locks
        """

        arbitrary_types_allowed = True

    # Possibly the most annoying type annotation I've seen so far
    lock: LockType = Field(default_factory=Lock)
    computations: dict[UUID, GenerationResult]

    def query_by_uuid(self, uuid: UUID) -> GenerationResult | None:
        """
        Get the computation with the input UUID if we have it, or None if we don't.
        """
        return self.computations.get(uuid)

    def to_list(self) -> list[GenerationResult]:
        """
        Return all of the current generation results as a list.
        """
        return list(self.computations.values())

    def add_generation_result(self, result: GenerationResult) -> Self:
        """
        Add a generation result to myself, and return myself.
        """
        # Oh please kill me.
        with self.lock:  # type: ignore[attr-defined] # pylint: disable=no-member
            self.computations[result.uuid] = result

        return self

    def flush_to_disk(self, path: Path | None = None) -> None:
        """
        Dump the outputs to disk as a JSON array of objects.
        """
        if self.computations:
            path = path or get_default_disk_flush_filepath(ensure_directory=True)
            cache_directory = path.parent
            extant_cache_files = [
                cache_directory / cache_file
                for cache_file in os.listdir(cache_directory)
            ]

            logger.info(f"Flushing generation outputs to `{path}` ...")
            results = [entry.model_dump() for entry in self.computations.values()]
            with open(path, "w") as file_handle:
                json.dump(results, file_handle)
            logger.info("Flushed generation outputs.")
            logger.info("Dropping old cache files ...")
            # Dropping old cache files can help us consolidate our cache.
            # Only do this once we've saved the current cache.
            for extant_cache_file in extant_cache_files:
                extant_cache_file.unlink()
        else:
            logger.info("Would've written computations, but nothing to write!")

    @classmethod
    def load_from_disk(cls, cache_directory: Path | None = None) -> Self:
        """
        Attempt to load generation outputs from disk as a JSON array of objects, if I can.
        """
        cache_directory = cache_directory or get_default_disk_flush_directory()
        logger.info(
            f"Attempting to load cached generation outputs from `{cache_directory}` ..."
        )
        loaded_entries: dict[UUID, GenerationResult] = {}
        for dirent in os.listdir(cache_directory):
            dirent_path = cache_directory / dirent
            if dirent_path.is_file():
                logger.info(f"Found potential cache file at `{dirent_path}`.")
                try:
                    with open(dirent_path, "r") as file_handle:
                        raw = json.load(file_handle)
                    if isinstance(raw, list):
                        logger.info(
                            f"Attempting to load {len(raw)} entries from cache file at `{dirent_path}` ..."
                        )
                        for candidate in raw:
                            loaded_entry = GenerationResult.model_validate(candidate)
                            loaded_entries[loaded_entry.uuid] = loaded_entry
                    else:
                        logger.info(
                            f"Malformed cache file at `{dirent_path}`. Skipping ahead."
                        )
                        continue
                except (JSONDecodeError, ValidationError) as exc:
                    logger.error(
                        f"Failed to load cache file at `{dirent_path}`: {exc}. Skipping ahead."
                    )
        return cls(computations=loaded_entries)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    A lifespan for the app, accessible through request.state.
    """
    pool = get_pool()
    # TODO make this ... non-mandatory
    generation_outputs = GenerationOutputs.load_from_disk()
    yield {"threadpool": pool, "outputs": generation_outputs}
    pool.terminate()
    generation_outputs.flush_to_disk()


app = FastAPI(lifespan=lifespan)


class ImageGenerationBody(BaseModel):
    """
    The shape of a POST body for generating images.
    """

    prompt: str


class ImageGenerationResponse(BaseModel):
    """
    The shape of the response when you ask to generate an image.
    """

    generation_uuid: str


def prompt_stable_diffusion(uuid: UUID, prompt: str) -> GenerationResult:
    """
    Give the prompt to stable diffusion, bringing back the generated image.
    """
    logger.info(f"Beginning computation of prompt with ID `{uuid}`.")
    # pipeline = load_pipeline(model_name=MODEL_SMALL)
    result = GenerationResult(uuid=uuid, prompt=prompt, data=b"deadbeef", errors=[])
    logger.info(f"Completed generation for prompt with ID `{uuid}`!")
    return result


def generate_image(
    uuid: UUID, prompt: str, generation_outputs: GenerationOutputs
) -> None:
    """
    Generate an image with the given prompt. Add it to the given
    GenerationOutputs under the provided UUID.
    """
    logger.info(f"Received a prompt! `{prompt}`")
    generation_outputs.add_generation_result(
        result=prompt_stable_diffusion(uuid=uuid, prompt=prompt)
    )
    logger.info(f"Result with UUID `{uuid}` added!")


@app.post("/image/generate")
async def route_generate_image(
    post_body: ImageGenerationBody, request: Request
) -> ImageGenerationResponse:
    """
    Given the prompt in the post body, generate the image with stable diffusion.
    Return the UUID by which this generation request can be referenced.
    If there's not enough compute, tell the caller to go away.
    """
    if pool_available(pool=request.state.threadpool):
        generation_uuid = uuid4()
        async_pool_execution(
            pool=request.state.threadpool,
            function=generate_image,
            uuid=generation_uuid,
            prompt=post_body.prompt,
            generation_outputs=request.state.outputs,
        )
        return ImageGenerationResponse(generation_uuid=str(generation_uuid))
    raise HTTPException(
        status_code=HTTPStatus.TOO_MANY_REQUESTS, detail="The service is busy. Go away."
    )


@app.get("/image/generate")
async def route_show_generations(
    request: Request,
    generation_uuid: UUID | None = None,
) -> list[GenerationResult]:
    """
    Get the currently available generations. Optionally provide a UUID to get
    just the one.
    """
    results = request.state.outputs
    if generation_uuid:
        generation_result = results.query_by_uuid(generation_uuid)
        if generation_result:
            return [generation_result]
        return []
    return results.to_list()
