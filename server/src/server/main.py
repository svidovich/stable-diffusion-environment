"""
The main entrypoint for the stable-diffusion-environment server.
Use run.sh or something. Uvicorn works, too.
"""

# pylint: disable = global-statement,protected-access,broad-exception-caught
# I use the global statement to help define singletons.
# I use protected access in this module to deal with the current worker count in a given
# ThreadPool instance and its overall worker count, which I don't think are exposed in
# a more transparent way.
# I catch broad exceptions to ensure the uptime of the server. Exceptions are reported
# back through the API and logging.

import io
import os
from _thread import LockType
from base64 import b64decode, b64encode
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import datetime
from enum import StrEnum, auto
from http import HTTPStatus
from json import JSONDecodeError
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Lock
from typing import Any, Self
from uuid import UUID, uuid4

import jsonlines
import torch
from diffusers import DiffusionPipeline
from fastapi import FastAPI, HTTPException, Request
from loguru import logger
from PIL.Image import Image
from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    ValidationError,
    field_serializer,
)
from typing_extensions import Annotated

MAX_THREADS = 1
MODEL_SMALL = "stabilityai/stable-diffusion-2-1"


class MachineSize(StrEnum):
    """
    Counter strike Beefy Computer?
    Or StrongBad's Lappy?
    """

    BIG = auto()
    MEDIUM = auto()
    LITTLE = auto()
    BABY = auto()


DEFAULT_MACHINE_SIZE = MachineSize.LITTLE


class GenerationOptions(BaseModel):
    """
    Options for using SD pipelines depending on our resource constraints.
    """

    size: MachineSize
    enable_attention_slicing: bool
    enable_sequential_cpu_offload: bool
    image_height: int  # TODO needs to be a multiple of 64. Write a validator.
    image_width: int  # TODO needs to be a multiple of 64. Write a validator.


GENERATION_DEFAULTS_LIST = [
    GenerationOptions(
        size=MachineSize.BIG,
        enable_attention_slicing=False,
        enable_sequential_cpu_offload=False,
        image_height=1024,
        image_width=1024,
    ),
    GenerationOptions(
        size=MachineSize.MEDIUM,
        enable_attention_slicing=True,
        enable_sequential_cpu_offload=False,
        image_height=512,
        image_width=512,
    ),
    GenerationOptions(
        size=MachineSize.LITTLE,
        enable_attention_slicing=True,
        enable_sequential_cpu_offload=False,
        image_height=384,
        image_width=384,
    ),
    GenerationOptions(
        size=MachineSize.BABY,
        enable_attention_slicing=True,
        enable_sequential_cpu_offload=True,
        image_height=256,
        image_width=256,
    ),
]

GENERATION_DEFAULTS = {entry.size: entry for entry in GENERATION_DEFAULTS_LIST}


def generation_defaults(machine_size: MachineSize) -> GenerationOptions:
    """
    Get the default image generation options for the input machine size.
    """
    return GENERATION_DEFAULTS[machine_size]


def generation_options_from_environment() -> GenerationOptions:
    """
    Attempt to get our generation options from the environment. Failing that, get sensible
    defaults.
    """
    if env_machine_size := os.environ.get("MACHINE_SIZE"):
        try:
            return generation_defaults(machine_size=MachineSize(env_machine_size))
        except ValueError:
            logger.error(
                f"Invalid MACHINE_SIZE `{env_machine_size}`; defaulting to "
                f"`{DEFAULT_MACHINE_SIZE}`."
            )
            return generation_defaults(machine_size=DEFAULT_MACHINE_SIZE)
    else:
        logger.info(
            f"Didn't find MACHINE_SIZE in the environment; defaulting to "
            f"`{DEFAULT_MACHINE_SIZE}`."
        )
        return generation_defaults(machine_size=DEFAULT_MACHINE_SIZE)


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


def load_pipeline(model_name: str, options: GenerationOptions) -> DiffusionPipeline:
    """
    Get a singleton DiffusionPipeline.
    """
    global pipeline
    if not pipeline:
        pipeline = DiffusionPipeline.from_pretrained(
            model_name,
            cache_dir=model_cache_dir(model_name=model_name),
        )
        assert pipeline, "Couldn't get the pipeline! Ahh!"
        # The typing for this library is very bad. They could use mypy in their system.
        # It's pooching my code. booo.
        if options.enable_attention_slicing:
            logger.info("NOTE: Enabling attention slicing due to settings.")
            # This helps us save some memory on a constrained system
            pipeline.enable_attention_slicing()  # type: ignore[attr-defined]
        if options.enable_sequential_cpu_offload:
            logger.info("NOTE: Enabling sequential CPU offload due to settings.")
            # Generally we do this when we're super short on resources. Not for the
            # faint of heart
            pipeline.enable_sequential_cpu_offload()  # type: ignore[attr-defined]
            # Note that we don't pipeline _to_ anything: we defer that command to the
            # library, which will handle it on its own. We can get goofy errors if we
            # attempt to do our own piping.
        else:
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


def load_base64(value: Any) -> bytes | None:
    """
    Try to load base64 from whatever you throw its way. If it's already bytes, just
    gives back bytes.
    Raises ValueError if it's not bytes or a string.
    Raises something interesting if it's an invalid base64 string.
    """
    if value is None:
        return value
    if isinstance(value, bytes):
        return value
    if not isinstance(value, str):
        raise ValueError("Input should be a base64 encoded string.")
    return b64decode(s=value)


def load_datetime_from_isoformat(value: Any) -> datetime:
    """
    Try to load a datetime from whatever you throw its way. It's expecting an ISO
    formatted date string. If it's already a datetime, it'll hand it back to you.
    Raises ValueError if you pass it garbage.
    """
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
    data: Annotated[bytes | None, BeforeValidator(load_base64)]
    created_time: Annotated[datetime, BeforeValidator(load_datetime_from_isoformat)] = (
        Field(default_factory=datetime.now)
    )
    errors: list[str] = Field(default_factory=list)

    def add_error(self, error: str) -> None:
        """
        Add an error to the list of errors.
        """
        # Pylint can't see as deeploy as mypy.
        self.errors.append(error)  # pylint: disable=no-member

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
    def serialize_data(self, data: bytes | None) -> str | None:
        """
        Dump bytes to base64 for JSON formatting.
        """
        if data is not None:
            return b64encode(data).decode("utf-8")
        return None


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
        self.flush_to_disk()
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
            with jsonlines.open(path, "w") as writer:
                writer.write_all(results)
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
                    with jsonlines.open(dirent_path, "r") as reader:
                        for entry in reader.iter(type=dict, skip_invalid=True):
                            loaded_entry = GenerationResult.model_validate(entry)
                            loaded_entries[loaded_entry.uuid] = loaded_entry
                except (JSONDecodeError, ValidationError) as exc:
                    logger.error(
                        f"Failed to load cache file at `{dirent_path}`: {exc}. Skipping ahead."
                    )
        return cls(computations=loaded_entries)


@asynccontextmanager
async def lifespan(app_: FastAPI):
    """
    A lifespan for the app, accessible through request.state. With it, you get...
    - request.state.pool: ThreadPool - one-man-army ThreadPool for generating images
    - request.state.outputs: GenerationOutputs - the shared object for storing the results
      of image generation
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


def prompt_stable_diffusion(
    uuid: UUID, prompt: str, options: GenerationOptions
) -> GenerationResult:
    """
    Give the prompt to stable diffusion, bringing back the generated image.
    """
    logger.info(f"Beginning computation of prompt with ID `{uuid}`.")
    result = GenerationResult(uuid=uuid, prompt=prompt, data=None, errors=[])
    try:
        sd_pipeline = load_pipeline(model_name=MODEL_SMALL, options=options)
        # According to the comments in pipeline_utils.py from diffusers.pipelines,
        # here images is a List[PIL.Image.Image]. Then, ideally generated_image results as an
        # image from PIL, and we can just convert it to bytes.
        # Mypy thinks it's not callable. But, the resultant DiffusionPipeline from load_pipeline
        # _should_ be ... maybe.
        with torch.inference_mode():  # Disables some training-time settings to save resources.
            generated_image: Image = sd_pipeline(  # type: ignore[operator]
                prompt, height=options.image_height, width=options.image_width
            ).images[0]
        # A buffer where we can dump our image. No, you can't just say tobytes().
        image_buffer = io.BytesIO()
        generated_image.save(image_buffer, format="PNG")
        result.data = image_buffer.getvalue()

        logger.info(f"Completed generation for prompt with ID `{uuid}`!")
    except Exception as exc:
        logger.exception(
            "Something awful happened when attempting to prompt and get an image back!"
        )
        result.add_error(str(exc))
    return result


def generate_image(
    uuid: UUID, prompt: str, generation_outputs: GenerationOutputs
) -> None:
    """
    Generate an image with the given prompt. Add it to the given
    GenerationOutputs under the provided UUID.
    """
    logger.info(f"Received a prompt! `{prompt}`")
    options = generation_options_from_environment()
    generation_outputs.add_generation_result(
        result=prompt_stable_diffusion(uuid=uuid, prompt=prompt, options=options)
    )
    logger.info(f"Result with UUID `{uuid}` added!")


class PoolAvailableResponse(BaseModel):
    """
    Is the server ready to generate images?
    """

    is_available: bool


@app.get("/server/status")
async def route_pool_available(request: Request) -> PoolAvailableResponse:
    """
    Determine whether the server is ready to generate images or not.
    """
    return PoolAvailableResponse(
        is_available=pool_available(pool=request.state.threadpool)
    )


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
