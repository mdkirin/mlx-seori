import argparse
import asyncio
import gc
import json
import logging
import logging.handlers
import os
import re
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Empty as QueueEmpty
from queue import Queue
from threading import Event, Lock, Thread
from typing import Any, Callable, Iterator, List, Literal, Optional, Tuple, Union

logger = logging.getLogger("mlx_vlm.server")

import mlx.core as mx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from huggingface_hub import scan_cache_dir
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing_extensions import Required, TypeAlias, TypedDict

from .generate import (
    DEFAULT_KV_GROUP_SIZE,
    DEFAULT_KV_QUANT_SCHEME,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_PATH,
    DEFAULT_PREFILL_STEP_SIZE,
    DEFAULT_QUANTIZED_KV_START,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    BatchGenerator,
    PrefixCache,
    _dflash_rounds_batch,
    _make_cache,
    generate,
    normalize_resize_shape,
    stream_generate,
)
from .prompt_utils import apply_chat_template
from .sample_utils import top_p_sampling
from .tool_parsers import _infer_tool_parser, load_tool_module
from .utils import load, prepare_inputs
from .version import __version__
from .vision_cache import VisionFeatureCache

DEFAULT_SERVER_HOST = "0.0.0.0"
DEFAULT_SERVER_PORT = 8080


# ── Metal memory limits ──────────────────────────────────────
# Set via environment variables or CLI args (see main()).
# Without limits the MLX cache can consume all unified memory.

def _configure_metal_limits():
    """Apply Metal memory/cache limits from environment variables.

    Environment variables:
        MLX_MEMORY_LIMIT_GB: Hard limit on total Metal memory (default: 87.5% of total)
        MLX_CACHE_LIMIT_GB: Metal cache limit (default: total_limit // 7, min 4GB)
    """
    try:
        total_gb = mx.device_info().get("memory_size", 96 * 1024**3) / 1024**3
    except AttributeError:
        total_gb = mx.metal.device_info().get("memory_size", 96 * 1024**3) / 1024**3

    mem_limit = int(os.environ.get("MLX_MEMORY_LIMIT_GB", 0)) or int(total_gb * 0.875)
    cache_limit = int(os.environ.get("MLX_CACHE_LIMIT_GB", 0)) or max(4, mem_limit // 7)

    try:
        mx.set_memory_limit(mem_limit * 1024**3)
        mx.set_cache_limit(cache_limit * 1024**3)
    except AttributeError:
        mx.metal.set_memory_limit(mem_limit * 1024**3)
        mx.metal.set_cache_limit(cache_limit * 1024**3)

    print(f"Metal limits: memory={mem_limit}GB, cache={cache_limit}GB (hardware={total_gb:.0f}GB)")
    return mem_limit, cache_limit

_metal_mem_limit_gb, _metal_cache_limit_gb = _configure_metal_limits()


def get_prefill_step_size():
    return int(os.environ.get("PREFILL_STEP_SIZE", DEFAULT_PREFILL_STEP_SIZE))


def get_quantized_kv_bits(model: str):
    kv_bits = float(os.environ.get("KV_BITS", 0))
    if kv_bits == 0:
        return None
    if "qat" in model:
        print(f"Model {model} is quantization aware, KV cache will not be quantized.")
        return None
    return kv_bits


def get_kv_group_size():
    return int(os.environ.get("KV_GROUP_SIZE", DEFAULT_KV_GROUP_SIZE))


def get_kv_quant_scheme():
    return os.environ.get("KV_QUANT_SCHEME", DEFAULT_KV_QUANT_SCHEME)


def get_max_kv_size(model: str):
    max_kv_tokens = int(os.environ.get("MAX_KV_SIZE", 0))
    if max_kv_tokens == 0:
        return None
    if get_quantized_kv_bits(model) is not None:
        print(f"Model {model} uses QuantizedKVCache, can't set max KV size.")
        return None
    return max_kv_tokens


def get_quantized_kv_start():
    return int(os.environ.get("QUANTIZED_KV_START", DEFAULT_QUANTIZED_KV_START))


def get_top_logprobs_k():
    """Max per-token top_logprobs honored by the server (0 = disabled).

    Set via TOP_LOGPROBS_K env var. OpenAI caps this at 20. When 0, requests
    with top_logprobs>0 still succeed but the top_logprobs list stays empty.
    """
    k = int(os.environ.get("TOP_LOGPROBS_K", 0))
    return max(0, min(k, 20))


# =============================================================================
# ResponseGenerator - Concurrent Request Handling with Threaded Batching
# =============================================================================


@dataclass
class GenerationArguments:
    """Arguments for a generation request."""

    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    top_k: int = 0
    min_p: float = 0.0
    seed: Optional[int] = None
    repetition_penalty: Optional[float] = None
    logit_bias: Optional[dict] = None
    enable_thinking: bool = True
    thinking_budget: Optional[int] = None
    thinking_start_token: Optional[str] = None

    def to_generate_kwargs(self) -> dict:
        """Convert to kwargs dict for generate()/stream_generate()."""
        kw = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "enable_thinking": self.enable_thinking,
        }
        if self.repetition_penalty is not None:
            kw["repetition_penalty"] = self.repetition_penalty
        if self.logit_bias is not None:
            kw["logit_bias"] = self.logit_bias
        if self.thinking_budget is not None:
            kw["thinking_budget"] = self.thinking_budget
        if self.thinking_start_token is not None:
            kw["thinking_start_token"] = self.thinking_start_token
        return kw

    def to_template_kwargs(self) -> dict:
        """Convert to kwargs for apply_chat_template()."""
        kw = {"enable_thinking": self.enable_thinking}
        if self.thinking_budget is not None:
            kw["thinking_budget"] = self.thinking_budget
        if self.thinking_start_token is not None:
            kw["thinking_start_token"] = self.thinking_start_token
        return kw


@dataclass
class GenerationContext:
    """Context returned when a request is queued."""

    uid: int
    prompt_tokens: int


@dataclass
class StreamingToken:
    """A single token response during streaming generation."""

    text: str
    token: int
    logprobs: float
    finish_reason: Optional[str]
    peak_memory: float = 0.0
    top_logprobs: Optional[List[Tuple[int, float]]] = None


class ResponseGenerator:
    """
    Continuous batching for concurrent requests via a single GPU thread.

    A dedicated thread owns all GPU work (BatchGenerator). FastAPI async
    handlers submit requests to a queue and read tokens back from
    per-request queues. Multiple requests are batched together for
    higher throughput — same pattern as mlx-lm's server.
    """

    def __init__(
        self,
        model_path: str,
        adapter_path: Optional[str] = None,
        vision_cache=None,
        kv_bits=None,
        kv_group_size=DEFAULT_KV_GROUP_SIZE,
        kv_quant_scheme=DEFAULT_KV_QUANT_SCHEME,
        quantized_kv_start=DEFAULT_QUANTIZED_KV_START,
        top_logprobs_k=0,
    ):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.model = None
        self.processor = None
        self.config = None
        self.stop_tokens = set()
        self.vision_cache = vision_cache
        self.draft_model = None
        self.kv_bits = kv_bits
        self.kv_group_size = kv_group_size
        self.kv_quant_scheme = kv_quant_scheme
        self.quantized_kv_start = quantized_kv_start
        self.top_logprobs_k = top_logprobs_k
        self.tokenizer = None
        self.requests: Queue = Queue()
        self._stop = False
        self._ready = Event()
        self._load_error: Optional[Exception] = None
        self._cancelled: set = set()
        self._cancel_lock = Lock()
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop_and_join(self):
        self._stop = True
        self.requests.put(None)
        self._thread.join(timeout=5.0)

    def wait_until_ready(self, timeout: Optional[float] = None):
        if not self._ready.wait(timeout):
            raise RuntimeError("Timed out waiting for generation thread to load model.")
        if self._load_error is not None:
            raise self._load_error
        return self.model, self.processor, self.config

    def _cancel(self, uid):
        with self._cancel_lock:
            self._cancelled.add(uid)

    def _drain_cancellations(self) -> set:
        with self._cancel_lock:
            pending, self._cancelled = self._cancelled, set()
            return pending

    def _initialize_model(self):
        model, processor, config = load_model_resources(
            self.model_path, self.adapter_path
        )

        stop_tokens = set()
        if hasattr(config, "eos_token_id"):
            if isinstance(config.eos_token_id, list):
                stop_tokens.update(config.eos_token_id)
            elif config.eos_token_id is not None:
                stop_tokens.add(config.eos_token_id)

        draft_model = None
        draft_model_path = os.environ.get("MLX_VLM_DRAFT_MODEL")
        if draft_model_path:
            from .speculative.drafters import load_drafter

            draft_kind = os.environ.get("MLX_VLM_DRAFT_KIND", "dflash")
            print(f"Loading speculative drafter ({draft_kind}): {draft_model_path}")
            draft_model = load_drafter(draft_model_path, kind=draft_kind)
            print("Drafter ready — speculative decoding enabled.")

        self.model = model
        self.processor = processor
        self.config = config
        self.stop_tokens = stop_tokens
        self.draft_model = draft_model
        self.tokenizer = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )

    def generate(
        self,
        prompt: str,
        images: Optional[List] = None,
        audio: Optional[List] = None,
        args: Optional[GenerationArguments] = None,
    ) -> Tuple[GenerationContext, Iterator[StreamingToken]]:
        self.wait_until_ready()
        args = args or GenerationArguments()
        rqueue: Queue = Queue()

        # CPU preprocessing (tokenize, load images) on caller thread.
        # GPU work (vision encoder) deferred to GPU thread.
        raw_inputs = self._cpu_preprocess(prompt, images, audio)
        prompt_tokens = (
            raw_inputs["input_ids"].size
            if hasattr(raw_inputs["input_ids"], "size")
            else len(raw_inputs["input_ids"])
        )

        self.requests.put((rqueue, raw_inputs, prompt_tokens, args, images))

        # Block until the GPU thread sends back the context
        ctx = rqueue.get()
        if isinstance(ctx, Exception):
            raise ctx

        uid = ctx.uid

        def token_iterator():
            # Mark ended before yielding the final token so a consumer that
            # closes immediately after seeing finish_reason isn't treated
            # as a client abort.
            ended = False
            try:
                while True:
                    item = rqueue.get(timeout=60.0)
                    if item is None:
                        ended = True
                        break
                    if isinstance(item, Exception):
                        ended = True
                        raise item
                    if getattr(item, "finish_reason", None):
                        ended = True
                    yield item
                    if ended:
                        break
            finally:
                if not ended:
                    self._cancel(uid)

        return ctx, token_iterator()

    def _cpu_preprocess(self, prompt, images=None, audio=None) -> dict:
        """CPU-only: tokenize text, load/resize images. Thread-safe."""
        add_special_tokens = (
            getattr(self.processor, "chat_template", None) is None
            if self.model.config.model_type in ["gemma3", "gemma3n", "gemma4"]
            else True
        )
        image_token_index = getattr(self.model.config, "image_token_index", None)
        return prepare_inputs(
            self.processor,
            images=images,
            audio=audio,
            prompts=prompt,
            image_token_index=image_token_index,
            add_special_tokens=add_special_tokens,
        )

    # -- internals --

    def _make_sampler(self, args: GenerationArguments) -> Optional[Callable]:
        if args.temperature == 0:
            return None

        def sampler(logprobs: mx.array) -> mx.array:
            if args.top_p > 0 and args.top_p < 1.0:
                return top_p_sampling(logprobs, args.top_p, args.temperature)
            else:
                return mx.random.categorical(logprobs * (1 / args.temperature))

        return sampler

    def _gpu_embed(self, raw_inputs: dict, images=None) -> Tuple[mx.array, dict]:
        """GPU-only: run vision encoder if needed. Must run on GPU thread."""
        input_ids = raw_inputs.get("input_ids")
        pixel_values = raw_inputs.get("pixel_values")
        mask = raw_inputs.get("attention_mask")
        data_kwargs = {
            k: v
            for k, v in raw_inputs.items()
            if k not in ["input_ids", "pixel_values", "attention_mask"]
        }
        # Pass vision cache for image feature caching
        if (
            pixel_values is not None
            and self.vision_cache is not None
            and images is not None
        ):
            data_kwargs["vision_cache"] = self.vision_cache
            data_kwargs["_image_key"] = images

        # Always call get_input_embeddings — BatchGenerator requires inputs_embeds
        embed = self.model.get_input_embeddings(
            input_ids, pixel_values, mask=mask, **data_kwargs
        )
        # Remove cache kwargs before passing to BatchGenerator
        data_kwargs.pop("vision_cache", None)
        data_kwargs.pop("_image_key", None)
        gen_kwargs = {**data_kwargs, **embed.to_dict()}
        return input_ids, gen_kwargs

    def _run(self):
        """Single GPU thread: owns BatchGenerator, runs tight next() loop."""
        try:
            self._initialize_model()
        except Exception as e:
            self._load_error = e
            self._ready.set()
            print(f"Error loading model in generation thread: {e}")
            traceback.print_exc()
            return

        self._ready.set()

        if self.draft_model is not None:
            self._run_speculative()
            return

        generation_stream = mx.default_stream(mx.default_device())

        batch_gen = None
        # uid -> {rqueue, tokens, gen_kwargs}
        active: dict = {}

        while not self._stop:
            try:
                # Poll the request queue — non-blocking when generating, short
                # blocking wait when idle so we don't spin.
                new_items = []
                if active:
                    try:
                        item = self.requests.get_nowait()
                        if item is None:
                            if self._stop:
                                break
                        else:
                            new_items.append(item)
                    except QueueEmpty:
                        pass
                else:
                    try:
                        item = self.requests.get(timeout=0.1)
                        if item is None:
                            if self._stop:
                                break
                        else:
                            new_items.append(item)
                    except QueueEmpty:
                        pass

                while True:
                    try:
                        item = self.requests.get_nowait()
                        if item is not None:
                            new_items.append(item)
                    except QueueEmpty:
                        break

                # Drop abandoned requests before doing more work.
                cancelled = self._drain_cancellations()
                if cancelled and batch_gen is not None:
                    for uid in cancelled:
                        if uid in active:
                            batch_gen.remove(uid)
                            info = active.pop(uid)
                            try:
                                info["rqueue"].put(None)
                            except Exception:
                                pass

                for rqueue, raw_inputs, prompt_tokens, args, images in new_items:
                    if batch_gen is None:
                        batch_gen = BatchGenerator(
                            self.model.language_model,
                            self.processor,
                            stop_tokens=self.stop_tokens,
                            sampler=self._make_sampler(args),
                            kv_bits=self.kv_bits,
                            kv_group_size=self.kv_group_size,
                            kv_quant_scheme=self.kv_quant_scheme,
                            quantized_kv_start=self.quantized_kv_start,
                            top_logprobs_k=self.top_logprobs_k,
                            stream=generation_stream,
                        )

                    # Vision encoder runs on the GPU thread; text tokenization
                    # already happened on the caller thread.
                    input_ids, gen_kwargs = self._gpu_embed(raw_inputs, images)
                    has_embeds = bool(gen_kwargs.get("inputs_embeds") is not None)

                    # Image/embed requests can't share a prefill batch with
                    # pending text-only prompts — drain them first.
                    if has_embeds and batch_gen.unprocessed_prompts:
                        self._flush(batch_gen, active)

                    try:
                        (uid,) = batch_gen.insert(
                            [input_ids.squeeze(0).tolist()],
                            max_tokens=args.max_tokens,
                            prompt_kwargs=[gen_kwargs],
                        )
                    except Exception as e:
                        rqueue.put(e)
                        continue

                    rqueue.put(GenerationContext(uid=uid, prompt_tokens=prompt_tokens))
                    active[uid] = {
                        "rqueue": rqueue,
                        "tokens": [],
                        "prev_text": "",
                        "gen_kwargs": gen_kwargs if has_embeds else None,
                    }

                    if has_embeds:
                        self._step(batch_gen, active)

                if not active or batch_gen is None:
                    continue

                self._step(batch_gen, active)

            except Exception as e:
                print(f"Error in generation thread: {e}")
                traceback.print_exc()

    def _run_speculative(self):
        """GPU thread loop with DFlash speculative decoding.

        Collects incoming requests, prefills them as a batch with
        ``capture_layer_ids``, then runs ``_dflash_rounds_batch`` for
        decode. Between speculative rounds the loop checks for new
        requests — new arrivals trigger a batch rebuild (re-prefill
        for the new sequences, extend target caches, cold-restart
        drafter). Finished sequences are filtered out automatically
        by ``_dflash_rounds_batch``'s ``stop_check`` callback.
        """
        from mlx_lm.sample_utils import make_sampler as _make_sampler

        generation_stream = mx.default_stream(mx.default_device())

        lm = self.model.language_model
        drafter = self.draft_model
        target_layer_ids = list(drafter.config.target_layer_ids)
        sampler = _make_sampler(temp=0)
        draft_block_size_str = os.environ.get("MLX_VLM_DRAFT_BLOCK_SIZE")
        draft_block_size = int(draft_block_size_str) if draft_block_size_str else None

        while not self._stop:
            try:
                # --- Phase 1: collect pending requests ---
                pending = []
                timeout = 0.1
                try:
                    item = self.requests.get(timeout=timeout)
                    if item is None and self._stop:
                        break
                    if item is not None:
                        pending.append(item)
                except QueueEmpty:
                    pass
                while True:
                    try:
                        item = self.requests.get_nowait()
                        if item is not None:
                            pending.append(item)
                    except QueueEmpty:
                        break

                if not pending:
                    continue

                # --- Phase 2: prefill new batch ---
                uids = []
                rqueues = {}
                token_lists = {}
                max_tokens_map = {}
                all_input_ids = []

                for rqueue, raw_inputs, prompt_tokens, args, images in pending:
                    input_ids, _ = self._gpu_embed(raw_inputs, images)
                    uid = id(rqueue)
                    uids.append(uid)
                    rqueues[uid] = rqueue
                    token_lists[uid] = []
                    max_tokens_map[uid] = args.max_tokens
                    all_input_ids.append(input_ids.squeeze(0).tolist())
                    rqueue.put(GenerationContext(uid=uid, prompt_tokens=prompt_tokens))
                    sampler = self._make_sampler(args) or _make_sampler(temp=0)

                B = len(uids)
                max_len = max(len(ids) for ids in all_input_ids)
                padded = [[0] * (max_len - len(ids)) + ids for ids in all_input_ids]
                input_mx = mx.array(padded, dtype=mx.int32)

                prompt_cache = _make_cache(lm, [0] * B)
                lm._position_ids = None
                lm._rope_deltas = None

                with mx.stream(generation_stream):
                    out = lm(
                        input_mx,
                        cache=prompt_cache,
                        capture_layer_ids=target_layer_ids,
                    )
                hidden = mx.concatenate(out.hidden_states, axis=-1)
                first_bonus = sampler(out.logits[:, -1:]).squeeze(-1)
                mx.eval(first_bonus, hidden, out.logits)

                # Send first bonus tokens to clients
                fb_list = first_bonus.tolist()
                for j, uid in enumerate(uids):
                    tok = int(fb_list[j])
                    token_lists[uid].append(tok)
                    text = self.tokenizer.decode([tok])
                    rqueues[uid].put(
                        StreamingToken(
                            text=text,
                            token=tok,
                            logprobs=0.0,
                            finish_reason=None,
                            peak_memory=mx.get_peak_memory() / 1e9,
                        )
                    )

                # --- Phase 3: speculative decode rounds ---
                max_tok = max(max_tokens_map[u] for u in uids)
                finished_uids = set()

                def stop_check(seq_idx, token_id):
                    uid = uids[seq_idx]
                    if uid in finished_uids:
                        return True
                    if token_id in self.stop_tokens:
                        return True
                    if len(token_lists[uid]) >= max_tokens_map[uid]:
                        return True
                    return False

                for tok_list, _ in _dflash_rounds_batch(
                    self.model,
                    drafter,
                    prompt_cache,
                    hidden,
                    first_bonus=first_bonus,
                    max_tokens=max_tok,
                    sampler=sampler,
                    draft_block_size=draft_block_size,
                    token_dtype=mx.int32,
                    stop_check=stop_check,
                ):
                    for j, tok in enumerate(tok_list):
                        if tok is None:
                            continue
                        uid = uids[j]
                        if uid in finished_uids:
                            continue

                        token_lists[uid].append(tok)
                        tokens = token_lists[uid]

                        if len(tokens) >= 2:
                            prev = self.tokenizer.decode(tokens[:-1])
                            curr = self.tokenizer.decode(tokens)
                            text = curr[len(prev) :]
                        else:
                            text = self.tokenizer.decode(tokens)

                        is_stop = tok in self.stop_tokens
                        is_max = len(tokens) >= max_tokens_map[uid]
                        finish = "stop" if is_stop else "length" if is_max else None

                        rqueues[uid].put(
                            StreamingToken(
                                text="" if is_stop else text,
                                token=tok,
                                logprobs=0.0,
                                finish_reason=finish,
                                peak_memory=mx.get_peak_memory() / 1e9,
                            )
                        )

                        if finish is not None:
                            rqueues[uid].put(None)
                            finished_uids.add(uid)

                # Log acceptance stats
                al = drafter.accept_lens
                if al:
                    mean_a = sum(al) / len(al)
                    print(
                        f"[DFlash] batch={B} tokens={sum(len(token_lists[u]) for u in uids)} "
                        f"accept={mean_a:.2f} rounds={len(al)}"
                    )

                # Finalize any remaining
                for uid in uids:
                    if uid not in finished_uids:
                        rqueues[uid].put(
                            StreamingToken(
                                text="",
                                token=0,
                                logprobs=0.0,
                                finish_reason="length",
                                peak_memory=mx.get_peak_memory() / 1e9,
                            )
                        )
                        rqueues[uid].put(None)

            except Exception as e:
                print(f"Error in speculative generation thread: {e}")
                traceback.print_exc()

    def _step(self, batch_gen, active, gen_kwargs=None):
        """One batch generation step: prefill + decode."""
        kwargs = gen_kwargs or {}
        _, responses = batch_gen.next(**kwargs)
        if not responses:
            return

        for r in responses:
            if r.uid not in active:
                continue

            info = active[r.uid]
            rqueue = info["rqueue"]

            tok = r.token
            if hasattr(tok, "item"):
                tok = tok.item()

            if r.finish_reason == "stop":
                text = ""
            else:
                info["tokens"].append(tok)
                curr = self.tokenizer.decode(info["tokens"])
                text = curr[len(info["prev_text"]) :]
                info["prev_text"] = curr

            lp = r.token_logprob

            rqueue.put(
                StreamingToken(
                    text=text,
                    token=tok,
                    logprobs=lp,
                    finish_reason=r.finish_reason,
                    peak_memory=mx.get_peak_memory() / 1e9 if r.finish_reason else 0,
                    top_logprobs=getattr(r, "top_logprobs", None),
                )
            )

            if r.finish_reason is not None:
                rqueue.put(None)
                del active[r.uid]

    def _flush(self, batch_gen, active):
        """Drain all pending text-only prompts before inserting an image request."""
        while batch_gen.has_pending_prompts:
            self._step(batch_gen, active)


def suppress_tool_call_content(
    full_output: str,
    in_tool_call: bool,
    tc_start: Optional[str],
    delta_content: Optional[str],
) -> Tuple[bool, Optional[str]]:
    """Suppress tool-call markup from streamed delta.content.

    Returns updated (in_tool_call, delta_content).
    """
    if not tc_start:
        return in_tool_call, delta_content
    if not in_tool_call:
        if tc_start in full_output:
            return True, None
        if any(full_output.endswith(tc_start[:j]) for j in range(1, len(tc_start))):
            return False, None
    else:
        return True, None
    return in_tool_call, delta_content


def build_generation_kwargs(
    request: Any,
    gen_args: "GenerationArguments",
    template_kwargs: dict[str, Any],
) -> dict[str, Any]:
    return {
        "prefill_step_size": get_prefill_step_size(),
        "kv_bits": get_quantized_kv_bits(request.model),
        "kv_group_size": get_kv_group_size(),
        "kv_quant_scheme": get_kv_quant_scheme(),
        "max_kv_size": get_max_kv_size(request.model),
        "quantized_kv_start": get_quantized_kv_start(),
        **gen_args.to_generate_kwargs(),
        **template_kwargs,
    }


def process_tool_calls(model_output: str, tool_module, tools):
    """Parse tool calls from model output using the appropriate tool parser."""
    called_tools = []
    remaining = model_output

    if tool_module.tool_call_start in model_output:
        if tool_module.tool_call_end == "":
            pattern = re.compile(
                f"{re.escape(tool_module.tool_call_start)}.*?(?:\n|$)", re.DOTALL
            )
        else:
            pattern = re.compile(
                f"{re.escape(tool_module.tool_call_start)}.*?{re.escape(tool_module.tool_call_end)}",
                re.DOTALL,
            )

        matches = re.findall(pattern, model_output)
        if matches:
            remaining = re.sub(pattern, " ", model_output).strip()
            for i, match in enumerate(matches):
                call = (
                    match.strip()
                    .removeprefix(tool_module.tool_call_start)
                    .removesuffix(tool_module.tool_call_end)
                )
                try:
                    tool_call = tool_module.parse_tool_call(call, tools)
                    args = tool_call["arguments"]
                    called_tools.append(
                        {
                            "type": "function",
                            "index": i,
                            "id": str(uuid.uuid4()),
                            "function": {
                                "name": tool_call["name"].strip(),
                                "arguments": (
                                    args
                                    if isinstance(args, str)
                                    else json.dumps(args, ensure_ascii=False)
                                ),
                            },
                        }
                    )
                except Exception:
                    print(f"Invalid tool call: {call}")
    return dict(calls=called_tools, remaining_text=remaining)


def _build_gen_args(request) -> GenerationArguments:
    """Build GenerationArguments from an OpenAIRequest or ChatRequest."""
    max_tokens = getattr(request, "max_tokens", None) or getattr(
        request, "max_output_tokens", DEFAULT_MAX_TOKENS
    )
    logit_bias = getattr(request, "logit_bias", None)
    if logit_bias is not None and isinstance(logit_bias, dict):
        logit_bias = {int(k): v for k, v in logit_bias.items()}
    return GenerationArguments(
        max_tokens=max_tokens,
        temperature=getattr(request, "temperature", DEFAULT_TEMPERATURE),
        top_p=getattr(request, "top_p", DEFAULT_TOP_P),
        top_k=getattr(request, "top_k", 0),
        min_p=getattr(request, "min_p", 0.0),
        repetition_penalty=getattr(request, "repetition_penalty", None),
        logit_bias=logit_bias,
        enable_thinking=getattr(request, "enable_thinking", True),
        thinking_budget=getattr(request, "thinking_budget", None),
        thinking_start_token=getattr(request, "thinking_start_token", None),
    )


def _count_thinking_tag_tokens(text: str) -> int:
    """Count tokens consumed by thinking tags (excluded from completion_tokens)."""
    count = 0
    # <|channel>thought (2 tokens) + <channel|> (1 token) + EOS (1 token)
    if "<|channel>thought" in text and "<channel|>" in text:
        count = 4
    elif "<think>" in text and "</think>" in text:
        count = 2  # <think> and </think> are 1 token each typically
    return count


def _split_thinking(text: str) -> Tuple[Optional[str], str]:
    """Split thinking tags from content. Returns (reasoning, content)."""
    # Handle <|channel>thought...<channel|> format (gemma4)
    # Also handle partial tag: text starting with "thought\n" (continuation)
    if "<|channel>thought" in text or (
        "<channel|>" in text and text.lstrip().startswith("thought")
    ):
        parts = text.split("<channel|>", 1)
        if len(parts) == 2:
            reasoning = (
                parts[0].replace("<|channel>thought", "").lstrip("thought").strip()
            )
            content = parts[1].strip()
            return reasoning or None, content
        reasoning = parts[0].replace("<|channel>thought", "").lstrip("thought").strip()
        return reasoning or None, ""
    # Handle <think>...</think> format (qwen3.5 etc)
    # Also handle partial: output starts with thinking text + </think> (no opening tag)
    if "<think>" in text or "</think>" in text:
        parts = text.split("</think>", 1)
        if len(parts) == 2:
            reasoning = parts[0].replace("<think>", "").strip()
            content = parts[1].strip()
            return reasoning or None, content
        return parts[0].replace("<think>", "").strip(), ""
    return None, text


def _decode_token(tokenizer, token_id: int) -> Tuple[str, Optional[List[int]]]:
    """Decode a single token id to its string + UTF-8 bytes."""
    try:
        text = tokenizer.decode([int(token_id)])
    except Exception:
        text = ""
    try:
        token_bytes = list(text.encode("utf-8"))
    except Exception:
        token_bytes = None
    return text, token_bytes


def _make_logprob_content(
    tokenizer,
    token_id: int,
    logprob: float,
    top_logprobs: Optional[List[Tuple[int, float]]] = None,
    top_k: int = 0,
) -> "ChatLogprobContent":
    """Build an OpenAI-style logprob entry for a single token."""
    token_text, token_bytes = _decode_token(tokenizer, token_id)
    top_list: List[TopLogprob] = []
    if top_k > 0 and top_logprobs:
        for tid, lp in top_logprobs[:top_k]:
            t_text, t_bytes = _decode_token(tokenizer, tid)
            top_list.append(TopLogprob(token=t_text, logprob=float(lp), bytes=t_bytes))
    return ChatLogprobContent(
        token=token_text,
        logprob=float(logprob),
        bytes=token_bytes,
        top_logprobs=top_list,
    )


# Global response generator for continuous batching
response_generator: Optional[ResponseGenerator] = None

# Loading/unloading utilities
model_cache = {}


@asynccontextmanager
async def lifespan(app):
    model_path = os.environ.pop("MLX_VLM_PRELOAD_MODEL", None)
    if model_path:
        adapter_path = os.environ.pop("MLX_VLM_PRELOAD_ADAPTER", None)
        try:
            logger.info("Pre-loading model: %s", model_path)
            print(f"Preloading model: {model_path}")
            get_cached_model(model_path, adapter_path)
            kv_bits = os.environ.get("KV_BITS")
            kv_scheme = os.environ.get("KV_QUANT_SCHEME", "uniform")
            if kv_bits:
                logger.info("KV cache quantization: bits=%s scheme=%s", kv_bits, kv_scheme)
            # Pin model: block swaps from API requests
            if os.environ.get("MLX_PIN_MODEL", "").lower() in ("1", "true"):
                _pinned_models.add(model_path)
                print(f"Model pinned: {model_path}")
            # Warmup은 /v1/warmup 엔드포인트로 수동 트리거 또는
            # PrefixCache.warmup()으로 처리 (서버 시작 시 자동 warmup 제거 —
            # 122B 65GB 모델 로딩 시간이 이미 길어서 startup lifespan 내에서
            # 추가 추론은 GPU Timeout 위험 증가)
            logger.info("Model ready, continuous batching enabled.")
        except Exception as e:
            print(f"Failed to preload model: {e}")
            print("Server will continue without a preloaded model.")
    yield


app = FastAPI(
    title="MLX-Seori Inference API",
    description="MLX VLM server with PrefixCache, Metal memory management, and inference tracking.",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Inference tracking + GC middleware ────────────────────────
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest

_gc_threshold_mb = _metal_mem_limit_gb * 0.75 * 1024


def _get_active_mb() -> float:
    try:
        return mx.get_active_memory() / 1e6
    except AttributeError:
        return mx.metal.get_active_memory() / 1e6


class InferenceTrackingMiddleware(BaseHTTPMiddleware):
    """Track inflight inference requests and trigger GC on memory pressure."""

    async def dispatch(self, request: StarletteRequest, call_next):
        global _inflight, _last_done_ts, _busy_since

        path = request.url.path
        is_inference = "chat/completions" in path or "/responses" in path

        if is_inference:
            with _inflight_lock:
                if _inflight == 0:
                    _busy_since = time.time()
                _inflight += 1

        try:
            return await call_next(request)
        finally:
            if is_inference:
                with _inflight_lock:
                    _inflight -= 1
                    _last_done_ts = time.time()

                active = _get_active_mb()
                if active > _gc_threshold_mb:
                    gc.collect()
                    try:
                        mx.clear_cache()
                    except AttributeError:
                        mx.metal.clear_cache()
                    print(
                        f"GC triggered (active={active:.0f}MB > "
                        f"{_gc_threshold_mb:.0f}MB threshold)"
                    )


app.add_middleware(InferenceTrackingMiddleware)

MAX_IMAGES = 10  # Maximum number of images to process at once

# Loading/unloading utilities

prefix_cache = PrefixCache()
_pinned_models: set = set()  # Models that cannot be swapped by API requests


# ── Inference tracking ───────────────────────────────────────
import threading

_inflight = 0
_inflight_lock = threading.Lock()
_last_done_ts = 0.0
_busy_since = 0.0
_last_request: Optional[dict] = None  # last completed inference summary


def _record_last_request(
    prompt_tokens: int,
    gen_tokens: int,
    elapsed_s: float,
    prompt_tps: float = 0,
    gen_tps: float = 0,
    cached_prefix: int = 0,
):
    global _last_request
    _last_request = {
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_tokens,
        "elapsed_s": round(elapsed_s, 2),
        "prompt_tps": round(prompt_tps, 1),
        "gen_tps": round(gen_tps, 1),
        "cached_prefix": cached_prefix,
        "ts": time.strftime("%H:%M:%S"),
    }


# ── Request logger (JSONL) ──────────────────────────────────
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
# 일부 모델(122B MoE)은 <think> 태그 없이 "Thinking Process:" plain text로 thinking 출력
_PLAIN_THINK_RE = re.compile(
    r"^(?:Thinking Process:.*?)(?=\n(?:##|\*\*|[A-Z가-힣])|$)",
    re.DOTALL,
)


def _strip_thinking(text: str) -> str:
    """Remove <think>…</think> blocks and plain-text thinking preamble."""
    if not text:
        return text
    # 1) 완전한 <think>...</think> 태그
    result = _THINK_RE.sub("", text)
    # 2) 닫는 태그만 있는 경우 (프리필에서 <think>가 주입되고 모델이 바로 </think> 출력)
    if "</think>" in result:
        result = result.split("</think>", 1)[-1]
    # 3) 열린 <think> 태그가 남아있으면 제거
    if "<think>" in result:
        result = result.split("<think>", 1)[0]
    # 4) Plain text thinking (태그 없이 "Thinking Process:" 로 시작)
    if result.lstrip().startswith("Thinking Process:"):
        # 아직 실제 답변이 안 나왔으면 전체가 thinking — 빈 문자열 반환
        # 실제 답변이 나왔으면 thinking 부분 제거
        lines = result.split("\n")
        answer_start = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            # 빈 줄이 아니고, 번호 매기기/인덴트가 아닌 실제 답변 시작 감지
            if stripped and not stripped.startswith(("Thinking", "*", "-", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.")):
                # 마크다운 헤더나 실제 내용인지 확인
                if stripped.startswith(("#", "##")) or (len(stripped) > 5 and not stripped[0].isdigit()):
                    answer_start = i
                    break
        if answer_start is not None:
            result = "\n".join(lines[answer_start:])
        else:
            result = ""  # 아직 thinking만 — 빈 문자열
    return result.strip()


_req_logger: Optional[logging.Logger] = None


def _init_request_logger():
    """Lazy-init a file logger that writes to /tmp/mlx_requests.jsonl."""
    global _req_logger
    if _req_logger is not None:
        return
    _req_logger = logging.getLogger("mlx_vlm.requests")
    _req_logger.setLevel(logging.INFO)
    _req_logger.propagate = False
    handler = logging.handlers.RotatingFileHandler(
        "/tmp/mlx_requests.jsonl",
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    _req_logger.addHandler(handler)


def _log_request(
    messages: list,
    *,
    prompt_tokens: int = 0,
    gen_tokens: int = 0,
    elapsed_s: float = 0,
    gen_tps: float = 0,
    error: str = "",
    stream: bool = False,
    mtp: bool = False,
):
    """Log a single request as one JSONL line."""
    _init_request_logger()
    # Summarise messages: role + first 200 chars of content
    msgs_summary = []
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content", "")
        if len(content) > 200:
            content = content[:200] + f"…({len(content)}ch)"
        msgs_summary.append({"role": role, "content": content})

    entry = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "messages": msgs_summary,
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_tokens,
        "elapsed_s": round(elapsed_s, 2),
        "gen_tps": round(gen_tps, 1),
        "stream": stream,
        "mtp": mtp,
    }
    if error:
        entry["error"] = error
    _req_logger.info(json.dumps(entry, ensure_ascii=False))


class FlexibleBaseModel(BaseModel):
    """Base model that ignores/accepts any unknown OpenAI SDK fields."""

    model_config = ConfigDict(extra="allow")


def load_model_resources(model_path: str, adapter_path: Optional[str]):
    """
    Loads model, processor, and config based on paths.
    Handles potential loading errors.
    """
    try:
        print(f"Loading model from: {model_path}")
        if adapter_path:
            print(f"Loading adapter from: {adapter_path}")
        # Use the load function from utils.py which handles path resolution and loading
        trust_remote_code = (
            os.environ.get("MLX_TRUST_REMOTE_CODE", "false").lower() == "true"
        )
        model, processor = load(
            model_path, adapter_path, trust_remote_code=trust_remote_code
        )
        config = model.config
        print("Model and processor loaded successfully.")
        return model, processor, config
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        traceback.print_exc()  # Print detailed traceback for debugging
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")


_INHERIT_ADAPTER = object()


def get_cached_model(model_path: str, adapter_path=_INHERIT_ADAPTER):
    """
    Factory function to get or load the appropriate model resources from cache or by loading.
    Also creates/updates the ResponseGenerator for continuous batching.
    """
    global model_cache, response_generator

    if adapter_path is _INHERIT_ADAPTER:
        cached = model_cache.get("cache_key")
        adapter_path = cached[1] if cached and cached[0] == model_path else None

    cache_key = (model_path, adapter_path)

    # Return from cache if already loaded and matches the requested paths
    if model_cache.get("cache_key") == cache_key:
        return model_cache["model"], model_cache["processor"], model_cache["config"]

    # If a pinned model is loaded, block swap attempts
    loaded_path = model_cache.get("model_path")
    if loaded_path and loaded_path in _pinned_models and model_path != loaded_path:
        print(f"Model swap blocked: {model_path} → using pinned {loaded_path}")
        return model_cache["model"], model_cache["processor"], model_cache["config"]

    # If cache exists but doesn't match, clear it
    if model_cache:
        print("New model request, clearing existing cache...")
        unload_model_sync()  # Use a synchronous version for internal call

    vision_cache_size = int(os.environ.get("MLX_VLM_VISION_CACHE_SIZE", "20"))
    vision_cache = VisionFeatureCache(max_size=vision_cache_size)

    # KV cache quantization (uniform or TurboQuant)
    kv_bits = get_quantized_kv_bits(model_path)
    kv_group_size = get_kv_group_size()
    quantized_kv_start = get_quantized_kv_start()
    kv_quant_scheme = get_kv_quant_scheme()

    response_generator = ResponseGenerator(
        model_path=model_path,
        adapter_path=adapter_path,
        vision_cache=vision_cache,
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
        kv_quant_scheme=kv_quant_scheme,
        quantized_kv_start=quantized_kv_start,
        top_logprobs_k=get_top_logprobs_k(),
    )
    try:
        model, processor, config = response_generator.wait_until_ready()
    except Exception:
        response_generator.stop_and_join()
        response_generator = None
        vision_cache.clear()
        raise

    model_cache = {
        "cache_key": cache_key,
        "model_path": model_path,
        "adapter_path": adapter_path,
        "model": model,
        "processor": processor,
        "config": config,
        "vision_cache": vision_cache,
    }

    return model, processor, config


# Synchronous unload function for internal use
def unload_model_sync():
    global model_cache, response_generator
    if not model_cache:
        return False

    print(
        f"Unloading model: {model_cache.get('model_path')}, Adapter: {model_cache.get('adapter_path')}"
    )

    # Stop the ResponseGenerator if running
    if response_generator is not None:
        print("Stopping ResponseGenerator...")
        response_generator.stop_and_join()
        response_generator = None

    # Clear vision cache before dropping references
    if "vision_cache" in model_cache:
        model_cache["vision_cache"].clear()
    model_cache = {}
    # Force garbage collection
    gc.collect()
    mx.clear_cache()
    print("Model unloaded and cache cleared.")
    return True


# OpenAI API Models

# Models for /responses endpoint


class ResponseInputTextParam(TypedDict, total=False):
    text: Required[str]
    type: Required[
        Literal["input_text", "text"]
    ]  # The type of the input item. Always `input_text`.


class ResponseInputImageParam(TypedDict, total=False):
    detail: Literal["high", "low", "auto"] = Field(
        "auto", description="The detail level of the image to be sent to the model."
    )
    """The detail level of the image to be sent to the model.

    One of `high`, `low`, or `auto`. Defaults to `auto`.
    """
    type: Required[
        Literal["input_image"]
    ]  # The type of the input item. Always `input_image`.
    image_url: Required[str]
    file_id: Optional[str]
    """The ID of the file to be sent to the model.
     NOTE : wouldn't this help the model if we passed the file_id as well to the vlm models
    """


class InputAudio(TypedDict, total=False):
    data: Required[str]
    format: Required[str]


class ResponseInputAudioParam(TypedDict, total=False):
    type: Required[
        Literal["input_audio"]
    ]  # The type of the input item. Always `input_audio`.
    input_audio: Required[InputAudio]


class ImageUrl(TypedDict, total=False):
    url: Required[str]


class ResponseImageUrlParam(TypedDict, total=False):
    type: Required[
        Literal["image_url"]
    ]  # The type of the input item. Always`image_url`.
    image_url: Required[ImageUrl]


ResizeShapeInput: TypeAlias = Union[Tuple[int], Tuple[int, int]]

ResponseInputContentParam: TypeAlias = Union[
    ResponseInputTextParam,
    ResponseInputImageParam,
    ResponseImageUrlParam,
    ResponseInputAudioParam,
]

ResponseInputMessageContentListParam: TypeAlias = List[ResponseInputContentParam]


class ResponseOutputText(TypedDict, total=False):
    text: Required[str]
    type: Required[
        Literal["output_text"]
    ]  # The type of the output item. Always `output_text`


ResponseOutputMessageContentList: TypeAlias = List[ResponseOutputText]


class ChatMessage(FlexibleBaseModel):
    role: Literal["user", "assistant", "system", "developer", "tool"] = Field(
        ...,
        description="Role of the message sender.",
    )
    content: Union[
        str,
        None,
        ResponseInputMessageContentListParam,
        ResponseOutputMessageContentList,
    ] = Field(None, description="Content of the message.")
    reasoning: Optional[str] = Field(
        None, description="Thinking/reasoning content (when thinking is enabled)."
    )
    tool_calls: Optional[List[Any]] = Field(
        None, description="Tool calls made by the assistant."
    )
    tool_call_id: Optional[str] = Field(
        None, description="ID of the tool call this message is a response to."
    )
    name: Optional[str] = Field(None, description="Name of the tool/function.")


class OpenAIRequest(FlexibleBaseModel):
    """
    OpenAI-compatible request structure.
    Using this structure : https://github.com/openai/openai-python/blob/main/src/openai/resources/responses/responses.py
    """

    input: Union[str, List[ChatMessage]] = Field(
        ..., description="Input text or list of chat messages."
    )
    model: str = Field(..., description="The model to use for generation.")
    max_output_tokens: int = Field(
        DEFAULT_MAX_TOKENS, description="Maximum number of tokens to generate."
    )
    temperature: float = Field(
        DEFAULT_TEMPERATURE, description="Temperature for sampling."
    )
    top_p: float = Field(DEFAULT_TOP_P, description="Top-p sampling.")
    top_k: int = Field(0, description="Top-k sampling.")
    min_p: float = Field(0.0, description="Min-p sampling.")
    repetition_penalty: Optional[float] = Field(None, description="Repetition penalty.")
    logit_bias: Optional[Any] = Field(None, description="Logit bias dict.")
    enable_thinking: bool = Field(True, description="Enable thinking mode.")
    thinking_budget: Optional[int] = Field(None, description="Max thinking tokens.")
    thinking_start_token: Optional[str] = Field(
        None, description="Thinking start token."
    )
    stream: bool = Field(
        False, description="Whether to stream the response chunk by chunk."
    )


class OpenAIUsage(BaseModel):
    """Token usage details including input tokens, output tokens, breakdown, and total tokens used."""

    input_tokens: int
    output_tokens: int
    total_tokens: int


class OpenAIErrorObject(BaseModel):
    """Error object returned when the model fails to generate a Response."""

    code: Optional[str] = None
    message: Optional[str] = None
    param: Optional[str] = None
    type: Optional[str] = None


class OpenAIResponse(BaseModel):
    id: str = Field(..., description="Unique identifier for this Response")
    object: Literal["response"] = Field(
        ..., description="The object type of this resource - always set to response"
    )
    created_at: int = Field(
        ..., description="Unix timestamp (in seconds) of when this Response was created"
    )
    status: Literal["completed", "failed", "in_progress", "incomplete"] = Field(
        ..., description="The status of the response generation"
    )
    error: Optional[OpenAIErrorObject] = Field(
        None,
        description="An error object returned when the model fails to generate a Response",
    )
    instructions: Optional[str] = Field(
        None,
        description="Inserts a system (or developer) message as the first item in the model's context",
    )
    max_output_tokens: Optional[int] = Field(
        None,
        description="An upper bound for the number of tokens that can be generated for a response",
    )
    model: str = Field(..., description="Model ID used to generate the response")
    output: List[Union[ChatMessage, Any]] = Field(
        ..., description="An array of content items generated by the model"
    )
    output_text: Optional[str] = Field(
        None,
        description="SDK-only convenience property containing aggregated text output",
    )
    temperature: Optional[float] = Field(
        None, ge=0, le=2, description="Sampling temperature between 0 and 2"
    )
    top_p: Optional[float] = Field(
        None, ge=0, le=1, description="Nucleus sampling probability mass"
    )
    truncation: Union[Literal["auto", "disabled"], str] = Field(
        "disabled", description="The truncation strategy to use"
    )
    usage: OpenAIUsage = Field(
        ..., description="Token usage details"
    )  # we need the model to return stats
    user: Optional[str] = Field(
        None, description="A unique identifier representing your end-user"
    )


class BaseStreamEvent(BaseModel):
    type: str


class ContentPartOutputText(BaseModel):
    type: Literal["output_text"]
    text: str
    annotations: List[str] = []


class MessageItem(BaseModel):
    id: str
    type: Literal["message"]
    status: Literal["in_progress", "completed"]
    role: str
    content: List[ContentPartOutputText] = []


class ResponseCreatedEvent(BaseStreamEvent):
    type: Literal["response.created"]
    response: OpenAIResponse


class ResponseInProgressEvent(BaseStreamEvent):
    type: Literal["response.in_progress"]
    response: OpenAIResponse


class ResponseOutputItemAddedEvent(BaseStreamEvent):
    type: Literal["response.output_item.added"]
    output_index: int
    item: MessageItem


class ResponseContentPartAddedEvent(BaseStreamEvent):
    type: Literal["response.content_part.added"]
    item_id: str
    output_index: int
    content_index: int
    part: ContentPartOutputText


class ResponseOutputTextDeltaEvent(BaseStreamEvent):
    type: Literal["response.output_text.delta"]
    item_id: str
    output_index: int
    content_index: int
    delta: str


class ResponseOutputTextDoneEvent(BaseStreamEvent):
    type: Literal["response.output_text.done"]
    item_id: str
    output_index: int
    content_index: int
    text: str


class ResponseContentPartDoneEvent(BaseStreamEvent):
    type: Literal["response.content_part.done"]
    item_id: str
    output_index: int
    content_index: int
    part: ContentPartOutputText


class ResponseOutputItemDoneEvent(BaseStreamEvent):
    type: Literal["response.output_item.done"]
    output_index: int
    item: MessageItem


class ResponseCompletedEvent(BaseStreamEvent):
    type: Literal["response.completed"]
    response: OpenAIResponse


StreamEvent = Union[
    ResponseCreatedEvent,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseContentPartAddedEvent,
    ResponseOutputTextDeltaEvent,
    ResponseOutputTextDoneEvent,
    ResponseContentPartDoneEvent,
    ResponseOutputItemDoneEvent,
    ResponseCompletedEvent,
]

# Models for /chat/completion endpoint


class VLMRequest(FlexibleBaseModel):
    model: str = Field(
        DEFAULT_MODEL_PATH,
        description="The path to the local model directory or Hugging Face repo.",
    )
    adapter_path: Optional[str] = Field(
        None, description="The path to the adapter weights."
    )
    max_tokens: int = Field(
        DEFAULT_MAX_TOKENS, description="Maximum number of tokens to generate."
    )
    temperature: float = Field(
        DEFAULT_TEMPERATURE, description="Temperature for sampling."
    )
    top_p: float = Field(DEFAULT_TOP_P, description="Top-p sampling.")
    top_k: int = Field(0, description="Top-k sampling.")
    min_p: float = Field(0.0, description="Min-p sampling.")
    seed: int = Field(DEFAULT_SEED, description="Seed for random generation.")
    repetition_penalty: Optional[float] = Field(None, description="Repetition penalty.")
    logit_bias: Optional[Any] = Field(None, description="Logit bias dict.")
    enable_thinking: bool = Field(True, description="Enable thinking mode.")
    thinking_budget: Optional[int] = Field(None, description="Max thinking tokens.")
    thinking_start_token: Optional[str] = Field(
        None, description="Thinking start token."
    )
    logprobs: Optional[bool] = Field(
        None,
        description="Return log-probabilities for each output token.",
    )
    top_logprobs: Optional[int] = Field(
        None,
        description=(
            "Number of most-likely tokens to return at each position "
            "(0-20). Requires logprobs=true. The server-side cap is set by "
            "the TOP_LOGPROBS_K env var; values above the cap are clamped."
        ),
    )
    resize_shape: Optional[ResizeShapeInput] = Field(
        None,
        description="Resize shape for the image. Provide one integer for square or two for (height, width).",
    )

    @field_validator("resize_shape", mode="before")
    @classmethod
    def normalize_resize_shape_field(cls, value):
        return normalize_resize_shape(value)


class GenerationRequest(VLMRequest):
    """
    Inherits from VLMRequest and adds additional fields for the generation request.
    """

    stream: bool = Field(
        False, description="Whether to stream the response chunk by chunk."
    )


class PromptTokensDetails(BaseModel):
    cached_tokens: int = 0


class UsageStats(BaseModel):
    """OpenAI-compatible usage statistics for chat completions."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens_details: PromptTokensDetails = PromptTokensDetails()
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    peak_memory: float = 0.0


class ChatRequest(GenerationRequest):
    messages: List[ChatMessage]


class TopLogprob(BaseModel):
    token: str
    logprob: float
    bytes: Optional[List[int]] = None


class ChatLogprobContent(BaseModel):
    token: str
    logprob: float
    bytes: Optional[List[int]] = None
    top_logprobs: List[TopLogprob] = []


class ChatLogprobs(BaseModel):
    content: List[ChatLogprobContent] = []


class ChatChoice(BaseModel):
    index: int = 0
    finish_reason: str = "stop"
    message: ChatMessage
    logprobs: Optional[ChatLogprobs] = None


class ChatResponse(BaseModel):
    id: str = ""
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: List[ChatChoice] = []
    usage: Optional[UsageStats] = None


class ChatStreamChoice(BaseModel):
    index: int = 0
    finish_reason: Optional[str] = None
    delta: ChatMessage
    logprobs: Optional[ChatLogprobs] = None


class ChatStreamChunk(BaseModel):
    id: str = ""
    object: str = "chat.completion.chunk"
    created: int = 0
    model: str = ""
    choices: List[ChatStreamChoice] = []
    usage: Optional[UsageStats] = None


# Models for /models endpoint


class ModelInfo(BaseModel):
    id: str
    object: str
    created: int


class ModelsResponse(BaseModel):
    object: Literal["list"]
    data: List[ModelInfo]


# OpenAI compatile endpoints


@app.post("/responses")
@app.post("/v1/responses", include_in_schema=False)
async def responses_endpoint(request: Request):
    """
    OpenAI-compatible endpoint for generating text based on a prompt and optional images.

    using client.responses.create method.

    example:

    from openai import OpenAI

    API_URL = "http://0.0.0.0:8000"
    API_KEY = 'any'

    def run_openai(prompt, img_url,system, stream=False, max_output_tokens=512, model="mlx-community/Qwen2.5-VL-3B-Instruct-8bit"):
        ''' Calls the OpenAI API
        '''

        client = OpenAI(base_url=f"{API_URL}", api_key=API_KEY)

        try :
            response = client.responses.create(
                model=model,
                input=[
                    {"role":"system",
                    "content": f"{system}"
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": f"{img_url}"},
                        ],
                    }
                ],
                max_output_tokens=max_output_tokens,
                stream=stream
            )
            if not stream:
                print(response.output[0].content[0].text)
                print(response.usage)
            else:
                for event in response:
                    # Process different event types if needed
                    if hasattr(event, 'delta') and event.delta:
                        print(event.delta, end="", flush=True)
                    elif event.type == 'response.completed':
                        print("\n--- Usage ---")
                        print(event.response.usage)

        except Exception as e:
            # building a response object to match the one returned when request is successful so that it can be processed in the same way
            return {"model - error":str(e),"content":{}, "model":model}

    """

    request_start = time.perf_counter()
    body = await request.json()
    openai_request = OpenAIRequest(**body)

    try:
        # Get model, processor, config - loading if necessary
        model, processor, config = get_cached_model(openai_request.model)

        kwargs = {}

        chat_messages = []
        images = []
        instructions = None
        if openai_request.input:
            if isinstance(openai_request.input, str):
                # If input is a string, treat it as a single text message
                chat_messages.append({"role": "user", "content": openai_request.input})
            elif isinstance(openai_request.input, list):
                # If input is a list, treat it as a series of chat messages
                for message in openai_request.input:
                    if isinstance(message, ChatMessage):
                        if isinstance(message.content, str):
                            chat_messages.append(
                                {"role": message.role, "content": message.content}
                            )
                            if message.role == "system":
                                instructions = message.content
                        elif isinstance(message.content, list):
                            # Handle list of content items
                            for item in message.content:
                                if isinstance(item, dict):
                                    if item["type"] == "input_text":
                                        chat_messages.append(
                                            {
                                                "role": message.role,
                                                "content": item["text"],
                                            }
                                        )
                                        if message.role == "system":
                                            instructions = item["text"]
                                    # examples for multiple images (https://platform.openai.com/docs/guides/images?api-mode=responses)
                                    elif item["type"] == "input_image":
                                        images.append(item["image_url"])
                                    else:
                                        print(
                                            f"invalid input item type: {item['type']}"
                                        )
                                        raise HTTPException(
                                            status_code=400,
                                            detail="Invalid input item type.",
                                        )
                                else:
                                    print(
                                        f"Invalid message content item format: {item}"
                                    )
                                    raise HTTPException(
                                        status_code=400,
                                        detail="Missing type in input item.",
                                    )
                        else:
                            print("Invalid message content format.")
                            raise HTTPException(
                                status_code=400, detail="Invalid input format."
                            )
                    else:
                        print("not a ChatMessage")
                        raise HTTPException(
                            status_code=400, detail="Invalid input format."
                        )
            else:
                print("neither string not list")
                raise HTTPException(status_code=400, detail="Invalid input format.")

        else:
            print("no input")
            raise HTTPException(status_code=400, detail="Missing input.")

        gen_args = _build_gen_args(openai_request)

        formatted_prompt = apply_chat_template(
            processor,
            config,
            chat_messages,
            num_images=len(images),
            **gen_args.to_template_kwargs(),
        )

        logger.debug(
            "responses request: model=%s images=%d max_tokens=%s temp=%s stream=%s",
            openai_request.model,
            len(images),
            gen_args.max_tokens,
            gen_args.temperature,
            openai_request.stream,
        )

        generated_at = datetime.now().timestamp()
        response_id = f"resp_{uuid.uuid4().hex}"
        message_id = f"msg_{uuid.uuid4().hex}"

        if openai_request.stream:
            # Streaming response
            async def stream_generator():
                token_iterator = None
                token_iter = None  # For ResponseGenerator cleanup
                try:
                    # Create base response object (to match the openai pipeline)
                    base_response = OpenAIResponse(
                        id=response_id,
                        object="response",
                        created_at=int(generated_at),
                        status="in_progress",
                        instructions=instructions,
                        max_output_tokens=openai_request.max_output_tokens,
                        model=openai_request.model,
                        output=[],
                        output_text="",
                        temperature=openai_request.temperature,
                        top_p=openai_request.top_p,
                        usage={
                            "input_tokens": 0,  # get prompt tokens
                            "output_tokens": 0,
                            "total_tokens": 0,
                        },
                    )

                    # Send response.created event  (to match the openai pipeline)
                    yield f"event: response.created\ndata: {ResponseCreatedEvent(type='response.created', response=base_response).model_dump_json()}\n\n"

                    # Send response.in_progress event  (to match the openai pipeline)
                    yield f"event: response.in_progress\ndata: {ResponseInProgressEvent(type='response.in_progress', response=base_response).model_dump_json()}\n\n"

                    # Send response.output_item.added event  (to match the openai pipeline)
                    message_item = MessageItem(
                        id=message_id,
                        type="message",
                        status="in_progress",
                        role="assistant",
                        content=[],
                    )
                    yield f"event: response.output_item.added\ndata: {ResponseOutputItemAddedEvent(type='response.output_item.added', output_index=0, item=message_item).model_dump_json()}\n\n"

                    # Send response.content_part.added event
                    content_part = ContentPartOutputText(
                        type="output_text", text="", annotations=[]
                    )
                    yield f"event: response.content_part.added\ndata: {ResponseContentPartAddedEvent(type='response.content_part.added', item_id=message_id, output_index=0, content_index=0, part=content_part).model_dump_json()}\n\n"

                    # Stream text deltas using ResponseGenerator (continuous batching)
                    full_text = ""
                    usage_stats = {"input_tokens": 0, "output_tokens": 0}

                    if response_generator is not None:
                        ctx, token_iter = response_generator.generate(
                            prompt=formatted_prompt,
                            images=images if images else None,
                            args=gen_args,
                        )

                        output_tokens = 0
                        for token in token_iter:
                            output_tokens += 1
                            delta = token.text
                            full_text += delta
                            usage_stats = {
                                "input_tokens": ctx.prompt_tokens,
                                "output_tokens": output_tokens,
                            }

                            yield f"event: response.output_text.delta\ndata: {ResponseOutputTextDeltaEvent(type='response.output_text.delta', item_id=message_id, output_index=0, content_index=0, delta=delta).model_dump_json()}\n\n"
                            await asyncio.sleep(0.01)

                            if token.finish_reason:
                                break
                    else:
                        # Fallback to stream_generate
                        token_iterator = stream_generate(
                            model=model,
                            processor=processor,
                            prompt=formatted_prompt,
                            image=images,
                            temperature=openai_request.temperature,
                            max_tokens=openai_request.max_output_tokens,
                            top_p=openai_request.top_p,
                            vision_cache=model_cache.get("vision_cache"),
                            **kwargs,
                        )

                        for chunk in token_iterator:
                            if chunk is None or not hasattr(chunk, "text"):
                                continue

                            delta = chunk.text
                            full_text += delta
                            usage_stats = {
                                "input_tokens": chunk.prompt_tokens,
                                "output_tokens": chunk.generation_tokens,
                            }

                            yield f"event: response.output_text.delta\ndata: {ResponseOutputTextDeltaEvent(type='response.output_text.delta', item_id=message_id, output_index=0, content_index=0, delta=delta).model_dump_json()}\n\n"
                            await asyncio.sleep(0.01)

                    # Split thinking from content for final events
                    _, clean_text = _split_thinking(full_text)

                    # Send response.output_text.done event (to match the openai pipeline)
                    yield f"event: response.output_text.done\ndata: {ResponseOutputTextDoneEvent(type='response.output_text.done', item_id=message_id, output_index=0, content_index=0, text=clean_text).model_dump_json()}\n\n"

                    # Send response.content_part.done event (to match the openai pipeline)
                    final_content_part = ContentPartOutputText(
                        type="output_text", text=clean_text, annotations=[]
                    )
                    yield f"event: response.content_part.done\ndata: {ResponseContentPartDoneEvent(type='response.content_part.done', item_id=message_id, output_index=0, content_index=0, part=final_content_part).model_dump_json()}\n\n"

                    # Send response.output_item.done event (to match the openai pipeline)
                    final_message_item = MessageItem(
                        id=message_id,
                        type="message",
                        status="completed",
                        role="assistant",
                        content=[final_content_part],
                    )
                    yield f"event: response.output_item.done\ndata: {ResponseOutputItemDoneEvent(type='response.output_item.done', output_index=0, item=final_message_item).model_dump_json()}\n\n"

                    # Send response.completed event (to match the openai pipeline)
                    completed_response = base_response.model_copy(
                        update={
                            "status": "completed",
                            "output": [final_message_item],
                            "usage": {
                                "input_tokens": usage_stats["input_tokens"],
                                "output_tokens": usage_stats["output_tokens"],
                                "total_tokens": usage_stats["input_tokens"]
                                + usage_stats["output_tokens"],
                            },
                        }
                    )
                    yield f"event: response.completed\ndata: {ResponseCompletedEvent(type='response.completed', response=completed_response).model_dump_json()}\n\n"

                    _record_last_request(
                        prompt_tokens=usage_stats.get("input_tokens", 0),
                        gen_tokens=usage_stats.get("output_tokens", 0),
                        elapsed_s=time.time() - _busy_since if _busy_since else 0,
                        prompt_tps=usage_stats.get("prompt_tps", 0),
                        gen_tps=usage_stats.get("generation_tps", 0),
                    )

                except Exception as e:
                    print(f"Error during stream generation: {e}")
                    traceback.print_exc()
                    error_data = json.dumps({"error": str(e)})
                    yield f"data: {error_data}\n\n"

                finally:
                    if token_iter is not None:
                        try:
                            token_iter.close()
                        except Exception:
                            pass
                    mx.clear_cache()
                    gc.collect()

            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        else:
            # Non-streaming response
            try:
                _t0_resp = time.time()
                full_text = ""
                prompt_tokens = 0
                output_tokens = 0
                prompt_tps = 0.0
                gen_tps = 0.0

                if response_generator is not None:
                    ctx, token_iter = response_generator.generate(
                        prompt=formatted_prompt,
                        images=images if images else None,
                        args=gen_args,
                    )
                    prompt_tokens = ctx.prompt_tokens
                    for token in token_iter:
                        full_text += token.text
                        output_tokens += 1
                        if token.finish_reason:
                            break
                    try:
                        token_iter.close()
                    except Exception:
                        pass
                else:
                    result = generate(
                        model=model,
                        processor=processor,
                        prompt=formatted_prompt,
                        image=images,
                        verbose=logger.isEnabledFor(logging.DEBUG),
                        vision_cache=model_cache.get("vision_cache"),
                        **gen_args.to_generate_kwargs(),
                        **kwargs,
                    )
                    full_text = result.text
                    prompt_tokens = result.prompt_tokens
                    output_tokens = result.generation_tokens
                    prompt_tps = getattr(result, "prompt_tps", 0.0)
                    gen_tps = getattr(result, "generation_tps", 0.0)

                _record_last_request(
                    prompt_tokens=prompt_tokens,
                    gen_tokens=output_tokens,
                    elapsed_s=time.time() - _t0_resp,
                    prompt_tps=prompt_tps,
                    gen_tps=gen_tps,
                )

                mx.clear_cache()
                gc.collect()

                reasoning, content = _split_thinking(full_text)

                response = OpenAIResponse(
                    id=response_id,
                    object="response",
                    created_at=int(generated_at),
                    status="completed",
                    instructions=instructions,
                    max_output_tokens=openai_request.max_output_tokens,
                    model=openai_request.model,
                    output=[
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": content,
                                }
                            ],
                            "reasoning": reasoning,
                        }
                    ],
                    output_text=content,
                    temperature=openai_request.temperature,
                    top_p=openai_request.top_p,
                    usage={
                        "input_tokens": prompt_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": prompt_tokens + output_tokens,
                    },
                )

                elapsed = time.perf_counter() - request_start
                logger.debug(
                    "responses done: prompt_tokens=%d output_tokens=%d "
                    "total_time=%.2fs",
                    prompt_tokens,
                    output_tokens,
                    elapsed,
                )
                if logger.isEnabledFor(logging.DEBUG):
                    resp_text = content or ""
                    logger.debug(
                        "  response: %s",
                        resp_text[:200] + ("..." if len(resp_text) > 200 else ""),
                    )

                return response

            except Exception as e:
                print(f"Error during generation: {e}")
                traceback.print_exc()
                mx.clear_cache()
                gc.collect()
                raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Unexpected error in /responses endpoint: {e}")
        traceback.print_exc()
        mx.clear_cache()
        gc.collect()
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        )


@app.post("/chat/completions", response_model=None)
@app.post("/v1/chat/completions", response_model=None, include_in_schema=False)
async def chat_completions_endpoint(request: ChatRequest):
    """
    Generate text based on a prompt and optional images.
    Prompt must be a list of chat messages, including system, user, and assistant messages.
    System message will be ignored if not already in the prompt.
    Can operate in streaming or non-streaming mode.
    """

    request_start = time.perf_counter()
    try:
        adapter_path = (
            request.adapter_path
            if "adapter_path" in request.model_fields_set
            else _INHERIT_ADAPTER
        )
        model, processor, config = get_cached_model(request.model, adapter_path)

        kwargs = {}

        if request.resize_shape is not None:
            if len(request.resize_shape) not in [1, 2]:
                raise HTTPException(
                    status_code=400,
                    detail="resize_shape must contain exactly two integers (height, width)",
                )
            kwargs["resize_shape"] = (
                (request.resize_shape[0],) * 2
                if len(request.resize_shape) == 1
                else tuple(request.resize_shape)
            )

        images = []
        audio = []
        processed_messages = []
        for message in request.messages:
            msg = {"role": message.role}

            if isinstance(message.content, str):
                msg["content"] = message.content
            elif isinstance(message.content, list):
                text_content = ""
                for item in message.content:
                    if isinstance(item, dict):
                        if message.role == "user":
                            if item["type"] == "input_image":
                                images.append(item["image_url"])
                            elif item["type"] == "image_url":
                                images.append(item["image_url"]["url"])
                            elif item["type"] == "input_audio":
                                audio.append(item["input_audio"]["data"])
                        if item["type"] in ("text", "input_text"):
                            text_content = item.get("text", "")
                msg["content"] = text_content
            else:
                msg["content"] = message.content

            # Preserve tool-calling metadata.
            # Ensure arguments are dicts (not JSON strings) for Jinja templates
            # that iterate them with |items (e.g. Qwen3.5).
            if message.tool_calls is not None:
                normalized_calls = []
                for tc in message.tool_calls:
                    tc = dict(tc) if isinstance(tc, dict) else tc
                    if isinstance(tc, dict) and "function" in tc:
                        fn = dict(tc["function"])
                        args = fn.get("arguments", {})
                        if isinstance(args, str):
                            try:
                                fn["arguments"] = json.loads(args)
                            except (json.JSONDecodeError, TypeError):
                                fn["arguments"] = {}
                        tc["function"] = fn
                    normalized_calls.append(tc)
                msg["tool_calls"] = normalized_calls
            if message.tool_call_id is not None:
                msg["tool_call_id"] = message.tool_call_id
            if message.name is not None:
                msg["name"] = message.name

            processed_messages.append(msg)

        # Detect tool parser from chat template
        tools = getattr(request, "tools", None)
        tool_parser_type = None
        tool_module = None
        tokenizer = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
            tool_parser_type = _infer_tool_parser(tokenizer.chat_template)
            if tool_parser_type is not None:
                tool_module = load_tool_module(tool_parser_type)

        gen_args = _build_gen_args(request)
        template_kwargs = gen_args.to_template_kwargs()

        formatted_prompt = apply_chat_template(
            processor,
            config,
            processed_messages,
            num_images=len(images),
            num_audios=len(audio),
            tools=tools,
            **template_kwargs,
        )

        logger.debug(
            "chat/completions request: model=%s images=%d audio=%d "
            "max_tokens=%s temp=%s stream=%s",
            request.model,
            len(images),
            len(audio),
            gen_args.max_tokens,
            gen_args.temperature,
            request.stream,
        )
        generation_kwargs = build_generation_kwargs(request, gen_args, template_kwargs)
        _should_strip_thinking = not template_kwargs.get("enable_thinking", False)

        # PrefixCache 통합 (2026-05-05): warmup 된 prefix가 있고 텍스트만 요청이면
        # ResponseGenerator(continuous batching) 우회하고 stream_generate fallback 사용
        # → fallback path에 prefix_cache.try_restore() 통합되어 있음
        _prefer_prefix_cache = (
            prefix_cache.is_ready
            and not images
            and not audio
        )

        if request.stream:
            # Streaming response using ResponseGenerator for continuous batching
            async def stream_generator():
                global response_generator
                token_iterator = None
                token_iter = None  # For ResponseGenerator cleanup
                try:
                    # Use ResponseGenerator if available, otherwise fall back to stream_generate
                    if response_generator is not None and not _prefer_prefix_cache:
                        # generate() does blocking Queue.get — run off event loop
                        ctx, token_iter = await asyncio.to_thread(
                            response_generator.generate,
                            formatted_prompt,
                            images if images else None,
                            audio if audio else None,
                            gen_args,
                        )

                        output_tokens = 0
                        request_id = f"chatcmpl-{uuid.uuid4()}"
                        usage_stats = {}
                        # Track thinking state for reasoning/content split
                        in_thinking = False
                        accumulated = ""
                        full_output = ""  # raw output for tool call parsing
                        # Track tool-call state to suppress markup from content
                        in_tool_call = False
                        tc_start = tool_module.tool_call_start if tool_module else None
                        tc_end = tool_module.tool_call_end if tool_module else None

                        def _next_token():
                            try:
                                return next(token_iter)
                            except StopIteration:
                                return None

                        while True:
                            token = await asyncio.to_thread(_next_token)
                            if token is None:
                                break
                            output_tokens += 1
                            accumulated += token.text
                            full_output += token.text

                            # Detect thinking boundaries
                            delta_reasoning = None
                            delta_content = None

                            if not in_thinking and (
                                "<|channel>thought" in accumulated
                                or "<think>" in accumulated
                            ):
                                in_thinking = True
                                accumulated = ""
                                # Don't emit opening tag tokens
                            elif in_thinking and (
                                "<channel|>" in accumulated or "</think>" in accumulated
                            ):
                                in_thinking = False
                                accumulated = ""
                                # Don't emit closing tag tokens
                            elif in_thinking:
                                delta_reasoning = token.text
                            elif not in_thinking and (
                                "<|channel>" in accumulated or "<think" in accumulated
                            ):
                                pass  # Partial tag, don't emit yet
                            else:
                                delta_content = token.text

                            # Suppress tool-call markup from content
                            in_tool_call, delta_content = suppress_tool_call_content(
                                full_output, in_tool_call, tc_start, delta_content
                            )

                            chunk_logprobs = None
                            if request.logprobs and token.finish_reason != "stop":
                                req_top_k = int(request.top_logprobs or 0)
                                chunk_logprobs = ChatLogprobs(
                                    content=[
                                        _make_logprob_content(
                                            response_generator.tokenizer,
                                            token.token,
                                            token.logprobs,
                                            top_logprobs=token.top_logprobs,
                                            top_k=req_top_k,
                                        )
                                    ]
                                )

                            # Skip empty deltas (e.g. suppressed tool-call tokens)
                            has_payload = (
                                delta_content is not None
                                or delta_reasoning is not None
                                or token.finish_reason is not None
                                or chunk_logprobs is not None
                            )
                            if has_payload:
                                choices = [
                                    ChatStreamChoice(
                                        finish_reason=token.finish_reason,
                                        delta=ChatMessage(
                                            role="assistant",
                                            content=delta_content,
                                            reasoning=delta_reasoning,
                                        ),
                                        logprobs=chunk_logprobs,
                                    )
                                ]
                                chunk_data = ChatStreamChunk(
                                    id=request_id,
                                    created=int(time.time()),
                                    model=request.model,
                                    usage={
                                        "prompt_tokens": ctx.prompt_tokens,
                                        "completion_tokens": output_tokens,
                                        "total_tokens": ctx.prompt_tokens
                                        + output_tokens,
                                    },
                                    choices=choices,
                                )

                                yield f"data: {chunk_data.model_dump_json()}\n\n"

                            if token.finish_reason:
                                break

                        # Parse tool calls from full output and emit final chunk
                        if tool_module is not None:
                            tc = process_tool_calls(full_output, tool_module, tools)
                            if tc["calls"]:
                                choices = [
                                    ChatStreamChoice(
                                        finish_reason="tool_calls",
                                        delta=ChatMessage(
                                            role="assistant",
                                            tool_calls=tc["calls"],
                                        ),
                                    )
                                ]
                                chunk_data = ChatStreamChunk(
                                    id=request_id,
                                    created=int(time.time()),
                                    model=request.model,
                                    choices=choices,
                                )
                                yield f"data: {chunk_data.model_dump_json()}\n\n"
                    else:
                        # Fallback to stream_generate (fork: prefix_cache + mtp + thinking strip)
                        token_iterator = stream_generate(
                            model=model,
                            processor=processor,
                            prompt=formatted_prompt,
                            image=images,
                            audio=audio,
                            vision_cache=model_cache.get("vision_cache"),
                            prefix_cache=prefix_cache,
                            mtp=os.environ.get("MLX_MTP", "").lower() in ("1", "true"),
                            **generation_kwargs,
                        )

                        request_id = f"chatcmpl-{uuid.uuid4()}"
                        output_text = ""
                        output_tokens = 0
                        usage_stats = {}
                        _tf_emitted = 0  # thinking strip: emit한 바이트 수
                        _tf_raw = ""     # 전체 raw 텍스트
                        for chunk in token_iterator:
                            if chunk is None or not hasattr(chunk, "text"):
                                continue

                            usage_stats = {
                                "input_tokens": chunk.prompt_tokens,
                                "output_tokens": chunk.generation_tokens,
                                "total_tokens": chunk.prompt_tokens
                                + chunk.generation_tokens,
                                "prompt_tps": chunk.prompt_tps,
                                "generation_tps": chunk.generation_tps,
                                "peak_memory": chunk.peak_memory,
                            }
                            output_tokens = chunk.generation_tokens

                            if _should_strip_thinking:
                                _tf_raw += chunk.text
                                clean = _strip_thinking(_tf_raw)
                                chunk_text = clean[_tf_emitted:]
                                if not chunk_text:
                                    continue
                                _tf_emitted = len(clean)
                            else:
                                chunk_text = chunk.text

                            if not chunk_text:
                                continue
                            output_text += chunk_text

                            choices = [
                                ChatStreamChoice(
                                    delta=ChatMessage(
                                        role="assistant", content=chunk_text
                                    )
                                )
                            ]
                            chunk_data = ChatStreamChunk(
                                id=request_id,
                                created=int(time.time()),
                                model=request.model,
                                usage={
                                    "prompt_tokens": chunk.prompt_tokens,
                                    "completion_tokens": chunk.generation_tokens,
                                    "total_tokens": chunk.prompt_tokens
                                    + chunk.generation_tokens,
                                },
                                choices=choices,
                            )

                            yield f"data: {chunk_data.model_dump_json(exclude_none=True)}\n\n"
                            await asyncio.sleep(0.01)

                    # Record last_request summary + log (fork)
                    try:
                        _elapsed = time.time() - _busy_since if _busy_since else 0
                        _record_last_request(
                            prompt_tokens=usage_stats.get("input_tokens", 0) if isinstance(usage_stats, dict) else 0,
                            gen_tokens=usage_stats.get("output_tokens", output_tokens) if isinstance(usage_stats, dict) else output_tokens,
                            elapsed_s=_elapsed,
                            prompt_tps=usage_stats.get("prompt_tps", 0) if isinstance(usage_stats, dict) else 0,
                            gen_tps=usage_stats.get("generation_tps", 0) if isinstance(usage_stats, dict) else 0,
                        )
                        _log_request(
                            processed_messages,
                            prompt_tokens=usage_stats.get("input_tokens", 0) if isinstance(usage_stats, dict) else 0,
                            gen_tokens=usage_stats.get("output_tokens", output_tokens) if isinstance(usage_stats, dict) else output_tokens,
                            elapsed_s=_elapsed,
                            gen_tps=usage_stats.get("generation_tps", 0) if isinstance(usage_stats, dict) else 0,
                            stream=True,
                            mtp=os.environ.get("MLX_MTP", "").lower() in ("1", "true"),
                        )
                    except Exception:
                        pass

                    # Signal stream end
                    yield "data: [DONE]\n\n"

                    elapsed = time.perf_counter() - request_start
                    logger.debug(
                        "chat/completions stream done: tokens=%d " "total_time=%.2fs",
                        output_tokens,
                        elapsed,
                    )

                except Exception as e:
                    print(f"Error during stream generation: {e}")
                    traceback.print_exc()
                    _log_request(
                        processed_messages, stream=True, error=str(e),
                    )
                    error_data = json.dumps({"error": str(e)})
                    yield f"data: {error_data}\n\n"

                finally:
                    # Close the token iterator to trigger cleanup (important for ResponseGenerator)
                    if token_iter is not None:
                        try:
                            token_iter.close()
                        except Exception:
                            pass
                    mx.clear_cache()
                    gc.collect()

            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        else:
            # Non-streaming response
            try:
                _t0 = time.time()
                full_text = ""
                prompt_tokens = 0
                output_tokens = 0
                peak_memory = 0.0
                prompt_tps = 0.0
                gen_tps = 0.0

                collected_logprobs: List[
                    Tuple[int, float, Optional[List[Tuple[int, float]]]]
                ] = []

                if response_generator is not None and not _prefer_prefix_cache:

                    def _blocking_generate():
                        text = ""
                        pt = gt = 0
                        pm = 0.0
                        ctx, token_iter = response_generator.generate(
                            prompt=formatted_prompt,
                            images=images if images else None,
                            audio=audio if audio else None,
                            args=gen_args,
                        )
                        pt = ctx.prompt_tokens
                        for token in token_iter:
                            text += token.text
                            gt += 1
                            pm = token.peak_memory
                            if request.logprobs and token.finish_reason != "stop":
                                collected_logprobs.append(
                                    (token.token, token.logprobs, token.top_logprobs)
                                )
                            if token.finish_reason:
                                break
                        try:
                            token_iter.close()
                        except Exception:
                            pass
                        return text, pt, gt, pm

                    full_text, prompt_tokens, output_tokens, peak_memory = (
                        await asyncio.to_thread(_blocking_generate)
                    )
                else:
                    gen_result = generate(
                        model=model,
                        processor=processor,
                        prompt=formatted_prompt,
                        image=images,
                        audio=audio,
                        verbose=logger.isEnabledFor(logging.DEBUG),
                        vision_cache=model_cache.get("vision_cache"),
                        prefix_cache=prefix_cache,
                        **gen_args.to_generate_kwargs(),
                        **kwargs,
                    )
                    if _should_strip_thinking:
                        gen_result.text = _strip_thinking(gen_result.text)
                    full_text = gen_result.text
                    prompt_tokens = gen_result.prompt_tokens
                    output_tokens = gen_result.generation_tokens
                    peak_memory = gen_result.peak_memory
                    prompt_tps = getattr(gen_result, "prompt_tps", 0.0)
                    gen_tps = getattr(gen_result, "generation_tps", 0.0)

                _elapsed = time.time() - _t0
                _record_last_request(
                    prompt_tokens=prompt_tokens,
                    gen_tokens=output_tokens,
                    elapsed_s=_elapsed,
                    prompt_tps=prompt_tps,
                    gen_tps=gen_tps,
                )
                _log_request(
                    processed_messages,
                    prompt_tokens=prompt_tokens,
                    gen_tokens=output_tokens,
                    elapsed_s=_elapsed,
                    gen_tps=gen_tps,
                    stream=False,
                )

                mx.clear_cache()
                gc.collect()

                reasoning, content = _split_thinking(full_text)

                # Count raw generated tokens minus thinking tag tokens
                completion_tokens = output_tokens - _count_thinking_tag_tokens(
                    full_text
                )

                usage_stats = UsageStats(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    peak_memory=peak_memory,
                )

                # Parse tool calls from generated output
                parsed_tool_calls = None
                if tool_module is not None:
                    tc = process_tool_calls(
                        model_output=full_text,
                        tool_module=tool_module,
                        tools=tools,
                    )
                    if tc["calls"]:
                        parsed_tool_calls = tc["calls"]
                        # Clean thinking tags and control tokens from remaining text
                        _, clean_remaining = _split_thinking(tc["remaining_text"] or "")
                        if clean_remaining:
                            # Strip model control tokens
                            clean_remaining = re.sub(
                                r"<\|[^>]+\|>|<[^>]+>", "", clean_remaining
                            ).strip()
                        content = clean_remaining or None

                response_logprobs = None
                if request.logprobs and collected_logprobs:
                    tokenizer = (
                        processor.tokenizer
                        if hasattr(processor, "tokenizer")
                        else processor
                    )
                    req_top_k = int(request.top_logprobs or 0)
                    response_logprobs = ChatLogprobs(
                        content=[
                            _make_logprob_content(
                                tokenizer,
                                tid,
                                lp,
                                top_logprobs=top_lps,
                                top_k=req_top_k,
                            )
                            for tid, lp, top_lps in collected_logprobs
                        ]
                    )

                choices = [
                    ChatChoice(
                        finish_reason="tool_calls" if parsed_tool_calls else "stop",
                        message=ChatMessage(
                            role="assistant",
                            content=content if content else None,
                            reasoning=reasoning,
                            tool_calls=parsed_tool_calls,
                        ),
                        logprobs=response_logprobs,
                    )
                ]
                result = ChatResponse(
                    id=f"chatcmpl-{uuid.uuid4()}",
                    created=int(time.time()),
                    model=request.model,
                    usage=usage_stats,
                    choices=choices,
                )

                elapsed = time.perf_counter() - request_start
                logger.debug(
                    "chat/completions done: prompt_tokens=%d completion_tokens=%d "
                    "total_time=%.2fs peak_memory=%.2fGB",
                    prompt_tokens,
                    completion_tokens,
                    elapsed,
                    peak_memory,
                )
                if logger.isEnabledFor(logging.DEBUG):
                    resp_text = content or ""
                    logger.debug(
                        "  response: %s",
                        resp_text[:200] + ("..." if len(resp_text) > 200 else ""),
                    )

                return result

            except Exception as e:
                print(f"Error during generation: {e}")
                traceback.print_exc()
                _log_request(processed_messages, stream=False, error=str(e))
                mx.clear_cache()
                gc.collect()
                raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    except HTTPException as http_exc:
        # Re-raise HTTP exceptions (like model loading failure)
        raise http_exc
    except Exception as e:
        # Catch unexpected errors
        print(f"Unexpected error in /generate endpoint: {e}")
        traceback.print_exc()
        mx.clear_cache()
        gc.collect()
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        )


@app.get("/models", response_model=ModelsResponse)
@app.get("/v1/models", response_model=ModelsResponse, include_in_schema=False)
def models_endpoint():
    """
    현재 로드된(pinned) 모델을 맨 앞에, 그 뒤에 HF cache 의 MLX 모델들을 나열.

    OpenAI 호환 클라이언트들이 첫 모델을 기본으로 쓰기 때문에 실제 구동 중인
    모델이 가장 먼저 나와야 한다. upstream 기본 구현은 HF cache scan 만 하므로
    `--model /absolute/path` 로 로드한 경우 응답 목록에서 누락되는 문제를 고친다.
    """

    files = ["config.json", "model.safetensors.index.json", "tokenizer_config.json"]

    def probably_mlx_lm(repo):
        if repo.repo_type != "model":
            return False
        if "main" not in repo.refs:
            return False
        file_names = {f.file_path.name for f in repo.refs["main"].files}
        return all(f in file_names for f in files)

    # Scan the cache directory for downloaded mlx models
    hf_cache_info = scan_cache_dir()
    downloaded_models = [repo for repo in hf_cache_info.repos if probably_mlx_lm(repo)]

    # Create a list of available models
    models = [
        {"id": repo.repo_id, "object": "model", "created": int(repo.last_modified)}
        for repo in downloaded_models
    ]

    # 현재 실제 로드된 모델을 맨 앞에 삽입 (중복 시 제거 후 재삽입)
    loaded_path = model_cache.get("model_path") if isinstance(model_cache, dict) else None
    if loaded_path:
        # 로컬 절대경로면 디렉토리 이름을 id 로, HF repo id 형식이면 그대로
        if loaded_path.startswith("/"):
            loaded_id = Path(loaded_path).name
        else:
            loaded_id = loaded_path
        # 캐시 스캔 결과에 동일 id 가 이미 있으면 제거
        models = [m for m in models if m["id"] != loaded_id]
        loaded_entry = {
            "id": loaded_id,
            "object": "model",
            "created": int(time.time()),
        }
        models.insert(0, loaded_entry)

    response = {"object": "list", "data": models}

    return response


# MLX_VLM API endpoints


@app.get("/health")
async def health_check():
    """
    Check if the server is healthy and what model is loaded.
    """
    return {
        "status": "healthy",
        "loaded_model": model_cache.get("model_path", None),
        "loaded_adapter": model_cache.get("adapter_path", None),
        "continuous_batching_enabled": response_generator is not None,
    }


@app.get("/v1/status")
async def server_status():
    """Return detailed server status: busy, memory, model, and prefix cache stats."""
    try:
        active = mx.get_active_memory()
        cache_mem = mx.get_cache_memory()
        peak = mx.get_peak_memory()
    except AttributeError:
        active = mx.metal.get_active_memory()
        cache_mem = mx.metal.get_cache_memory()
        peak = mx.metal.get_peak_memory()

    with _inflight_lock:
        busy = _inflight > 0
        inflight = _inflight
        last_done = _last_done_ts
        busy_start = _busy_since

    now = time.time()
    idle_sec = round(now - last_done, 1) if last_done > 0 else None
    busy_sec = round(now - busy_start, 1) if busy and busy_start > 0 else None

    result = {
        "busy": busy,
        "inflight": inflight,
        "idle_seconds": idle_sec,
        "busy_seconds": busy_sec,
        "model": model_cache.get("model_path", None),
        "memory": {
            "active_mb": round(active / 1e6, 1),
            "cache_mb": round(cache_mem / 1e6, 1),
            "peak_mb": round(peak / 1e6, 1),
            "limit_gb": _metal_mem_limit_gb,
            "cache_limit_gb": _metal_cache_limit_gb,
        },
        "prompt_cache": prefix_cache.stats(),
    }
    if _last_request is not None:
        result["last_request"] = _last_request
    return result


@app.post("/v1/warmup")
async def warmup_endpoint(request: dict):
    """Prefill a system prompt and cache the KV/SSM state for TTFT optimization.

    This is particularly useful for hybrid models (e.g. Qwen3.5 with
    Attention + Mamba layers) where mlx-lm's LRUPromptCache is broken.

    Request body::

        {
            "messages": [{"role": "system", "content": "..."}],
            "tools": [...]  // optional
        }

    The stable prefix (tokens shared across different user messages) is
    detected automatically by comparing two prompt variants.  Subsequent
    ``/v1/chat/completions`` requests whose token sequence starts with
    the cached prefix will skip the prefix prefill entirely.
    """
    try:
        model, processor, config = get_cached_model(
            model_cache.get("model_path", request.get("model", ""))
        )
        messages = request.get("messages", [])
        tools = request.get("tools")
        template_kwargs = {}
        if tools:
            template_kwargs["tools"] = tools

        # Detect stable prefix by comparing two variants with different user messages
        system_msgs = [m for m in messages if m.get("role") == "system"]
        if not system_msgs:
            return {"success": False, "error": "no system messages found"}

        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

        variant_a = system_msgs + [{"role": "user", "content": "\u03b1"}]
        variant_b = system_msgs + [{"role": "user", "content": "\u03b2 \u03b3 \u03b4 \u03b5 \u03b6 \u03b7 \u03b8 \u03b9 \u03ba \u03bb"}]

        prompt_a = apply_chat_template(processor, config, variant_a, **template_kwargs)
        prompt_b = apply_chat_template(processor, config, variant_b, **template_kwargs)

        tokens_a = tokenizer.encode(prompt_a)
        tokens_b = tokenizer.encode(prompt_b)

        plen = 0
        for a, b in zip(tokens_a, tokens_b):
            if a == b:
                plen += 1
            else:
                break

        if plen < 20:
            return {"success": False, "error": f"prefix too short ({plen} tokens)"}

        prefix_tokens = tokens_a[:plen]
        prefix_cache.warmup(model, prefix_tokens)

        return {
            "success": True,
            "prefix_tokens": plen,
            "warmup_ms": prefix_cache.last_warmup_ms,
        }
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.get("/v1/prefix_cache/stats")
async def prefix_cache_stats():
    """Return prefix cache hit/miss statistics."""
    return prefix_cache.stats()


@app.post("/unload")
async def unload_model_endpoint():
    """
    Unload the currently loaded model from memory.
    """
    unloaded_info = {
        "model_name": model_cache.get("model_path", None),
        "adapter_name": model_cache.get("adapter_path", None),
    }

    if not unload_model_sync():  # Use the synchronous unload function
        return {"status": "no_model_loaded", "message": "No model is currently loaded"}

    return {
        "status": "success",
        "message": f"Model unloaded successfully",
        "unloaded": unloaded_info,
    }


def main():
    parser = argparse.ArgumentParser(description="MLX VLM Http Server.")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host for the HTTP server (default:0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the HTTP server (default: 8080)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading models from Hugging Face Hub.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Pre-load a model at startup (e.g. mlx-community/Qwen2.5-VL-3B-Instruct-4bit).",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Adapter weights to load with the model.",
    )
    parser.add_argument(
        "--vision-cache-size",
        type=int,
        default=20,
        help="Max number of cached vision features (default: 20).",
    )
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=DEFAULT_PREFILL_STEP_SIZE,
        help="Tokens per prefill step (default: %(default)s).",
    )
    parser.add_argument(
        "--kv-bits",
        type=float,
        default=None,
        help="Number of bits for KV cache quantization (e.g. 3.5 for TurboQuant).",
    )
    parser.add_argument(
        "--kv-quant-scheme",
        type=str,
        choices=("uniform", "turboquant"),
        default=DEFAULT_KV_QUANT_SCHEME,
        help="KV cache quantization backend.",
    )
    parser.add_argument(
        "--kv-group-size",
        type=int,
        default=DEFAULT_KV_GROUP_SIZE,
        help="Group size for uniform KV cache quantization.",
    )
    parser.add_argument(
        "--max-kv-size",
        type=int,
        default=None,
        help="Maximum KV cache size in tokens.",
    )
    parser.add_argument(
        "--quantized-kv-start",
        type=int,
        default=DEFAULT_QUANTIZED_KV_START,
        help="Start index for quantized KV cache.",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        default=None,
        help="Speculative drafter path or HF id (e.g. z-lab/Qwen3.5-4B-DFlash).",
    )
    parser.add_argument(
        "--draft-kind",
        type=str,
        default="dflash",
        help="Drafter family (default: dflash).",
    )
    parser.add_argument(
        "--draft-block-size",
        type=int,
        default=None,
        help="Override the drafter's configured block size.",
    )
    parser.add_argument(
        "--top-logprobs-k",
        type=int,
        default=None,
        help=(
            "Server-side cap for per-token top_logprobs (0-20, default 0 = "
            "disabled). Maps to the TOP_LOGPROBS_K env var."
        ),
    )
    parser.add_argument(
        "--pin-model",
        action="store_true",
        default=False,
        help="Pin the loaded model and block swap attempts from API requests.",
    )
    parser.add_argument(
        "--mtp",
        action="store_true",
        default=False,
        help="Use native Multi-Token Prediction for speculative decoding "
        "(requires a model with an MTP head, e.g. Qwen3.5-27B-MTP).",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="Enable auto-reload for development.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO).",
    )
    args = parser.parse_args()
    if args.trust_remote_code:
        os.environ["MLX_TRUST_REMOTE_CODE"] = "true"
    if args.model:
        os.environ["MLX_VLM_PRELOAD_MODEL"] = args.model
        if args.adapter_path:
            os.environ["MLX_VLM_PRELOAD_ADAPTER"] = args.adapter_path
    if args.pin_model:
        os.environ["MLX_PIN_MODEL"] = "true"
    if args.mtp:
        os.environ["MLX_MTP"] = "true"
    os.environ["MLX_VLM_VISION_CACHE_SIZE"] = str(args.vision_cache_size)
    if args.draft_model:
        os.environ["MLX_VLM_DRAFT_MODEL"] = args.draft_model
        os.environ["MLX_VLM_DRAFT_KIND"] = args.draft_kind
        if args.draft_block_size is not None:
            os.environ["MLX_VLM_DRAFT_BLOCK_SIZE"] = str(args.draft_block_size)
    if args.prefill_step_size:
        os.environ["PREFILL_STEP_SIZE"] = str(args.prefill_step_size)
    if args.kv_bits is not None:
        os.environ["KV_BITS"] = str(args.kv_bits)
    os.environ["KV_GROUP_SIZE"] = str(args.kv_group_size)
    os.environ["KV_QUANT_SCHEME"] = args.kv_quant_scheme
    if args.max_kv_size is not None:
        os.environ["MAX_KV_SIZE"] = str(args.max_kv_size)
    os.environ["QUANTIZED_KV_START"] = str(args.quantized_kv_start)
    if args.top_logprobs_k is not None:
        os.environ["TOP_LOGPROBS_K"] = str(args.top_logprobs_k)

    # Configure logging
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger.setLevel(log_level)

    uvicorn.run(
        "mlx_vlm.server:app",
        host=args.host,
        port=args.port,
        workers=1,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
