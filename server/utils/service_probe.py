"""Pre-flight health checks for translation services.

The plugin can request ``service=auto``, which means: probe the services this
machine is actually configured for, then use the first responsive one.

Probing runs in the *translation* interpreter (the managed pdf2zh_next venv),
because only that process can build the exact translator objects with the exact
credentials the real run will use. A probe that passes on a different HTTP stack
than the one used for translation proves nothing.

Each probe performs one tiny real translation. That is heavier than a TCP
connect, but it is the only check that catches the failure mode this project
keeps hitting: an endpoint that accepts connections and then rejects every
actual translation request.

Timing has two distinct costs, so they are budgeted separately:

* importing pdf2zh_next costs ~10s and happens once per probe run;
* one translation attempt against a dead provider can stall for minutes, because
  the engine retries with long backoffs.

The probe therefore imports once in a single child process and gives every
service its own deadline. A service that overruns is abandoned (its worker
thread is a daemon and dies with the child) and the next candidate is tried.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

from utils.config_map import pdf2zh_next_service_aliases

# Attempt order. Only a tie-breaker among services that pass: LLM providers come
# before machine translation, and free/rate-limited services come last.
DEFAULT_PROBE_ORDER = [
    'openailiked',
    'deepseek',
    'openai',
    'silicon',
    'zhipu',
    'modelscope',
    'gemini',
    'groq',
    'grok',
    'qwen-mt',
    'AliyunDashScope',
    'azure-openai',
    'ollama',
    'xinference',
    'deepl',
    'bing',
    'google',
    'siliconflowfree',
]

# Services whose config section requires a user-supplied secret. If the field is
# still the template placeholder, skip without any network round trip.
_REQUIRES_CREDENTIAL = {
    'openailiked': 'openai_compatible_api_key',
    'deepseek': 'deepseek_api_key',
    'openai': 'openai_api_key',
    'silicon': 'siliconflow_api_key',
    'zhipu': 'zhipu_api_key',
    'modelscope': 'modelscope_api_key',
    'gemini': 'gemini_api_key',
    'groq': 'groq_api_key',
    'grok': 'grok_api_key',
    'qwen-mt': 'qwenmt_api_key',
    'AliyunDashScope': 'aliyun_dashscope_api_key',
    'azure-openai': 'azure_openai_api_key',
    'deepl': 'deepl_auth_key',
    'azure': 'azure_api_key',
    'anythingllm': 'anythingllm_apikey',
    'dify': 'dify_apikey',
    'tencent': 'tencentcloud_secret_key',
}

# Engine class names for services that take no settings at all. config.toml
# carries only an on/off flag for these, so there is no `*_detail` section to
# read the engine type from. Without this table they would be silently skipped,
# even though they are the only services that work on a fresh install.
_ENGINE_TYPE_DEFAULTS = {
    'bing': 'Bing',
    'google': 'Google',
    'siliconflowfree': 'SiliconFlowFree',
}

# Never auto-selected: engines that shell out to external binaries.
_NEVER_AUTO = {'claudecode', 'clitranslator'}

PROBE_TEXT = 'Hello'
# Per-service deadline. A healthy provider answers in a few seconds; anything
# slower than this is unusable for translating a paper anyway.
SERVICE_TIMEOUT_SECONDS = 20
# Engine import alone is ~10s on a cold cache; give it room without letting it
# dominate the whole budget.
IMPORT_TIMEOUT_SECONDS = 120

# Runs inside the translation venv. A single string so the parent interpreter
# does not need pdf2zh_next importable.
_PROBE_SCRIPT = r'''
import json
import sys
import threading

payload = json.loads(sys.argv[1])
candidates = payload['candidates']
text = payload['text']
translation = payload['translation']
per_service_timeout = float(payload['service_timeout'])

results = {}

try:
    from pdf2zh_next.config.model import SettingsModel, TranslationSettings
    from pdf2zh_next.config.translate_engine_model import (
        TRANSLATION_ENGINE_METADATA_MAP,
    )
    from pdf2zh_next.translator.utils import get_translator
except BaseException as exc:
    print(json.dumps({
        'import_error': f'{type(exc).__name__}: {exc}'[:400],
    }, ensure_ascii=False))
    sys.exit(0)

metadata_map = TRANSLATION_ENGINE_METADATA_MAP


def probe(candidate, slot):
    try:
        metadata = metadata_map.get(candidate['engine_type'])
        if metadata is None:
            slot['error'] = 'engine type not registered: ' + candidate['engine_type']
            return
        settings = SettingsModel(
            translate_engine_settings=metadata.setting_model_type(
                **candidate['settings']
            ),
            translation=TranslationSettings(**translation),
        )
        translator = get_translator(settings)
        out = translator.translate(text, ignore_cache=True)
        if not (out and str(out).strip()):
            slot['error'] = 'empty translation result'
            return
        slot['ok'] = True
        slot['sample'] = str(out)[:60]
    except BaseException as exc:
        slot['error'] = f'{type(exc).__name__}: {exc}'[:300]


import time


def probe_with_deadline(candidate):
    slot = {'ok': False}
    worker = threading.Thread(target=probe, args=(candidate, slot), daemon=True)
    started = time.time()
    worker.start()
    worker.join(per_service_timeout)
    if worker.is_alive():
        slot['ok'] = False
        slot['error'] = 'timeout after %.0fs' % (time.time() - started)
    results[candidate['service']] = slot
    return slot


ordered = []
for candidate in candidates:
    slot = probe_with_deadline(candidate)
    item = {'service': candidate['service']}
    item.update(slot)
    ordered.append(item)
    print('PROBE ' + json.dumps(item, ensure_ascii=False), flush=True)

print('RESULT ' + json.dumps(ordered, ensure_ascii=False), flush=True)
'''


def _section_name(service: str) -> str:
    return pdf2zh_next_service_aliases.get(service, service)


def _is_placeholder(value) -> bool:
    """Engine templates spell "not configured" as the literal string "null"."""
    if value is None:
        return True
    if isinstance(value, str):
        text = value.strip()
        return text == '' or text.lower() in {'null', 'none', 'your_api_key'}
    return False


def _clean_settings(detail: dict) -> dict:
    """Keep only fields the engine setting model accepts."""
    settings = {}
    for key, value in (detail or {}).items():
        if key in {'translate_engine_type', 'support_llm'}:
            continue
        if isinstance(value, str) and value.strip().lower() == 'null':
            continue
        settings[key] = value
    return settings


def collect_candidates(config: dict) -> list[dict]:
    """Return probe candidates in attempt order, skipping unconfigured ones."""
    candidates = []
    for service in DEFAULT_PROBE_ORDER:
        if service in _NEVER_AUTO:
            continue

        section = config.get(f'{_section_name(service)}_detail')
        detail = section if isinstance(section, dict) else {}

        engine_type = detail.get('translate_engine_type')
        if not (isinstance(engine_type, str) and engine_type.strip()):
            engine_type = _ENGINE_TYPE_DEFAULTS.get(service)
        if not engine_type:
            continue

        # A detail section exists but the service was never configured. Flag-only
        # services have no section at all, so they are not affected here.
        credential_field = _REQUIRES_CREDENTIAL.get(service)
        if credential_field and _is_placeholder(detail.get(credential_field)):
            continue

        candidates.append(
            {
                'service': service,
                'engine_type': str(engine_type).strip(),
                'settings': _clean_settings(detail),
            }
        )
    return candidates


def _resolve_python(env_manager) -> tuple[str | None, dict | None]:
    """Resolve the *pdf2zh_next* interpreter, ignoring the legacy engine.

    ``VirtualEnvManager.get_command_and_env`` infers the engine from the command
    it is given and picks whichever env that maps to. Probing must always use the
    new engine's environment, so resolve that environment explicitly instead.
    """
    if env_manager is None:
        return sys.executable, None

    try:
        from utils.environment_lifecycle import (
            environment_path_entries,
            find_existing_environment,
            managed_python_env,
        )

        existing = find_existing_environment('pdf2zh_next', getattr(env_manager, 'default_env_tool', 'auto'))
        if not existing:
            return None, None
        env_tool, env_dir, python_path = existing

        env = managed_python_env()
        env['PYTHONUNBUFFERED'] = '1'
        env['PYTHONIOENCODING'] = 'utf-8'
        entries = environment_path_entries(env_dir, env_tool) or [env_dir]
        current = env.get('PATH', '')
        env['PATH'] = os.pathsep.join(
            [str(p) for p in entries] + ([current] if current else [])
        )
        return str(python_path), env
    except Exception:
        return None, None


def probe_services(candidates: list[dict], lang_in='en', lang_out='zh-CN',
                   env_manager=None, text=PROBE_TEXT,
                   service_timeout=SERVICE_TIMEOUT_SECONDS,
                   import_timeout=IMPORT_TIMEOUT_SECONDS) -> list[dict]:
    """Probe candidates sequentially in one child process.

    Returns one result dict per candidate, in candidate order, so the caller can
    honour attempt order even when several services work.
    """
    if not candidates:
        return []

    python_path, env = _resolve_python(env_manager)
    if not python_path:
        return [
            {'service': c['service'], 'ok': False, 'error': 'no python interpreter'}
            for c in candidates
        ]

    payload = {
        'candidates': candidates,
        'text': text,
        'translation': {'lang_in': lang_in, 'lang_out': lang_out, 'qps': 1000},
        'service_timeout': service_timeout,
    }

    child_env = dict(env if env is not None else os.environ)
    child_env['PYTHONUNBUFFERED'] = '1'
    child_env['PYTHONIOENCODING'] = 'utf-8'

    creationflags = (
        getattr(subprocess, 'CREATE_NO_WINDOW', 0) if sys.platform == 'win32' else 0
    )

    total_budget = import_timeout + service_timeout * len(candidates) + 15

    try:
        completed = subprocess.run(
            [python_path, '-c', _PROBE_SCRIPT, json.dumps(payload, ensure_ascii=False)],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=total_budget,
            env=child_env,
            creationflags=creationflags,
        )
    except subprocess.TimeoutExpired as exc:
        return _partial_results(candidates, exc.stdout)
    except Exception as exc:
        return [
            {
                'service': c['service'],
                'ok': False,
                'error': f'{type(exc).__name__}: {exc}'[:300],
            }
            for c in candidates
        ]

    stdout = completed.stdout or ''
    for line in reversed(stdout.strip().splitlines()):
        if line.startswith('RESULT '):
            try:
                parsed = json.loads(line[len('RESULT '):])
            except Exception:
                break
            if isinstance(parsed, list):
                return _order_results(candidates, parsed)
            break

    for line in reversed(stdout.strip().splitlines()):
        if line.startswith('import_error '):
            detail = line[len('import_error '):]
            return [
                {'service': c['service'], 'ok': False,
                 'error': f'engine import failed: {detail}'}
                for c in candidates
            ]
        break

    return _partial_results(candidates, stdout, completed.returncode)


def _partial_results(candidates: list[dict], stdout, exit_code=None) -> list[dict]:
    """Recover whatever PROBE lines the child emitted before dying."""
    found = {}
    for line in (stdout or '').splitlines() if isinstance(stdout, str) else []:
        if not line.startswith('PROBE '):
            continue
        try:
            parsed = json.loads(line[len('PROBE '):])
        except Exception:
            continue
        if isinstance(parsed, dict) and parsed.get('service'):
            found[parsed['service']] = parsed

    results = []
    for candidate in candidates:
        name = candidate['service']
        if name in found:
            results.append(found[name])
        else:
            suffix = f' (child exit {exit_code})' if exit_code is not None else ''
            results.append(
                {'service': name, 'ok': False, 'error': f'probe did not run{suffix}'}
            )
    return results


def _order_results(candidates: list[dict], parsed: list[dict]) -> list[dict]:
    by_name = {item.get('service'): item for item in parsed if isinstance(item, dict)}
    return [
        by_name.get(c['service'], {'service': c['service'], 'ok': False,
                                   'error': 'missing probe result'})
        for c in candidates
    ]


def pick_service(candidates: list[dict], results: list[dict]) -> str | None:
    """First candidate (in attempt order) that probed successfully."""
    ok = {r.get('service') for r in results if r.get('ok')}
    for candidate in candidates:
        if candidate['service'] in ok:
            return candidate['service']
    return None


def format_report(results: list[dict]) -> str:
    lines = []
    for item in results:
        if item.get('ok'):
            lines.append(f"  [OK]   {item['service']}: {item.get('sample') or 'ok'}")
        else:
            lines.append(f"  [FAIL] {item['service']}: {item.get('error') or 'unavailable'}")
    return '\n'.join(lines)
