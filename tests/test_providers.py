import httpx
import openai
import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from cvertidor.config import ROLES
from cvertidor.llm import providers
from cvertidor.llm.capabilities import pick_structured_method
from cvertidor.llm.providers import (
    OPENROUTER_BASE_URL,
    ModelConfigError,
    ModelSpec,
    build_chat_model,
    get_structured_model,
    message_text,
    resolve_structured_method,
    with_model_fallbacks,
)


def rate_limit_error() -> openai.RateLimitError:
    request = httpx.Request("POST", f"{OPENROUTER_BASE_URL}/chat/completions")
    return openai.RateLimitError("rate limited", response=httpx.Response(429, request=request), body=None)


@pytest.mark.parametrize(
    ("value", "provider", "model"),
    [
        ("openrouter:nvidia/nemotron-3.5-super-vl:free", "openrouter", "nvidia/nemotron-3.5-super-vl:free"),
        ("nvidia/nemotron-3.5-super-vl:free", "openrouter", "nvidia/nemotron-3.5-super-vl:free"),
        ("google:gemini-3.1-flash-lite", "google", "gemini-3.1-flash-lite"),
        ("  openrouter/free ", "openrouter", "openrouter/free"),
    ],
)
def test_parse_model_spec(value, provider, model):
    assert ModelSpec.parse(value) == ModelSpec(provider, model)


def test_openrouter_usa_chat_openai_con_la_url_de_openrouter(make_settings):
    model = build_chat_model(ModelSpec("openrouter", "a/b:free"), "generator", make_settings())

    assert type(model).__name__ == "ChatOpenAI"
    assert model.model_name == "a/b:free"
    assert model.openai_api_base == OPENROUTER_BASE_URL
    assert model.max_tokens == 8192
    assert model.temperature == 0.1
    assert model.max_retries == 3
    assert model.extra_body == {"reasoning": {"enabled": False}}


def test_razonamiento_solo_si_el_modelo_lo_soporta(make_settings):
    spec = ModelSpec("openrouter", "a/b")
    con = {"a/b": frozenset({"reasoning", "tools"})}
    sin = {"a/b": frozenset({"tools"})}

    assert providers.openrouter_extra_body(spec, "parser", con) == {"reasoning": {"enabled": False}}
    assert providers.openrouter_extra_body(spec, "refiner", con) == {"reasoning": {"effort": "low"}}
    assert providers.openrouter_extra_body(spec, "parser", sin) is None
    # Modelo fuera del catálogo (o catálogo caído): se envía y OpenRouter lo ignora si no aplica
    assert providers.openrouter_extra_body(spec, "parser", {}) == {"reasoning": {"enabled": False}}
    model = build_chat_model(spec, "parser", make_settings(), capabilities=sin)
    assert model.extra_body is None


def test_google_traduce_max_tokens(make_settings):
    model = build_chat_model(ModelSpec("google", "gemini-x"), "parser", make_settings(google_api_key="g-key"))

    assert type(model).__name__ == "ChatGoogleGenerativeAI"
    assert model.max_output_tokens == 8192


def test_google_traduce_el_razonamiento_a_thinking_level(make_settings):
    assert providers.google_thinking_level("parser") == "minimal"
    assert providers.google_thinking_level("generator") == "minimal"
    assert providers.google_thinking_level("refiner") == "low"

    model = build_chat_model(ModelSpec("google", "gemini-x"), "generator", make_settings())
    assert model.thinking_level == "minimal"


def test_por_defecto_todos_los_roles_usan_gemini(make_settings):
    settings = make_settings()
    for role in ROLES:
        specs, _ = providers.role_specs(role, settings)
        assert [str(s) for s in specs] == ["google:gemini-3.1-flash-lite", "google:gemini-3-flash-preview"]


def test_falta_la_clave_de_openrouter(make_settings):
    with pytest.raises(ModelConfigError, match="OPENROUTER_API_KEY"):
        build_chat_model(ModelSpec("openrouter", "a/b"), "parser", make_settings(openrouter_api_key=None))


def test_falta_la_clave_de_google(make_settings):
    with pytest.raises(ModelConfigError, match="GOOGLE_API_KEY"):
        build_chat_model(ModelSpec("google", "gemini-x"), "parser", make_settings(google_api_key=None))


@pytest.mark.parametrize(
    ("supported", "method"),
    [
        (frozenset({"structured_outputs", "tools", "response_format"}), "function_calling"),
        (frozenset({"structured_outputs", "response_format"}), "json_schema"),
        (frozenset({"tools", "response_format"}), "function_calling"),
        (frozenset({"response_format"}), "json_mode"),
        (frozenset(), "function_calling"),
        (None, "function_calling"),
    ],
)
def test_metodo_de_salida_segun_capacidades(supported, method):
    assert pick_structured_method(supported) == method


def test_metodo_forzado_por_configuracion_y_google_usa_el_predeterminado():
    caps = {"a/b": frozenset({"structured_outputs"}), "c/d": frozenset({"structured_outputs", "tools"})}
    assert resolve_structured_method(ModelSpec("openrouter", "a/b"), None, caps) == "json_schema"
    assert resolve_structured_method(ModelSpec("openrouter", "c/d"), None, caps) == "function_calling"
    assert resolve_structured_method(ModelSpec("openrouter", "a/b"), "json_mode", caps) == "json_mode"
    assert resolve_structured_method(ModelSpec("google", "gemini-x"), None, caps) is None


def test_fallback_ante_limite_de_uso():
    def primary(_):
        raise rate_limit_error()

    chain = with_model_fallbacks([RunnableLambda(primary), RunnableLambda(lambda _: "respaldo")])
    assert chain.invoke("hola") == "respaldo"


def test_sin_modelos_da_error():
    with pytest.raises(ModelConfigError):
        with_model_fallbacks([])


class _Salida(BaseModel):
    valor: str


class FakeChatModel:
    """Modelo mínimo que registra el método de salida estructurada que recibe"""

    def __init__(self, name: str, fails: bool, calls: list):
        self.name, self.fails, self.calls = name, fails, calls

    def with_structured_output(self, schema, **kwargs):
        self.calls.append((self.name, kwargs.get("method")))

        def run(_):
            if self.fails:
                raise rate_limit_error()
            return {"parsed": schema(valor=self.name), "raw": None, "parsing_error": None}

        return RunnableLambda(run)


def test_structured_model_usa_el_metodo_de_cada_modelo_y_cae_al_fallback(make_settings, monkeypatch):
    calls: list = []
    retries = {}

    def fake_build(spec, role, settings, max_retries=None, capabilities=None):
        retries[spec.model] = max_retries
        return FakeChatModel(spec.model, fails=spec.model == "principal", calls=calls)

    monkeypatch.setattr(providers, "build_chat_model", fake_build)
    settings = make_settings(parser_model="openrouter:principal", parser_model_fallbacks="openrouter:respaldo")
    caps = {"principal": frozenset({"structured_outputs"}), "respaldo": frozenset({"tools", "structured_outputs"})}

    result = get_structured_model("parser", _Salida, settings=settings, capabilities=caps).invoke("cv")

    assert result == _Salida(valor="respaldo")
    assert calls == [("principal", "json_schema"), ("respaldo", "function_calling")]
    # El principal no reintenta (cede al respaldo); el último usa los reintentos configurados
    assert retries == {"principal": 0, "respaldo": None}


def test_message_text_normaliza_bloques_de_contenido():
    assert message_text(AIMessage(content="  hola  ")) == "hola"
    blocks = AIMessage(content=[
        {"type": "reasoning", "reasoning": "pensando..."},
        {"type": "text", "text": "Parte 1"},
        "Parte 2",
        {"text": "Parte 3"},
    ])
    assert message_text(blocks) == "Parte 1\nParte 2\nParte 3"
    assert message_text("directo") == "directo"


def test_salida_invalida_pasa_al_siguiente_modelo(make_settings, monkeypatch):
    class InvalidThenValid(FakeChatModel):
        def with_structured_output(self, schema, **kwargs):
            if self.name == "principal":
                # Responde, pero sin cumplir el esquema (ej. 'none' donde va una lista)
                return RunnableLambda(lambda _: {"parsed": None, "raw": AIMessage(content=""), "parsing_error": "list_type"})
            return super().with_structured_output(schema, **kwargs)

    monkeypatch.setattr(providers, "build_chat_model",
                        lambda spec, role, settings, max_retries=None, capabilities=None: InvalidThenValid(spec.model, False, []))
    settings = make_settings(generator_model="principal", generator_model_fallbacks="respaldo")

    result = get_structured_model("generator", _Salida, include_raw=True, settings=settings, capabilities={}).invoke("x")
    assert result["parsed"] == _Salida(valor="respaldo")


def test_ningun_modelo_valido_lanza_error(make_settings, monkeypatch):
    class AlwaysInvalid(FakeChatModel):
        def with_structured_output(self, schema, **kwargs):
            return RunnableLambda(lambda _: {"parsed": None, "raw": AIMessage(content=""), "parsing_error": "list_type"})

    monkeypatch.setattr(providers, "build_chat_model",
                        lambda spec, role, settings, max_retries=None, capabilities=None: AlwaysInvalid(spec.model, False, []))
    settings = make_settings(generator_model="a", generator_model_fallbacks="b")

    with pytest.raises(providers.StructuredOutputError, match="list_type"):
        get_structured_model("generator", _Salida, settings=settings, capabilities={}).invoke("x")
