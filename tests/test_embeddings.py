from langchain_core.embeddings import DeterministicFakeEmbedding

from cvertidor.llm import embeddings as emb
from cvertidor.llm.providers import ModelSpec


def test_slug_de_colecciones():
    assert emb.embedding_slug(ModelSpec("openrouter", "nvidia/nemotron-3.5-embed-1b:free")) == "nemotron_3_5_embed_1b"
    # Se mantiene el nombre de la colección creada por el notebook
    assert emb.embedding_slug(ModelSpec("google", "models/gemini-embedding-2")) == "google"


def test_usa_el_proveedor_configurado(make_settings, monkeypatch):
    fake = DeterministicFakeEmbedding(size=8)
    monkeypatch.setattr(emb, "build_embeddings", lambda spec, settings: fake)

    info = emb.get_embeddings(make_settings(), fallback_factory=lambda: None)

    assert info.embeddings is fake
    # Por defecto Gemini, en la misma colección del notebook (cv_embeddings_google)
    assert info.slug == "google"
    assert info.model == "google:models/gemini-embedding-2"
    assert not info.is_fallback

    info = emb.get_embeddings(make_settings(embedding_model="openrouter:nvidia/nemotron-3.5-embed-1b:free"),
                              fallback_factory=lambda: None)
    assert info.slug == "nemotron_3_5_embed_1b"


def test_cae_a_minilm_si_el_proveedor_falla(make_settings, monkeypatch):
    class Broken:
        def embed_query(self, text):
            raise ConnectionError("sin red")

    local = DeterministicFakeEmbedding(size=8)
    monkeypatch.setattr(emb, "build_embeddings", lambda spec, settings: Broken())

    info = emb.get_embeddings(make_settings(), fallback_factory=lambda: local)

    assert info.embeddings is local
    assert info.slug == "minilm"
    assert info.is_fallback


def test_cae_a_minilm_sin_clave(make_settings):
    local = DeterministicFakeEmbedding(size=8)
    info = emb.get_embeddings(make_settings(google_api_key=None), fallback_factory=lambda: local)

    assert info.embeddings is local
    assert info.is_fallback
