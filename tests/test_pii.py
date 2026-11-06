from pathlib import Path

from cvertidor.ingestion.pii import contains_placeholder, mask_pii, unmask

SAMPLE_CV = Path(__file__).parents[1] / "samples" / "cv_sebastian_navarro.txt"


def test_enmascara_contacto_y_restaura():
    text = (
        "Correo: ana.perez@empresa.cl. Tel: +56 9 6123 4567, fijo (2) 2345 6789, móvil (+56) 9 8765 4321. RUT 12.345.678-K\n"
        "LinkedIn: linkedin.com/in/ana-perez, portafolio https://ana.dev/proyectos."
    )
    result = mask_pii(text)

    for value in ["ana.perez@empresa.cl", "+56 9 6123 4567", "(2) 2345 6789", "(+56) 9 8765 4321", "12.345.678-K",
                  "linkedin.com/in/ana-perez", "https://ana.dev/proyectos"]:
        assert value not in result.text
        assert value in result.mapping.values()
    # La puntuación final de la frase se conserva fuera del marcador
    assert result.text.startswith("Correo: [EMAIL_1]. Tel:")
    assert result.text.endswith("[URL_2].")
    assert unmask(result.text, result.mapping) == text


def test_el_mismo_valor_usa_el_mismo_marcador():
    result = mask_pii("a@b.cl y otra vez a@b.cl, además c@d.cl")
    assert result.text == "[EMAIL_1] y otra vez [EMAIL_1], además [EMAIL_2]"


def test_no_enmascara_fechas_ni_cifras():
    text = ("Ingeniería (2015 - 2020). Marzo 2022 – Presente. Redujo tiempos en un 40%. Ventas de 485,000.50 USD.\n"
            "Diciembre 2022\n2015 – 2019\nDiciembre 2022 2015 – 2019")
    assert mask_pii(text).text == text


def test_cv_de_ejemplo_no_envia_contacto():
    text = SAMPLE_CV.read_text(encoding="utf-8")
    result = mask_pii(text)

    for value in ["sebastian.navarro.dev@emailficticio.cl", "+56 9 6123 4567",
                  "linkedin.com/in/snavarro-data-engineer", "github.com/snavarro-data"]:
        assert value in text
        assert value not in result.text
    # Los años y períodos del CV no se confunden con teléfonos
    assert "Marzo 2022 – Presente" in result.text
    assert "2012 – 2017" in result.text


def test_unmask_y_marcadores():
    assert unmask(None, {}) is None
    assert unmask("[EMAIL_9]", {}) == "[EMAIL_9]"
    assert contains_placeholder("ver [URL_1]")
    assert not contains_placeholder("sin datos")
