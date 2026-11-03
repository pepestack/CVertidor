import pytest
from PIL import Image

from cvertidor.ingestion.loaders import DocumentLoadError, load_text, normalize_whitespace, read_input


@pytest.fixture
def png(tmp_path):
    path = tmp_path / "foto.png"
    Image.new("RGB", (40, 40), color=(200, 30, 30)).save(path)
    return path


def test_txt_y_md(tmp_path):
    (tmp_path / "cv.txt").write_text("﻿Ana   Pérez\n\n\n\nData Engineer  \n", encoding="utf-8")
    (tmp_path / "cv.md").write_text("# Ana\n- Python", encoding="utf-8")

    assert load_text(tmp_path / "cv.txt") == "Ana Pérez\n\nData Engineer"
    assert load_text(tmp_path / "cv.md") == "# Ana\n- Python"


def test_pdf_extrae_texto_e_ignora_imagenes(tmp_path, png):
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=12)
    pdf.cell(text="Sebastián Navarro - Data Engineer", new_x="LMARGIN", new_y="NEXT")
    pdf.image(str(png), w=20)
    pdf.cell(text="Habilidades: Python, SQL, DBT")
    pdf.add_page()
    pdf.cell(text="Educación: Universidad de Chile")
    path = tmp_path / "cv.pdf"
    pdf.output(str(path))

    text = load_text(path)

    assert "Sebastián Navarro - Data Engineer" in text
    assert "Habilidades: Python, SQL, DBT" in text
    assert "Educación: Universidad de Chile" in text


def test_pdf_escaneado_sin_texto_da_error_claro(tmp_path, png):
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.image(str(png), w=100)
    path = tmp_path / "escaneado.pdf"
    pdf.output(str(path))

    with pytest.raises(DocumentLoadError, match="escaneado"):
        load_text(path)


def test_docx_con_tabla_e_imagen(tmp_path, png):
    from docx import Document

    document = Document()
    document.add_heading("Ana Pérez", level=1)
    document.add_picture(str(png))
    document.add_paragraph("Ingeniera de datos con experiencia en Python.")
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Experiencia"
    table.rows[0].cells[1].text = "TechCorp (2021 - Presente)"
    merged = document.add_table(rows=1, cols=2)
    merged.rows[0].cells[0].merge(merged.rows[0].cells[1]).text = "Idiomas: Español, Inglés"
    path = tmp_path / "cv.docx"
    document.save(path)

    text = load_text(path)

    assert text.splitlines()[0] == "Ana Pérez"
    assert "Ingeniera de datos con experiencia en Python." in text
    assert "Experiencia | TechCorp (2021 - Presente)" in text
    # Una celda combinada se lee una sola vez
    assert text.count("Idiomas: Español, Inglés") == 1


def test_errores_de_archivo(tmp_path):
    with pytest.raises(DocumentLoadError, match="No existe"):
        load_text(tmp_path / "no_existe.pdf")

    (tmp_path / "cv.odt").write_bytes(b"x")
    with pytest.raises(DocumentLoadError, match="Formato no soportado"):
        load_text(tmp_path / "cv.odt")

    (tmp_path / "vacio.txt").write_text("   \n", encoding="utf-8")
    with pytest.raises(DocumentLoadError, match="no contiene texto"):
        load_text(tmp_path / "vacio.txt")


def test_read_input_acepta_ruta_o_texto(tmp_path):
    (tmp_path / "oferta.txt").write_text("Buscamos Data Engineer", encoding="utf-8")

    assert read_input(str(tmp_path / "oferta.txt")) == "Buscamos Data Engineer"
    assert read_input("Buscamos   Data Engineer\ncon Python") == "Buscamos Data Engineer\ncon Python"


def test_normalize_whitespace():
    assert normalize_whitespace("a  b\t c\n\n\n\nd") == "a b c\n\nd"
