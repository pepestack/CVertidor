from cvertidor.schemas.cv import StandardCV


def test_valores_vacios_escritos_como_texto_se_normalizan():
    cv = StandardCV.model_validate({
        "candidate_name": "Ana",
        "email": "none",
        "phone": "N/A",
        "interests": "none",
        "certifications": "null",
        "technical_skills": "Python",
        "work_history": [{"company": "TechCorp", "role": "DE", "duration": "-", "responsibilities": "none"}],
        "education": [{"institution": "UChile", "degree": "Ing.", "year": "null"}],
    })

    assert cv.email is None and cv.phone is None
    assert cv.interests is None and cv.certifications is None
    assert cv.technical_skills == ["Python"]
    assert cv.work_history[0].duration is None
    assert cv.work_history[0].responsibilities == []
    assert cv.education[0].year is None


def test_valores_reales_no_se_alteran(sample_cv):
    assert StandardCV.model_validate(sample_cv.model_dump()) == sample_cv
