from cvertidor.tools.query import get_candidate_profile, recall_preferences, search_candidate_experience
from cvertidor.tools.write import edit_cv, forget_preference, remember_preference, save_cv_files

# Solo el agente de refinamiento tiene herramientas; los demás roles no pueden llamar ninguna
REFINER_TOOLS = [
    search_candidate_experience,
    get_candidate_profile,
    recall_preferences,
    edit_cv,
    save_cv_files,
    remember_preference,
    forget_preference,
]
