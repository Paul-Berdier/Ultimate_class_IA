# main.py
from orchestrator import PredictionOrchestrator

def main():
    # --- PARAMÈTRES ---
    dataset_path = "bbc_dataset.csv"
    text_col = "text"
    label_col = "label"

    # --- INITIALISATION ---
    print("\n====== PIPELINE ULTIMATE_CLASS_IA ======")
    pipeline = PredictionOrchestrator(dataset_path, text_col, label_col)

    # --- MACHINE LEARNING ---
    ml_model = pipeline.run_ml()

    # --- DEEP LEARNING ---
    dl_model = pipeline.run_dl()

    # --- PREDICTION SUR NOUVELLES DONNÉES (Annexe 1) ---
    new_texts = [
        "Cricket Australia is set to begin the team’s pre-season later this month under a set of new training protocols devised for the safety of players amid the COVID-19 pandemic.",
        "Additionally, the microsite on Amazon.in highlights some of the key features of the Mi 10. It shows that the phone will come with a 108-megapixel primary camera that will have optical image stabilisation (OIS) and 8K video recording. The microsite also claims that the Mi 10 will carry the worlds fastest wireless charging and include Qualcomm Snapdragon 865 SoC. You can also expect a 3D curved TrueColor E3 AMOLED display with a Corning Gorilla Glass protection on top and stereo speakers.",
        "Having undergone a surgery for shoulder dislocation last month, young Australian pacer Jhye Richardson is hopeful of recovering from the recurring injury by the time cricketing action resumes. Cricket Australias chief medical officer Alex Kountouris sounded optimistic of the fast bowlers recovery process with respect to a comeback later this year.Its a lengthy surgery but it does give him an opportunity now that were not going to play until ... September, October, November or December ... hes obviously a chance with that, Kountouris told News Corp in an interaction."
    ]

    print("\n--- Prédiction ML ---")
    pred_ml = pipeline.predict_text(ml_model, new_texts)
    print(pred_ml)

    print("\n--- Prédiction DL ---")
    pred_dl = pipeline.predict_text(dl_model, new_texts)
    print(pred_dl)

if __name__ == "__main__":
    main()
