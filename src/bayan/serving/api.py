"""Lab 7 + capstone starter: Bayan FastAPI service.

The final service should integrate the outputs of Labs 1-7. Keep this file as
orchestration; reusable logic belongs in the package modules.
"""
from fastapi import FastAPI

app = FastAPI(title="Bayan — Bilingual Citizen-Feedback Intelligence Service")


@app.get("/health")
def health():
    return {"status": "starter", "message": "Complete Labs 1-7 and wire startup canaries."}


@app.post("/v1/classify")
def classify(payload: dict):
    # TODO(Lab 7): shared preprocess -> winning classifier artefact -> response.
    raise NotImplementedError("Wire the Lab 7 classifier artefact")


@app.post("/v1/entities")
def entities(payload: dict):
    # TODO(Capstone): shared preprocessing/Arabic segmentation -> NER -> case fields.
    raise NotImplementedError("Wire the NER artefact")


@app.post("/v1/search")
def search(payload: dict):
    # TODO(Capstone): Lab 5 two-stage bilingual search.
    raise NotImplementedError("Wire the semantic-search component")


@app.post("/v1/analyse")
def analyse(payload: dict):
    # TODO(Capstone): one bilingual request -> classification + entities + similar cases.
    raise NotImplementedError("Assemble the Bayan capstone service")
