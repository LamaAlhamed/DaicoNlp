.PHONY: doctor lab1 lab2 lab3 lab4 lab5 lab6 lab7 test serve

doctor:
	python scripts/doctor.py

lab1:
	pytest tests/test_preprocessing.py tests/test_pii_recall.py -q

lab2:
	pytest tests/test_attention.py -q

lab3:
	pytest tests/test_model_data.py tests/test_ner_alignment.py tests/test_qa.py -q

lab4:
	pytest tests/test_arabic_normalize.py -q

lab5:
	pytest tests/test_search_contract.py -q

lab6:
	pytest tests/test_evaluation.py -q

lab7:
	pytest tests/test_serving_contract.py -q

test:
	pytest -q

serve:
	uvicorn bayan.serving.api:app --host 0.0.0.0 --port 8000
