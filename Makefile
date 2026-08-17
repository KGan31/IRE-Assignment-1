.PHONY: data download parse split features test clean

# One command, full rebuild from raw files.
data: download parse split features

download:
	python src/download.py --config configs/pipeline.yaml

parse:
	python src/parse_mind.py --split train
	python src/parse_mind.py --split dev
	python src/parse_ebnerd.py --split train
	python src/parse_ebnerd.py --split validation

split:
	python src/split.py --config configs/pipeline.yaml

features:
	python src/feature_store.py --config configs/pipeline.yaml

test:
	pytest src/tests/ -v

clean:
	rm -rf data/interim/* data/processed/*
