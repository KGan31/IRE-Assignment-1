.PHONY: data download parse split features bm25 stats test clean

# One command, full rebuild from raw files.
data: download parse split features stats

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

stats:
	python processed_stats.py

bm25:
	python src/eval_bm25.py --dataset all --split val --max_history_len 20 --eval_mode global

stats:
	python processed_stats.py

test:
	pytest src/tests/ -v

clean:
	rm -rf data/interim/* data/processed/*
