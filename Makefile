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
	python scripts/stats/processed_stats.py

bm25:
	python src/eval_bm25.py --dataset all --split val --max_history_len 20 --eval_mode global

# --- Large Datasets & Codabench Submission Targets ---
download-mind-large:
	python src/download.py --dataset mind_large

parse-mind-large:
	python src/parse_mind.py --dataset_type large --split all

features-mind-large:
	python src/feature_store.py --dataset mind_large --config configs/pipeline.yaml

submit-mind-bm25:
	python scripts/submissions/generate_mind_submission.py --dataset_type large --eval_dev --output_dir submissions

submit-mind-semantic:
	python scripts/submissions/generate_mind_semantic_submission.py --dataset_type large --eval_dev --output_dir submissions_semantic

submit-mind-reranker:
	python scripts/submissions/generate_mind_reranker_submission.py --dataset_type large --eval_dev --output_dir submissions/submissions_mind_reranker

download-ebnerd-large:
	python src/download.py --dataset ebnerd_large

parse-ebnerd-large:
	python src/parse_ebnerd.py --dataset_type large --split all

submit-ebnerd-bm25:
	python scripts/submissions/generate_ebnerd_submission.py --dataset_type large --eval_dev --output_dir submissions_ebnerd

submit-ebnerd-bm25-fulltext:
	python scripts/submissions/generate_ebnerd_submission.py --dataset_type large --include_body --eval_dev --output_dir submissions_ebnerd_fulltext

submit-ebnerd-semantic:
	python scripts/submissions/generate_ebnerd_semantic_submission.py --dataset_type large --eval_dev --output_dir submissions_ebnerd_semantic

submit-ebnerd-reranker:
	python scripts/submissions/generate_ebnerd_reranker_submission.py --dataset_type large --output_dir submissions/submissions_ebnerd_reranker

eval-ebnerd-reranker:
	python src/evaluate_ebnerd_large_validation.py --max_impressions 50000

submit-ebnerd-nrms-affinity:
	python scripts/submissions/generate_nrms_codabench_submission.py --dataset ebnerd

submit-mind-nrms-subcategory:
	python scripts/submissions/generate_nrms_codabench_submission.py --dataset mind

submit-nrms-both:
	python scripts/submissions/generate_nrms_codabench_submission.py --dataset both

test:
	pytest src/tests/ -v

clean:
	rm -rf data/interim/* data/processed/*

