PYTHON_VERSION       := 3.12
PYTHON               := uv run python

MISTRAL_SCRIPT       := src/data/mistral.py
OPENAI_SCRIPT        := src/data/openai.py
PUSH_DATASET_SCRIPT  := src/data/push_dataset.py
TRAIN_SCRIPT         := src/train/train.py
TRAIN_DIST_SCRIPT    := src/train/train_dist.py
INFERENCE_SCRIPT     := src/inference/inference.py
PUSH_MODEL_SCRIPT    := src/train/push_model.py
EVAL_SCRIPT          := src/eval/eval.py

# Mistral Data generation parameters
MISTRAL_MODEL        ?= mistral-large-2411
MISTRAL_REQUESTS     ?= 100000
NUM_WORKERS          ?= 4

# OpenAI Data generation parameters
OPENAI_MODEL         ?= gpt-3.5-turbo
OPENAI_REQUESTS      ?= 100000

# Push Dataset parameters
FOLDER_PATH          ?= dataset/
REPO_NAME            ?= MaxLSB/LeCarnet

# Train parameters
MODEL_CONFIG ?= 3M

# Push Model parameters
HF_REPO ?= MaxLSB/LeCarnet-3M
MODEL_DIR ?= LeCarnet-3M/model_weights/

# Inference parameters
MODEL_NAME ?= MaxLSB/LeCarnet-3M
MAX_NEW_TOKENS ?= 512
PROMPT ?= Il était une fois

# Evaluation parameters
EVAL_MODEL ?= MaxLSB/LeCarnet-3M
JUDGE_MODEL ?= mistral-large-2411

# Visible gpu
CUDA_VISIBLE_DEVICES ?= 0

.PHONY: env generate-mistral generate-openai push-dataset train train-dist inference eval push-model

env:
	@command -v uv >/dev/null 2>&1 || { \
		echo "Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	}
	@echo "Setting up environment..."
	@uv sync --python $(PYTHON_VERSION)
	@echo "Environment ready."

generate-mistral:
	$(PYTHON) $(MISTRAL_SCRIPT) \
		--model_name $(MISTRAL_MODEL) \
		--total_requests $(MISTRAL_REQUESTS) \
		--num_workers $(NUM_WORKERS)

generate-openai:
	$(PYTHON) $(OPENAI_SCRIPT) \
		--model_name $(OPENAI_MODEL) \
		--total_requests $(OPENAI_REQUESTS)

push-dataset:
	$(PYTHON) $(PUSH_DATASET_SCRIPT) \
		--folder_path $(FOLDER_PATH) \
		--repo_name $(REPO_NAME)

train:
	$(PYTHON) $(TRAIN_SCRIPT) \
		--model_config $(MODEL_CONFIG)

train-dist:
	@CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) deepspeed $(TRAIN_DIST_SCRIPT) \
		--model_config $(MODEL_CONFIG) \

inference:
	$(PYTHON) $(INFERENCE_SCRIPT) \
		--model_name $(MODEL_NAME) \
		--prompt "$(PROMPT)" \
		--max_new_tokens $(MAX_NEW_TOKENS)

eval:
	$(PYTHON) $(EVAL_SCRIPT) \
		--model_name $(EVAL_MODEL) \
		--judge_model_name $(JUDGE_MODEL) \

push-model:
	$(PYTHON) $(PUSH_MODEL_SCRIPT) \
		--repo_name $(HF_REPO) \
		--model_dir "$(MODEL_DIR)" 