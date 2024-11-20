# Set default shell
set shell := ["bash", "-c"]

# # Data pipelines
# run-digital-data-etl-alex:
#     echo 'It is not supported anymore.'

run-digital-data-etl-maxime:
    python -m tools.run digital_data_etl_maxime_labonne.yaml

run-digital-data-etl-paul:
    python -m tools.run digital_data_etl_paul_iusztin.yaml

# # Compound command for running both ETL tasks
# run-digital-data-etl: run-digital-data-etl-maxime run-digital-data-etl-paul
#
# run-feature-engineering-pipeline:
#     python -m tools.run --no-cache --run-feature-engineering
#
# run-generate-instruct-datasets-pipeline:
#     python -m tools.run --no-cache --run-generate-instruct-datasets
#
# run-generate-preference-datasets-pipeline:
#     python -m tools.run --no-cache --run-generate-preference-datasets
#
# run-end-to-end-data-pipeline:
#     python -m tools.run --no-cache --run-end-to-end-data
#
# # Utility pipelines
# run-export-artifact-to-json-pipeline:
#     python -m tools.run --no-cache --run-export-artifact-to-json
#
# run-export-data-warehouse-to-json:
#     python -m tools.data_warehouse --export-raw-data
#
# run-import-data-warehouse-from-json:
#     python -m tools.data_warehouse --import-raw-data
#
# # Training pipelines
# run-training-pipeline:
#     python -m tools.run --no-cache --run-training
#
# run-evaluation-pipeline:
#     python -m tools.run --no-cache --run-evaluation
#
# # Inference
# call-rag-retrieval-module:
#     python -m tools.rag
#
# run-inference-ml-service:
#     uvicorn tools.ml_service:app --host 0.0.0.0 --port 8000 --reload
#
# call-inference-ml-service:
#     curl -X POST 'http://127.0.0.1:8000/rag' \
#         -H 'Content-Type: application/json' \
#         -d '{"query": "My name is Paul Iusztin. Could you draft a LinkedIn post discussing RAG systems? I am particularly interested in how RAG works and how it is integrated with vector DBs and LLMs."}'
#
# # Infrastructure
# # Local infrastructure
# local-docker-infrastructure-up:
#     docker compose up -d
#
# local-docker-infrastructure-down:
#     docker compose stop
#
local-zenml-server-down:
    zenml logout --local

# Conditional command for macOS vs other platforms
local-zenml-server-up:
    #!/usr/bin/env bash
    if [ "$(uname)" = "Darwin" ]; then
        export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
        zenml login --local
    else
        zenml login --local
    fi

# # Compound infrastructure commands
# local-infrastructure-up: local-docker-infrastructure-up local-zenml-server-down local-zenml-server-up
#
# local-infrastructure-down: local-docker-infrastructure-down local-zenml-server-down
#
# set-local-stack:
#     zenml stack set default
#
# set-aws-stack:
#     zenml stack set aws-stack
#
# set-asynchronous-runs:
#     zenml orchestrator update aws-stack --synchronous=False
#
# zenml-server-disconnect:
#     zenml disconnect
#
# # Settings
# export-settings-to-zenml:
#     python -m tools.run --export-settings
#
# delete-settings-zenml:
#     zenml secret delete settings
#
# # SageMaker
# create-sagemaker-role:
#     python -m llm_engineering.infrastructure.aws.roles.create_sagemaker_role
#
# create-sagemaker-execution-role:
#     python -m llm_engineering.infrastructure.aws.roles.create_execution_role
#
# deploy-inference-endpoint:
#     python -m llm_engineering.infrastructure.aws.deploy.huggingface.run
#
# test-sagemaker-endpoint:
#     python -m llm_engineering.model.inference.test
#
# delete-inference-endpoint:
#     python -m llm_engineering.infrastructure.aws.deploy.delete_sagemaker_endpoint
#
# # Docker
# build-docker-image:
#     docker buildx build --platform linux/amd64 -t llmtwin -f Dockerfile .
#
# run-docker-end-to-end-data-pipeline:
#     docker run --rm --network host --shm-size=2g --env-file .env llmtwin poetry poe --no-cache --run-end-to-end-data
#
# bash-docker-container:
#     docker run --rm -it --network host --env-file .env llmtwin bash
#
# # QA
# lint-check:
#     ruff check .
#
# format-check:
#     ruff format --check .
#
# lint-check-docker:
#     sh -c 'docker run --rm -i hadolint/hadolint < Dockerfile'
#
# gitleaks-check:
#     docker run -v .:/src zricethezav/gitleaks:latest dir /src/llm_engineering
#
# lint-fix:
#     ruff check --fix .
#
# format-fix:
#     ruff format .
#
# # Tests
# test:
#     #!/usr/bin/env bash
#     export ENV_FILE=.env.testing
#     pytest tests/
