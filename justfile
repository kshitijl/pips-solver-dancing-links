default:
    @just --list

check:
    uv run mypy --strict main.py && uvx ty check

convert difficulty='easy':
    uv run main.py data/2025-11-04.json -d "{{difficulty}}"

solve difficulty='easy':
    uv run main.py data/2025-11-04.json -d "{{difficulty}}" > generated-mcc-instances/2025-11-04-{{difficulty}} && ../cover/bin/mcc generated-mcc-instances/2025-11-04-{{difficulty}}
