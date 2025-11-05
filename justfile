default:
    @just --list

convert difficulty='easy':
    uv run main.py data/2025-11-04.json -d "{{difficulty}}"

solve:
    uv run main.py data/2025-11-04.json > generated-mcc-instances/2025-11-04-easy && ../cover/bin/mcc generated-mcc-instances/2025-11-04-easy
