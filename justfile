default:
    @just --list

check:
    uv run mypy --strict main.py && uvx ty check

convert date='2025-11-04' difficulty='easy':
    uv run main.py data/{{date}}.json -d "{{difficulty}}"

solve date='2025-11-04' difficulty='easy':
    uv run main.py data/{{date}}.json -d "{{difficulty}}" > generated-mcc-instances/{{date}}-{{difficulty}} && time ./mcc -v0 generated-mcc-instances/{{date}}-{{difficulty}}

download-todays:
    wget -P data/ https://www.nytimes.com/svc/pips/v1/$(date -I).json

download-for-date date:
    wget -P data/ https://www.nytimes.com/svc/pips/v1/{{date}}.json

[working-directory: 'notebooks']
notebook:
    uv run --with jupyter jupyter lab
