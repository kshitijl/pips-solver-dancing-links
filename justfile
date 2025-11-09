default:
    @just --list

check:
    uv run mypy --strict main.py mccw.py && uvx ty check

convert date='2025-11-04' difficulty='easy':
    uv run main.py data/{{date}}.json -d "{{difficulty}}"

count-solutions date difficulty='easy': (_generate-mcc date difficulty) (_mcc date difficulty "-v0")

print-solutions date difficulty='easy': (_generate-mcc date difficulty)
    @just _mcc {{date}} {{difficulty}} "-v1" | grep d_ | awk '{ $1=""; print }' | sort

_mcc date difficulty verbosity_flag:
    time ./mcc {{verbosity_flag}} generated-mcc-instances/{{date}}-{{difficulty}}

_generate-mcc date difficulty='easy':
    uv run main.py data/{{date}}.json -d "{{difficulty}}" > generated-mcc-instances/{{date}}-{{difficulty}}

download-todays:
    wget -P data/ https://www.nytimes.com/svc/pips/v1/$(date -I).json

download-for-date date:
    wget -P data/ https://www.nytimes.com/svc/pips/v1/{{date}}.json

[working-directory: 'notebooks']
notebook:
    uv run --with jupyter jupyter lab
