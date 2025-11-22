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

_mccw date difficulty *flags:
    time uv run mccw.py generated-weighted-instances/{{date}}-{{difficulty}} {{flags}}

_mccwdlx date difficulty *flags:
    time ./cover/bin/mccw generated-weighted-instances/{{date}}-{{difficulty}} {{flags}}

_generate-mcc date difficulty='easy':
    uv run main.py data/{{date}}.json -d "{{difficulty}}" > generated-mcc-instances/{{date}}-{{difficulty}}

count-weighted date difficulty *flags: (_generate-weighted date difficulty) (_mccw date difficulty flags)

count-weighted-dlx date difficulty: (_generate-weighted date difficulty) (_mccwdlx date difficulty "-v0")

_generate-weighted date difficulty='easy':
    uv run main.py data/{{date}}.json -d "{{difficulty}}" -w > generated-weighted-instances/{{date}}-{{difficulty}}

download-todays:
    wget -P data/ https://www.nytimes.com/svc/pips/v1/$(date -I).json

download-for-date date:
    wget -P data/ https://www.nytimes.com/svc/pips/v1/{{date}}.json

[working-directory: 'notebooks']
notebook:
    uv run --with jupyter jupyter lab
