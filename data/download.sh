#!/usr/bin/env bash

start_date="2025-08-18"
end_date="2025-11-04"

current_date="$start_date"
while [[ "$current_date" != $(gdate -I -d "$end_date + 1 day") ]]; do
    echo "$current_date"
    wget https://www.nytimes.com/svc/pips/v1/"$current_date".json
    current_date=$(gdate -I -d "$current_date + 1 day")
done
