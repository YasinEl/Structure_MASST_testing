#!/bin/bash

datasette -p 5234 -h 0.0.0.0 \
--setting base_url /datasette/ \
--setting max_returned_rows 50000 \
--setting sql_time_limit_ms 60000 \
--setting max_csv_mb 0 /app/database/database.db